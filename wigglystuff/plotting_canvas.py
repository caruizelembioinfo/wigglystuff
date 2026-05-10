"""PlottingCanvas -- high-performance interactive plotting widget."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union
import uuid

import anywidget
import numpy as np
import traitlets

from ._colormaps import MISSING_COLOR

# Discrete palette used for categorical color encoding (up to 20 categories).
_CATEGORICAL_PALETTE = [
    "#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
    "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _to_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _encode_color(color: Any, n: int) -> tuple[bytes, str, list, list]:
    """Return (color_bytes, mode, palette, domain) for the given color spec.

    Modes:
      "constant"    -- single color, no per-point encoding
      "categorical" -- Int32Array of category codes; palette = list of hex colors
      "continuous"  -- Float32Array of values normalized to [0,1]; JS applies colormap
    """
    if color is None:
        return b"", "constant", [], []

    arr = np.asarray(color)

    if arr.dtype.kind in ("U", "S", "O"):
        unique, codes = np.unique(arr.astype(str), return_inverse=True)
        palette = [
            _CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)]
            for i in range(len(unique))
        ]
        return codes.astype(np.int32).tobytes(), "categorical", palette, [str(v) for v in unique]

    if arr.dtype.kind == "b":
        codes = arr.astype(np.int32)
        return codes.tobytes(), "categorical", _CATEGORICAL_PALETTE[:2], ["False", "True"]

    if arr.dtype.kind in ("i", "u"):
        unique = np.unique(arr[~np.isnan(arr.astype(float))])
        if len(unique) <= 20:
            code_map = {v: i for i, v in enumerate(unique)}
            sentinel = len(unique)
            codes = np.array([code_map.get(v, sentinel) for v in arr], dtype=np.int32)
            palette = [
                _CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)]
                for i in range(len(unique))
            ]
            return codes.tobytes(), "categorical", palette, [str(v) for v in unique]
        arr = arr.astype(np.float64)

    # Float / high-cardinality int → continuous (normalized to [0,1]; JS applies colormap)
    valid = arr[np.isfinite(arr)]
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0
    span = vmax - vmin if vmax != vmin else 1.0
    normalized = np.where(np.isfinite(arr), (arr - vmin) / span, -1.0).astype(np.float32)
    return normalized.tobytes(), "continuous", [], [vmin, vmax]


def _dtype_meta(dtype: Any) -> dict:
    """Infer minimal color metadata from a Polars dtype without reading any data.

    Returns a placeholder entry that will be replaced with accurate metadata
    the first time the column is selected and encoded.
    """
    name = type(dtype).__name__
    if name == "Boolean":
        return {"mode": "categorical", "palette": _CATEGORICAL_PALETTE[:2], "domain": ["False", "True"]}
    if name in (
        "Int8", "Int16", "Int32", "Int64", "Int128",
        "UInt8", "UInt16", "UInt32", "UInt64",
        "Float32", "Float64",
        "Date", "Datetime", "Duration", "Time",
    ):
        return {"mode": "continuous", "palette": [], "domain": []}
    # String, Categorical, Enum, Null, Unknown, etc.
    return {"mode": "categorical", "palette": [], "domain": []}


def _probe_color_meta(series: Any) -> dict:
    """Return color metadata for a Polars series without byte-encoding the array.

    Mirrors the logic of _encode_color so JS gets the correct mode/palette/domain
    for legend rendering before the actual bytes are sent.
    """
    arr = series.to_numpy(allow_copy=True)

    if arr.dtype.kind in ("U", "S", "O"):
        unique = np.unique(arr.astype(str))
        palette = [_CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)] for i in range(len(unique))]
        return {"mode": "categorical", "palette": palette, "domain": [str(v) for v in unique]}

    if arr.dtype.kind == "b":
        return {"mode": "categorical", "palette": _CATEGORICAL_PALETTE[:2], "domain": ["False", "True"]}

    if arr.dtype.kind in ("i", "u"):
        arr_f = arr.astype(float)
        unique = np.unique(arr_f[np.isfinite(arr_f)])
        if len(unique) <= 20:
            palette = [_CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)] for i in range(len(unique))]
            return {"mode": "categorical", "palette": palette, "domain": [str(int(v)) for v in unique]}
        arr = arr_f

    valid = arr[np.isfinite(arr)]
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0
    return {"mode": "continuous", "palette": [], "domain": [vmin, vmax]}


def _to_numpy(data: Any, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Convert DataFrame or dict to {col: ndarray}."""
    if hasattr(data, "to_numpy"):
        # polars DataFrame
        if hasattr(data, "to_pandas"):
            try:
                import polars as pl
                if isinstance(data, pl.DataFrame):
                    return {c: data[c].to_numpy() for c in (columns or data.columns)}
            except ImportError:
                pass
        # pandas DataFrame
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            cols = columns or list(data.columns)
            return {c: data[c].to_numpy() for c in cols}
    if isinstance(data, dict):
        return {k: np.asarray(v) for k, v in data.items()}
    raise TypeError(f"Unsupported data type: {type(data)}")


def _points_in_polygon(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    polygon: list,
) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test. Returns a boolean mask."""
    poly = np.asarray(polygon, dtype=np.float64)
    n_verts = len(poly)
    if n_verts < 3:
        return np.zeros(len(x_arr), dtype=bool)
    px = x_arr.astype(np.float64)
    py = y_arr.astype(np.float64)
    inside = np.zeros(len(px), dtype=bool)
    j = n_verts - 1
    for i in range(n_verts):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        straddles = (yi > py) != (yj > py)
        safe_denom = np.where(straddles, yj - yi, 1.0)
        x_cross = xi + (xj - xi) * (py - yi) / safe_denom
        inside ^= straddles & (px < x_cross)
        j = i
    return inside


class PlottingCanvas(anywidget.AnyWidget):
    """High-performance interactive plotting canvas with layered architecture.

    Uses WebGL (regl-scatterplot) for scatter rendering and D3 for axis overlays.
    Handles millions of points via binary Float32 transfer directly to the GPU.

    Layer types (Phase 1):
      - scatter: WebGL scatter plot with lasso/rectangle selection

    Examples:
        ```python
        import numpy as np
        from wigglystuff import PlottingCanvas

        rng = np.random.default_rng(42)
        n = 500_000
        x = rng.standard_normal(n).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        labels = rng.choice(["A", "B", "C"], n)

        canvas = PlottingCanvas(width=800, height=500)
        canvas.add_scatter(x, y, color=labels)
        canvas
        ```
    """

    _esm = Path(__file__).parent / "static" / "plotting-canvas.js"
    _css = Path(__file__).parent / "static" / "plotting-canvas.css"

    # --- layout ---
    width = traitlets.Int(800).tag(sync=True)
    height = traitlets.Int(500).tag(sync=True)

    # --- text labels ---
    title = traitlets.Unicode("").tag(sync=True)
    x_label = traitlets.Unicode("").tag(sync=True)
    y_label = traitlets.Unicode("").tag(sync=True)

    # --- binary data channels (Float32 / Int32 bytes) ---
    _x_data = traitlets.Bytes(b"").tag(sync=True)
    _y_data = traitlets.Bytes(b"").tag(sync=True)
    _color_data = traitlets.Bytes(b"").tag(sync=True)
    _size_data = traitlets.Bytes(b"").tag(sync=True)

    # --- data metadata ---
    _n_points = traitlets.Int(0).tag(sync=True)

    # --- color encoding metadata ---
    _color_mode = traitlets.Unicode("constant").tag(sync=True)
    _color_palette = traitlets.List([]).tag(sync=True)
    _color_domain = traitlets.List([]).tag(sync=True)
    _constant_color = traitlets.Unicode("#636efa").tag(sync=True)
    _missing_color = traitlets.Unicode(MISSING_COLOR).tag(sync=True)

    # --- size encoding ---
    _point_size = traitlets.Float(4.0).tag(sync=True)

    # --- layer registry ---
    _layers = traitlets.List([]).tag(sync=True)

    # --- frame-backed color column picker ---
    # Column names available for color encoding (empty when data loaded via add_scatter)
    _frame_columns = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    # Currently selected color column ("" = no color)
    _selected_color_col = traitlets.Unicode("").tag(sync=True)
    # Kept for backward compat; always b"" with lazy encoding
    _color_bank_data = traitlets.Bytes(b"").tag(sync=True)
    # Lightweight metadata for every column: {col: {mode, palette, domain}}
    _color_bank_meta = traitlets.Dict({}).tag(sync=True)
    # Lazy column encoding — Python pushes one column at a time on demand
    _col_chunk_col = traitlets.Unicode("").tag(sync=True)
    _col_chunk_data = traitlets.Bytes(b"").tag(sync=True)
    _col_chunk_ts = traitlets.Int(0).tag(sync=True)

    # --- view state (JS → Python) ---
    selected_indices = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    # List of [xmin, ymin, xmax, ymax] bboxes, one per selected region; [] when nothing selected
    _selection_bbox = traitlets.List(default_value=[]).tag(sync=True)
    # List of polygon vertex lists [[x,y],...] in data space, one per selected region
    _selection_polygons = traitlets.List(default_value=[]).tag(sync=True)

    # --- theme ---
    theme = traitlets.Unicode("auto").tag(sync=True)

    # --- action signals (Python → JS) ---
    _reset_view_ts = traitlets.Float(0.0).tag(sync=True)
    _redraw_trigger = traitlets.Int(0).tag(sync=True)

    def __init__(
        self,
        *,
        width: int = 800,
        height: int = 500,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        max_display: int = 2_000_000,
        theme: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Create an empty PlottingCanvas.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            title: Chart title displayed in the widget toolbar.
            x_label: Label for the x axis.
            y_label: Label for the y axis.
            max_display: Maximum number of display points before downsampling
                kicks in for :meth:`add_scatter_frame`.  Can be overridden
                per-call by passing ``max_display`` to that method.
            theme: ``"auto"`` (follow system preference), ``"light"``, or
                ``"dark"``.  Controls the canvas background colour and the
                CSS custom-property overrides for axes and toolbar.
            **kwargs: Forwarded to ``anywidget.AnyWidget``.
        """
        super().__init__(
            width=width,
            height=height,
            title=title,
            x_label=x_label,
            y_label=y_label,
            theme=theme,
            **kwargs,
        )
        self._max_display = max_display
        # Frame-backed source (not synced; used for bbox queries and color re-encoding)
        self._source_lf = None
        self._source_x_col: str | None = None
        self._source_y_col: str | None = None
        self._display_df = None   # downsampled DataFrame kept for column re-encoding
        self._col_cache: dict[str, bytes] = {}  # Python-side encoded column cache
        self._sample_indices: np.ndarray | None = None  # row indices used for downsampling

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_column(self, col: str) -> None:
        """Encode one color column and push it to JS.

        If the column was not eagerly loaded into ``_display_df``, it is read
        from the source LazyFrame on demand using the stored sample indices.
        After encoding, ``_color_bank_meta`` is updated with accurate metadata
        for the column (replacing the dtype-inferred placeholder).  The JS
        ``change:_color_bank_meta`` handler does NOT clear ``colCache``, so
        updating the metadata here is safe.
        """
        if not col:
            return

        in_display = self._display_df is not None and col in self._display_df.columns
        has_source = self._source_lf is not None

        if not in_display and not has_source:
            return

        if col not in self._col_cache:
            if in_display:
                arr = self._display_df[col].to_numpy(allow_copy=True)
            else:
                import polars as pl
                if self._sample_indices is not None:
                    arr = (
                        self._source_lf
                        .select(pl.col(col))
                        .select(pl.all().gather(self._sample_indices))
                        .collect()[col]
                        .to_numpy(allow_copy=True)
                    )
                else:
                    arr = (
                        self._source_lf
                        .select(pl.col(col))
                        .collect()[col]
                        .to_numpy(allow_copy=True)
                    )

            color_bytes, mode, palette, domain = _encode_color(arr, len(arr))
            self._col_cache[col] = color_bytes

            # Push accurate metadata back so JS legend/legend mode is correct.
            # Updating _color_bank_meta is safe here because the JS handler
            # for that event no longer clears colCache (only _frame_columns does).
            updated_meta = dict(self._color_bank_meta)
            updated_meta[col] = {
                "mode": mode,
                "palette": palette,
                "domain": [
                    float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                    for v in domain
                ],
            }
            self._color_bank_meta = updated_meta

        self._col_chunk_col = col
        self._col_chunk_data = self._col_cache[col]
        self._col_chunk_ts = self._col_chunk_ts + 1

    @traitlets.observe("_selected_color_col")
    def _on_selected_color_col_change(self, change: dict) -> None:
        """Send encoded column data whenever the JS color picker changes."""
        self._send_column(change["new"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_scatter(
        self,
        x: Any,
        y: Any,
        *,
        color: Any = None,
        size: Union[float, Any] = 4.0,
        color_missing: str = MISSING_COLOR,
        layer_id: Optional[str] = None,
        x_label: str = "",
        y_label: str = "",
    ) -> str:
        """Add (or replace) a scatter layer on the canvas.

        Args:
            x: 1-D array-like of x coordinates (converted to float32).
            y: 1-D array-like of y coordinates (converted to float32).
            color: Per-point color encoding. Accepts:
                - ``None``: single default color.
                - Array of strings/objects: categorical palette.
                - Array of booleans: 2-category categorical.
                - Array of ints: categorical if ≤20 unique values, else gradient.
                - Array of floats: continuous gradient (colormap set in widget toolbar).
            size: Point radius in pixels, or array of per-point sizes.
            color_missing: Hex color used for NaN / null values.
            layer_id: Optional stable ID. If a layer with this ID already exists
                it is replaced. A UUID is generated when omitted.
            x_label: X axis label (overrides canvas-level label if set).
            y_label: Y axis label (overrides canvas-level label if set).

        Returns:
            The layer ID string.
        """
        x_arr = _to_float32(np.asarray(x).ravel())
        y_arr = _to_float32(np.asarray(y).ravel())
        n = len(x_arr)
        if len(y_arr) != n:
            raise ValueError(f"x and y must have the same length ({n} vs {len(y_arr)})")

        color_bytes, color_mode, color_palette, color_domain = _encode_color(color, n)

        if np.ndim(size) == 0:
            size_bytes = b""
            point_size = float(size)
        else:
            size_arr = _to_float32(np.asarray(size).ravel())
            if len(size_arr) != n:
                raise ValueError(f"size must have the same length as x ({n})")
            size_bytes = size_arr.tobytes()
            point_size = float(np.nanmedian(size_arr))

        lid = layer_id or str(uuid.uuid4())[:8]
        layers = [l for l in self._layers if l.get("id") != lid]
        layers.append({"id": lid, "type": "scatter", "visible": True})

        if x_label:
            self.x_label = x_label
        if y_label:
            self.y_label = y_label

        self._n_points = n
        self._x_data = x_arr.tobytes()
        self._y_data = y_arr.tobytes()
        self._color_data = color_bytes
        self._size_data = size_bytes
        self._color_mode = color_mode
        self._color_palette = color_palette
        self._color_domain = color_domain
        self._missing_color = color_missing
        self._point_size = point_size
        self._layers = layers
        return lid

    def add_scatter_frame(
        self,
        frame: Any,
        x: str,
        y: str,
        *,
        color: Optional[str] = None,
        size: Union[float, str] = 4.0,
        color_columns: Optional[List[str]] = None,
        color_missing: str = MISSING_COLOR,
        layer_id: Optional[str] = None,
        x_label: str = "",
        y_label: str = "",
        max_display: Optional[int] = None,
    ) -> str:
        """Add a scatter layer from a Polars LazyFrame or DataFrame.

        If the frame has more rows than ``max_display``, a random sample of
        that size is drawn for display only.  When the user makes a selection,
        :attr:`selected_data` queries the **full** original frame using the
        bounding box of the selection.

        The color picker lists every column that was read from disk.  By default
        (``color_columns=None``) that is all columns in the frame.  Pass
        ``color_columns`` to limit which columns are read, which speeds up
        loading significantly for wide files such as parquet with many columns.

        Args:
            frame: A ``polars.LazyFrame`` or ``polars.DataFrame``.
            x: Column name for x coordinates.
            y: Column name for y coordinates.
            color: Column name for the initial color encoding.
                Pass ``None`` for a single default color.
            size: Scalar point radius, or column name for per-point sizes.
            color_columns: Extra columns to load from disk and make available
                in the color picker beyond ``x``, ``y``, and ``color``.
                For wide parquet files, specify only the columns you care about
                to avoid reading the rest from disk.
            color_missing: Hex color for NaN / null values.
            layer_id: Optional stable layer ID.
            x_label: X axis label.
            y_label: Y axis label.
            max_display: Maximum number of points to render.  Overrides the
                value set on the widget constructor for this call.

        Returns:
            The layer ID string.
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError("polars is required for add_scatter_frame") from e

        lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        effective_max = max_display if max_display is not None else self._max_display

        # Determine columns available in the color picker.
        # This is schema-only — no disk read — so it's fast even for wide parquets.
        schema = lf.collect_schema()
        all_schema_cols = schema.names()
        if color_columns is not None:
            avail_cols = [c for c in color_columns if c in schema]
        else:
            avail_cols = all_schema_cols

        # Only eagerly gather the columns required for the initial display.
        # All other color columns are loaded lazily from the parquet on demand.
        eager: list[str] = [x, y]
        if color:
            eager.append(color)
        if isinstance(size, str):
            eager.append(size)
        seen_e: set = set()
        eager_cols = [c for c in eager if not (c in seen_e or seen_e.add(c))]  # type: ignore[func-returns-value]

        n_total: int = lf.select(pl.len()).collect().item()

        if n_total > effective_max:
            indices = np.random.default_rng(42).choice(n_total, size=effective_max, replace=False)
            display_df = lf.select(eager_cols).select(pl.all().gather(indices)).collect()
        else:
            indices = None
            display_df = lf.select(eager_cols).collect()

        x_arr = display_df[x].to_numpy()
        y_arr = display_df[y].to_numpy()
        color_arr = display_df[color].to_numpy() if color else None
        size_val: Union[float, Any] = display_df[size].to_numpy() if isinstance(size, str) else size

        self._source_lf = lf
        self._source_x_col = x
        self._source_y_col = y
        self._display_df = display_df
        self._sample_indices = indices

        # Metadata bank: accurate for eagerly-loaded columns, dtype-inferred for the
        # rest. The dtype guess will be corrected with accurate info the first time
        # a column is selected (via _send_column updating _color_bank_meta).
        bank_meta: dict = {"": {"mode": "constant", "palette": [], "domain": []}}
        for col_name in avail_cols:
            if col_name in display_df.columns:
                bank_meta[col_name] = _probe_color_meta(display_df[col_name])
            else:
                bank_meta[col_name] = _dtype_meta(schema[col_name])

        self._col_cache = {}
        self._color_bank_data = b""
        self._color_bank_meta = bank_meta
        self._frame_columns = avail_cols
        self._selected_color_col = color or ""
        # Always eagerly send the initial color column so the first render is colored.
        # (@observe may or may not fire above depending on whether the value changed.)
        if color:
            self._send_column(color)

        return self.add_scatter(
            x_arr, y_arr,
            color=color_arr,
            size=size_val,
            color_missing=color_missing,
            layer_id=layer_id,
            x_label=x_label,
            y_label=y_label,
        )

    def set_data(
        self,
        *,
        x: Any = None,
        y: Any = None,
        color: Any = None,
        size: Any = None,
    ) -> None:
        """Update scatter data in place without recreating the widget.

        Only the provided arguments are updated; omitted ones stay unchanged.

        Args:
            x: New x coordinates (must match current point count if y omitted).
            y: New y coordinates.
            color: New color encoding (same rules as :meth:`add_scatter`).
            size: New size encoding.
        """
        n = self._n_points

        if x is not None:
            x_arr = _to_float32(np.asarray(x).ravel())
            n = len(x_arr)
            self._x_data = x_arr.tobytes()

        if y is not None:
            y_arr = _to_float32(np.asarray(y).ravel())
            if x is None and len(y_arr) != n:
                raise ValueError(f"y length {len(y_arr)} must match existing n_points {n}")
            self._y_data = y_arr.tobytes()

        if x is not None or y is not None:
            self._n_points = n

        if color is not None:
            color_bytes, color_mode, color_palette, color_domain = _encode_color(color, n)
            self._color_data = color_bytes
            self._color_mode = color_mode
            self._color_palette = color_palette
            self._color_domain = color_domain

        if size is not None:
            if np.ndim(size) == 0:
                self._size_data = b""
                self._point_size = float(size)
            else:
                size_arr = _to_float32(np.asarray(size).ravel())
                self._size_data = size_arr.tobytes()

        self._redraw_trigger += 1

    def reset_view(self) -> None:
        """Reset camera to fit all points in view."""
        import time
        self._reset_view_ts = time.monotonic()

    @property
    def selected_data(self) -> dict:
        """Return data for the currently selected region.

        When data was loaded with :meth:`add_scatter_frame`, the full original
        frame is queried with precision:

        - **Polygon mode** (default for lasso/box selections): exact point-in-polygon
          test using the actual drawn shape — no false positives between regions.
        - **Bbox fallback**: axis-aligned bounding box used when polygon data is
          absent (e.g., programmatic selection).
        - **Index fallback**: display-level selection when no LazyFrame is attached.

        Returns a dict with:
          - ``"x"``, ``"y"``: float32 arrays of selected coordinates.
          - ``"frame"``: a ``polars.DataFrame`` of the matching rows (only when
            data was loaded via :meth:`add_scatter_frame`, else ``None``).
          - ``"polygon"``: list of per-region polygon vertex lists, or ``None``.
          - ``"bbox"``: list of per-region ``[xmin, ymin, xmax, ymax]`` or ``None``.
        """
        if self._source_lf is not None and self._selection_polygons:
            import polars as pl
            # Stage 1: combined bbox pre-filter to minimise rows collected from disk.
            expr = None
            for poly in self._selection_polygons:
                poly_arr = np.asarray(poly)
                if len(poly_arr) < 3:
                    continue
                xmin = float(poly_arr[:, 0].min())
                xmax = float(poly_arr[:, 0].max())
                ymin = float(poly_arr[:, 1].min())
                ymax = float(poly_arr[:, 1].max())
                region = (
                    pl.col(self._source_x_col).cast(pl.Float32).is_between(xmin, xmax) &
                    pl.col(self._source_y_col).cast(pl.Float32).is_between(ymin, ymax)
                )
                expr = region if expr is None else (expr | region)
            if expr is not None:
                candidates = self._source_lf.filter(expr).collect()
                cx = candidates[self._source_x_col].cast(pl.Float32).to_numpy()
                cy = candidates[self._source_y_col].cast(pl.Float32).to_numpy()
                # Stage 2: exact PIP test per region, OR the masks together.
                mask = np.zeros(len(cx), dtype=bool)
                for poly in self._selection_polygons:
                    if len(poly) >= 3:
                        mask |= _points_in_polygon(cx, cy, poly)
                result = candidates.filter(pl.Series(mask))
                return {
                    "x": result[self._source_x_col].cast(pl.Float32).to_numpy(),
                    "y": result[self._source_y_col].cast(pl.Float32).to_numpy(),
                    "frame": result,
                    "polygon": [list(p) for p in self._selection_polygons],
                    "bbox": list(self._selection_bbox) if self._selection_bbox else None,
                }

        if self._source_lf is not None and self._selection_bbox:
            import polars as pl
            expr = None
            for xmin, ymin, xmax, ymax in self._selection_bbox:
                region = (
                    pl.col(self._source_x_col).cast(pl.Float32).is_between(xmin, xmax) &
                    pl.col(self._source_y_col).cast(pl.Float32).is_between(ymin, ymax)
                )
                expr = region if expr is None else (expr | region)
            result = self._source_lf.filter(expr).collect()
            return {
                "x": result[self._source_x_col].cast(pl.Float32).to_numpy(),
                "y": result[self._source_y_col].cast(pl.Float32).to_numpy(),
                "frame": result,
                "polygon": None,
                "bbox": list(self._selection_bbox),
            }

        # Index fallback: return display-level selected points.
        idx = np.array(self.selected_indices, dtype=np.int32)
        if len(idx) == 0 or not self._x_data:
            return {"x": np.array([]), "y": np.array([]), "frame": None, "polygon": None, "bbox": None}
        x_all = np.frombuffer(self._x_data, dtype=np.float32)
        y_all = np.frombuffer(self._y_data, dtype=np.float32)
        return {"x": x_all[idx], "y": y_all[idx], "frame": None, "polygon": None, "bbox": None}
