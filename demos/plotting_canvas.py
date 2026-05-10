# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.19.7",
#     "numpy==2.4.4",
#     "polars>=1.0",
#     "wigglystuff==0.4.1",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from wigglystuff import PlottingCanvas

    return PlottingCanvas, mo, np, pl


@app.cell
def _(PlottingCanvas, mo, np, pl):
    rng = np.random.default_rng(42)
    n = 2_000_000

    lf = pl.LazyFrame({
        "x": rng.standard_normal(n).astype("float32"),
        "y": rng.standard_normal(n).astype("float32"),
        "cluster": rng.choice(["A", "B", "C", "D", "E"], n),
        "density_hint": (rng.standard_normal(n) ** 2).astype("float32"),
        "size_class": rng.integers(1, 6, n).astype("int32"),
    })

    widget = PlottingCanvas(width=800, height=480, title="Gaussian Blob")
    widget.add_scatter_frame(lf, x="x", y="y", color="cluster", x_label="X", y_label="Y")
    canvas = mo.ui.anywidget(widget)
    canvas
    return (canvas,)


@app.cell
def _(canvas, mo):
    sel = canvas.selected_indices
    bboxes = canvas.widget._selection_bbox
    if sel:
        region_str = f"{len(bboxes)} region{'s' if len(bboxes) != 1 else ''}"
        msg = mo.callout(
            mo.md(
                f"**{len(sel)} display points selected** across {region_str}. "
                f"Use `canvas.widget.selected_data` for full-resolution results."
            ),
            kind="info",
        )
    else:
        msg = mo.callout(
            mo.md(
                "Activate **lasso** or **box** in the ⚙ toolbar, then drag to select. "
                "**Hold Ctrl** while dragging to add a second region to the selection."
            ),
            kind="neutral",
        )
    msg
    return


if __name__ == "__main__":
    app.run()
