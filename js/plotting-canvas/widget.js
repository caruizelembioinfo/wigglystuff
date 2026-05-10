import createScatterplot from "regl-scatterplot";
import { axisBottom, axisLeft } from "d3-axis";
import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { COLORMAPS } from "./colormaps.js";

const MARGIN = { top: 16, right: 16, bottom: 48, left: 56 };
const BINS = 100;
const COLORMAP_NAMES = ["viridis", "plasma", "inferno", "magma", "greys", "fire"];

// --- typed-array helpers ---
function toFloat32(dv) {
  if (!dv || dv.byteLength === 0) return null;
  return new Float32Array(dv.buffer, dv.byteOffset, dv.byteLength / 4);
}

function toInt32(dv) {
  if (!dv || dv.byteLength === 0) return null;
  return new Int32Array(dv.buffer, dv.byteOffset, dv.byteLength / 4);
}

// Normalize arr to NDC [-1, 1] using the given data-space [lo, hi] extent.
function normalizeArr(arr, lo, hi) {
  const span = hi - lo || 1;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = (arr[i] - lo) / span * 2 - 1;
  return out;
}

// --- density computation ---
function gaussianKernel(sigma) {
  const r = Math.ceil(sigma * 3);
  const k = new Float64Array(2 * r + 1);
  let s = 0;
  for (let i = 0; i <= 2 * r; i++) { k[i] = Math.exp(-0.5 * ((i - r) / sigma) ** 2); s += k[i]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return k;
}

function blur1D(src, w, h, k, horiz) {
  const r = (k.length - 1) >> 1;
  const dst = new Float64Array(src.length);
  for (let row = 0; row < h; row++) {
    for (let col = 0; col < w; col++) {
      let v = 0;
      for (let ki = 0; ki < k.length; ki++) {
        const d = ki - r;
        const c2 = horiz ? Math.max(0, Math.min(w - 1, col + d)) : col;
        const r2 = horiz ? row : Math.max(0, Math.min(h - 1, row + d));
        v += src[r2 * w + c2] * k[ki];
      }
      dst[row * w + col] = v;
    }
  }
  return dst;
}

function computeDensity(x, y, sigma) {
  const n = x.length;
  let xMin = x[0], xMax = x[0], yMin = y[0], yMax = y[0];
  for (let i = 1; i < n; i++) {
    if (x[i] < xMin) xMin = x[i];
    if (x[i] > xMax) xMax = x[i];
    if (y[i] < yMin) yMin = y[i];
    if (y[i] > yMax) yMax = y[i];
  }
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const H = new Float64Array(BINS * BINS);
  for (let i = 0; i < n; i++) {
    const xi = Math.min(Math.floor((x[i] - xMin) / xRange * BINS), BINS - 1);
    const yi = Math.min(Math.floor((y[i] - yMin) / yRange * BINS), BINS - 1);
    H[yi * BINS + xi]++;
  }

  const k = gaussianKernel(sigma);
  const blurred = blur1D(blur1D(H, BINS, BINS, k, true), BINS, BINS, k, false);

  let dMax = 0;
  for (let i = 0; i < blurred.length; i++) if (blurred[i] > dMax) dMax = blurred[i];

  const density = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const xi = Math.min(Math.floor((x[i] - xMin) / xRange * BINS), BINS - 1);
    const yi = Math.min(Math.floor((y[i] - yMin) / yRange * BINS), BINS - 1);
    density[i] = dMax > 0 ? blurred[yi * BINS + xi] / dMax : 0;
  }
  return { density, dMax, xMin, xMax, yMin, yMax };
}

// --- color bank (metadata only; column bytes arrive lazily via _col_chunk_*) ---
function loadColBank(model) {
  const meta = model.get("_color_bank_meta");
  if (!meta || Object.keys(meta).length <= 1) return null;
  return { meta };
}

function getColorFromBank(colCache, bank, col, missingColor, lut) {
  if (!bank) return null;
  const entry = bank.meta[col];
  if (!entry || entry.mode === "constant") return null;
  const buf = colCache.get(col);
  if (!buf) return null;  // bytes not yet received from Python
  if (entry.mode === "categorical") {
    const zInt = new Int32Array(buf);
    const zFloat = new Float32Array(zInt);
    return { zArr: zFloat, colorBy: "valueZ", pointColor: [...entry.palette, missingColor], zDataType: "categorical" };
  }
  // continuous
  const zArr = new Float32Array(buf);
  return { zArr, colorBy: "valueZ", pointColor: lut, zDataType: "continuous" };
}

// --- color configuration ---
function getColorProps(model, displayMode, colormap, cachedDensity, colBank, colCache) {
  const lut = COLORMAPS[colormap] || COLORMAPS.viridis;
  const missingColor = model.get("_missing_color");

  if (displayMode === "density" && cachedDensity) {
    return { zArr: cachedDensity, colorBy: "valueZ", pointColor: lut, zDataType: "continuous" };
  }

  // Use lazily-received column bytes when frame data is loaded
  if (colBank && displayMode === "data") {
    const col = model.get("_selected_color_col");
    const fromBank = getColorFromBank(colCache, colBank, col, missingColor, lut);
    if (fromBank) return fromBank;
    // "" or bytes not yet arrived → constant color (will recolor when _col_chunk_ts fires)
    return { zArr: null, colorBy: null, pointColor: [model.get("_constant_color")], zDataType: undefined };
  }

  const mode = model.get("_color_mode");

  if (mode === "constant") {
    return { zArr: null, colorBy: null, pointColor: [model.get("_constant_color")], zDataType: undefined };
  }

  if (mode === "categorical") {
    const zArr = toInt32(model.get("_color_data"));
    const zFloat = zArr ? new Float32Array(zArr) : null;
    const palette = model.get("_color_palette");
    return { zArr: zFloat, colorBy: "valueZ", pointColor: [...palette, missingColor], zDataType: "categorical" };
  }

  // continuous — JS applies colormap to Python-normalized [0,1] values
  return { zArr: toFloat32(model.get("_color_data")), colorBy: "valueZ", pointColor: lut, zDataType: "continuous" };
}

function getSizeProps(model) {
  const wArr = toFloat32(model.get("_size_data"));
  if (wArr) return { wArr, sizeBy: "valueW", pointSize: undefined };
  return { wArr: null, sizeBy: null, pointSize: model.get("_point_size") };
}

// --- legend ---
function renderLegend(legendEl, model, displayMode, colormap, colBank, densityMax, callbacks = {}) {
  const { activeCategories, onCategoryToggle, numericFilterRange, onNumericRangeChange } = callbacks;

  let mode, domain, palette;
  if (displayMode === "density") {
    mode = "continuous";
    domain = [0, densityMax];
    palette = [];
  } else if (colBank) {
    const sel = model.get("_selected_color_col");
    const entry = (sel && colBank.meta[sel]) || { mode: "constant", palette: [], domain: [] };
    ({ mode, palette, domain } = entry);
  } else {
    mode = model.get("_color_mode");
    domain = model.get("_color_domain");
    palette = model.get("_color_palette");
  }

  const fmtNum = v => {
    if (Math.abs(v) >= 1e6) return (v / 1e6).toPrecision(3) + "M";
    if (Math.abs(v) >= 1e3) return (v / 1e3).toPrecision(3) + "k";
    return parseFloat(v.toPrecision(3)).toString();
  };

  if (mode === "categorical" && palette && palette.length > 0) {
    const nTotal = domain.length;
    const shown = domain.slice(0, 20);
    legendEl.innerHTML =
      `<div class="pc-legend-inner">` +
      shown.map((label, i) => {
        const active = activeCategories === null || activeCategories.has(i);
        return `<span class="pc-legend-item${active ? "" : " pc-legend-item-inactive"}" data-idx="${i}" style="--swatch:${palette[i] || "#888"}">` +
               `<span class="pc-legend-swatch"></span>${label}</span>`;
      }).join("") +
      (nTotal > 20 ? `<span style="color:#999;font-size:10px;">+${nTotal - 20} more</span>` : "") +
      `</div>`;
    if (onCategoryToggle) {
      legendEl.querySelectorAll(".pc-legend-item").forEach(el => {
        el.addEventListener("click", e => {
          onCategoryToggle(parseInt(el.dataset.idx), nTotal, e.detail);
        });
      });
    }
  } else if (mode === "continuous" && Array.isArray(domain) && domain.length === 2) {
    const lut = COLORMAPS[colormap] || COLORMAPS.viridis;
    const n = lut.length;
    const stops = [0, n >> 2, n >> 1, (3 * n) >> 2, n - 1].map(i => lut[Math.min(i, n - 1)]);
    const grad = `linear-gradient(to right, ${stops.join(",")})`;
    const [vmin, vmax] = domain;

    if (displayMode === "density") {
      legendEl.innerHTML =
        `<div class="pc-legend-inner pc-legend-continuous">` +
        `<span>0</span>` +
        `<div class="pc-legend-grad" style="background:${grad};"></div>` +
        `<span>${fmtNum(vmax)} pts/bin_cell</span></div>`;
    } else if (onNumericRangeChange) {
      // z values in the bank are normalized [0,1] by Python; slider shows raw domain values.
      // numericFilterRange stores normalized [0,1]; convert to/from raw for display.
      const span = vmax > vmin ? vmax - vmin : 1;
      const step = span / 1000;
      const loVal = numericFilterRange ? numericFilterRange[0] * span + vmin : vmin;
      const hiVal = numericFilterRange ? numericFilterRange[1] * span + vmin : vmax;
      legendEl.innerHTML =
        `<div class="pc-legend-inner pc-range-container">` +
        `<div class="pc-range-gradient">` +
        `<div class="pc-legend-grad" style="background:${grad};"></div>` +
        `<input type="range" class="pc-range-input pc-range-lo" min="${vmin}" max="${vmax}" step="${step}" value="${loVal}">` +
        `<input type="range" class="pc-range-input pc-range-hi" min="${vmin}" max="${vmax}" step="${step}" value="${hiVal}">` +
        `</div>` +
        `<div class="pc-range-val-row">` +
        `<span class="pc-range-lo-label">${fmtNum(loVal)}</span>` +
        `<span class="pc-range-hi-label">${fmtNum(hiVal)}</span>` +
        `</div>` +
        `</div>`;
      const loInput = legendEl.querySelector(".pc-range-lo");
      const hiInput = legendEl.querySelector(".pc-range-hi");
      const loLabel = legendEl.querySelector(".pc-range-lo-label");
      const hiLabel = legendEl.querySelector(".pc-range-hi-label");
      loInput.addEventListener("mousedown", () => { loInput.style.zIndex = "3"; hiInput.style.zIndex = "2"; });
      hiInput.addEventListener("mousedown", () => { hiInput.style.zIndex = "3"; loInput.style.zIndex = "2"; });
      loInput.addEventListener("input", () => {
        let lo = parseFloat(loInput.value);
        if (lo > parseFloat(hiInput.value)) { lo = parseFloat(hiInput.value); loInput.value = lo; }
        loLabel.textContent = fmtNum(lo);
      });
      hiInput.addEventListener("input", () => {
        let hi = parseFloat(hiInput.value);
        if (hi < parseFloat(loInput.value)) { hi = parseFloat(loInput.value); hiInput.value = hi; }
        hiLabel.textContent = fmtNum(hi);
      });
      const applyRange = () => {
        const lo = parseFloat(loInput.value);
        const hi = parseFloat(hiInput.value);
        onNumericRangeChange([(lo - vmin) / span, (hi - vmin) / span]);
      };
      loInput.addEventListener("change", applyRange);
      hiInput.addEventListener("change", applyRange);
    } else {
      legendEl.innerHTML =
        `<div class="pc-legend-inner pc-legend-continuous">` +
        `<span>${fmtNum(vmin)}</span>` +
        `<div class="pc-legend-grad" style="background:${grad};"></div>` +
        `<span>${fmtNum(vmax)}</span></div>`;
    }
  } else {
    legendEl.innerHTML = "";
  }
}

export default {
  render({ model, el }) {
    // --- mutable UI state (JS-local, no Python sync needed) ---
    let scatter = null;
    let displayMode = "data";   // "data" | "density"
    let colormap = "viridis";
    let sigma = 1.5;
    let selMode = null;         // null (pan) | "lasso" | "box"
    let ticksVisible = true;
    let controlsOpen = false;
    let cachedX = null;
    let cachedY = null;
    let cachedDensity = null;
    let densityMax = 0;        // max raw bin count from last density computation
    let colBank = loadColBank(model);
    let colCache = new Map();  // col name → ArrayBuffer of encoded bytes
    let _recolorRaf = null;    // requestAnimationFrame handle for debounced recolor
    let _drawInFlight = false; // true while scatter.draw() promise is pending
    let _pendingDrawFn = null; // latest queued draw (latest-wins coalescing)
    let uiPointSize = model.get("_point_size") ?? 4;
    let opacity = 1.0;
    let firstDraw = true;
    let axesVisible = true;
    let gridVisible = false;
    let dataExtent = null;   // {xMin,xMax,yMin,yMax} set during density computation
    let lastXScale = null;
    let lastYScale = null;
    let cachedZ = null;
    let activeCategories = null;   // null = all active; Set of active category indices
    let numericFilterRange = null; // null or [lo, hi] in raw data space
    let filteredToOriginal = null; // null or Int32Array mapping displayed→original index
    let displayedX = null;
    let displayedY = null;
    let dataDomainX = null;   // [lo, hi] in data space with padding
    let dataDomainY = null;

    // --- multi-region selection state ---
    let selectionBboxes = [];     // [[xmin,ymin,xmax,ymax], ...] one per Ctrl+drag region
    let selectionPolygons = [];   // [[x,y],...] polygon vertices in data space per region
    let prevSelectedSet = new Set(); // display indices selected before current event
    let ctrlAtLassoStart = false; // whether Ctrl was held at the start of the lasso gesture
    let pendingBboxes = null;     // bbox list buffered in select, committed in lassoEnd
    let pendingIsAdditive = false; // whether ctrlAtLassoStart was true when select fired

    const initW = model.get("width");
    const initH = model.get("height");

    // --- DOM skeleton ---
    const widget = document.createElement("div");
    widget.className = "pc-widget";
    widget.style.cssText = `width:${initW}px;display:inline-block;`;
    el.appendChild(widget);

    // Toolbar
    const toolbar = document.createElement("div");
    toolbar.className = "pc-toolbar";
    widget.appendChild(toolbar);

    // Header row: title + expand button
    const headerRow = document.createElement("div");
    headerRow.className = "pc-toolbar-header";
    toolbar.appendChild(headerRow);

    const titleEl = document.createElement("span");
    titleEl.className = "pc-title";
    titleEl.contentEditable = "true";
    titleEl.spellcheck = false;
    titleEl.dataset.placeholder = "Add title…";
    titleEl.textContent = model.get("title") || "";
    headerRow.appendChild(titleEl);

    const expandBtn = document.createElement("button");
    expandBtn.className = "pc-expand-btn";
    expandBtn.title = "Toggle controls";
    expandBtn.innerHTML = "⚙";
    headerRow.appendChild(expandBtn);

    // Controls panel (collapsible)
    const controlsPanel = document.createElement("div");
    controlsPanel.className = "pc-controls-panel";
    controlsPanel.hidden = true;
    toolbar.appendChild(controlsPanel);

    // Row 1: color, sigma, colormap, selection, clear
    const row1 = document.createElement("div");
    row1.className = "pc-controls-row";
    controlsPanel.appendChild(row1);

    function btnGroup(items, activeVal, containerId) {
      const g = document.createElement("div");
      g.className = "pc-btn-group";
      if (containerId) g.id = containerId;
      items.forEach(({ label, val }) => {
        const b = document.createElement("button");
        b.className = "pc-btn" + (val === activeVal ? " pc-active" : "");
        b.dataset.val = val;
        b.textContent = label;
        g.appendChild(b);
      });
      return g;
    }

    // Color column picker (only visible when data was loaded via add_scatter_frame)
    const colPickerWrap = document.createElement("span");
    colPickerWrap.className = "pc-col-picker-wrap";
    colPickerWrap.hidden = true;
    colPickerWrap.appendChild(Object.assign(document.createElement("span"), { className: "pc-label", textContent: "Col:" }));
    const colPickerSelect = document.createElement("select");
    colPickerSelect.className = "pc-colormap-select";
    colPickerWrap.appendChild(colPickerSelect);
    row1.appendChild(colPickerWrap);

    function refreshColPicker() {
      const cols = model.get("_frame_columns");
      if (!cols || cols.length === 0) { colPickerWrap.hidden = true; return; }
      colPickerWrap.hidden = false;
      colPickerSelect.innerHTML = "";
      const noneOpt = new Option("— none —", "");
      colPickerSelect.appendChild(noneOpt);
      cols.forEach(c => colPickerSelect.appendChild(new Option(c, c)));
      colPickerSelect.value = model.get("_selected_color_col");
    }
    refreshColPicker();

    const colorModeGroup = btnGroup([{ label: "Data", val: "data" }, { label: "Density", val: "density" }], "data");
    row1.appendChild(Object.assign(document.createElement("span"), { className: "pc-label", textContent: "Color:" }));
    row1.appendChild(colorModeGroup);

    const sigmaGroup = document.createElement("span");
    sigmaGroup.className = "pc-sigma-group";
    sigmaGroup.hidden = true;
    sigmaGroup.innerHTML =
      `<span class="pc-label">σ:</span>` +
      `<input type="range" class="pc-sigma-slider" min="0.5" max="5" step="0.1" value="1.5">` +
      `<span class="pc-sigma-val">1.5</span>`;
    row1.appendChild(sigmaGroup);

    const cmSelect = document.createElement("select");
    cmSelect.className = "pc-colormap-select";
    COLORMAP_NAMES.forEach(name => {
      const opt = new Option(name.charAt(0).toUpperCase() + name.slice(1), name);
      cmSelect.appendChild(opt);
    });
    row1.appendChild(cmSelect);

    const selModeGroup = btnGroup([{ label: "⧝ Lasso", val: "lasso" }, { label: "□ Box", val: "box" }], null);
    row1.appendChild(selModeGroup);

    const clearBtn = document.createElement("button");
    clearBtn.className = "pc-btn";
    clearBtn.textContent = "✕ Clear";
    row1.appendChild(clearBtn);

    // Row 2: axis labels + ticks
    const row2 = document.createElement("div");
    row2.className = "pc-controls-row";
    controlsPanel.appendChild(row2);

    const xInput = document.createElement("input");
    xInput.type = "text";
    xInput.className = "pc-axis-input";
    xInput.placeholder = "X axis label";
    xInput.value = model.get("x_label");

    const yInput = document.createElement("input");
    yInput.type = "text";
    yInput.className = "pc-axis-input";
    yInput.placeholder = "Y axis label";
    yInput.value = model.get("y_label");

    const axesLabel = document.createElement("label");
    axesLabel.className = "pc-check-label";
    const axesCheck = document.createElement("input");
    axesCheck.type = "checkbox";
    axesCheck.checked = true;
    axesLabel.appendChild(axesCheck);
    axesLabel.append(" Axes");

    const gridLabel = document.createElement("label");
    gridLabel.className = "pc-check-label";
    const gridCheck = document.createElement("input");
    gridCheck.type = "checkbox";
    gridCheck.checked = false;
    gridLabel.appendChild(gridCheck);
    gridLabel.append(" Grid");

    row2.appendChild(Object.assign(document.createElement("span"), { className: "pc-label", textContent: "X:" }));
    row2.appendChild(xInput);
    row2.appendChild(Object.assign(document.createElement("span"), { className: "pc-label", textContent: "Y:" }));
    row2.appendChild(yInput);
    row2.appendChild(axesLabel);
    row2.appendChild(gridLabel);

    const sizeGroup = document.createElement("span");
    sizeGroup.className = "pc-sigma-group";
    sizeGroup.innerHTML =
      `<span class="pc-label">Point Size:</span>` +
      `<input type="range" class="pc-sigma-slider" min="1" max="100" step="1" value="${uiPointSize}">` +
      `<span class="pc-sigma-val">${uiPointSize}</span>`;
    row2.appendChild(sizeGroup);

    const alphaGroup = document.createElement("span");
    alphaGroup.className = "pc-sigma-group";
    alphaGroup.innerHTML =
      `<span class="pc-label">Alpha:</span>` +
      `<input type="range" class="pc-sigma-slider" min="0.02" max="1" step="0.02" value="${opacity}">` +
      `<span class="pc-sigma-val">1</span>`;
    row2.appendChild(alphaGroup);

    // Plot area
    const plotDiv = document.createElement("div");
    plotDiv.className = "pc-plot";
    plotDiv.style.cssText = `position:relative;width:${initW}px;height:${initH}px;`;
    widget.appendChild(plotDiv);

    const canvas = document.createElement("canvas");
    plotDiv.appendChild(canvas);
    canvas.addEventListener("pointerdown", e => { ctrlAtLassoStart = e.ctrlKey; });

    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgEl.style.cssText = "position:absolute;top:0;left:0;overflow:visible;pointer-events:none;z-index:1;";
    plotDiv.appendChild(svgEl);

    const svg = select(svgEl);
    const gGrid = svg.append("g").attr("class", "pc-grid");   // behind axes
    const gAxes = svg.append("g").attr("class", "axes");
    const gXAxis = gAxes.append("g").attr("class", "x-axis");
    const gYAxis = gAxes.append("g").attr("class", "y-axis");
    const xAxisLabel = svg.append("text").attr("class", "axis-label x-label").attr("text-anchor", "middle");
    const yAxisLabel = svg.append("text").attr("class", "axis-label y-label").attr("text-anchor", "middle");

    // Legend
    const legendDiv = document.createElement("div");
    legendDiv.className = "pc-legend";
    widget.appendChild(legendDiv);

    // --- axis helpers ---
    function plotDims() {
      const w = model.get("width");
      const h = model.get("height");
      return { w, h, pw: w - MARGIN.left - MARGIN.right, ph: h - MARGIN.top - MARGIN.bottom };
    }

    function setAxisPositions() {
      const { w, h } = plotDims();
      gGrid.attr("transform", null);
      gAxes.attr("transform", null);
      gXAxis.attr("transform", `translate(0,${h})`);
      // gYAxis stays at (0,0) — axisLeft draws ticks into the overflow area to the left
      xAxisLabel.attr("x", w / 2).attr("y", h + 35);
      yAxisLabel.attr("transform", `translate(-45,${h / 2}) rotate(-90)`);
    }

    function setAxisLabelText() {
      xAxisLabel.text(model.get("x_label"));
      yAxisLabel.text(model.get("y_label"));
    }

    function styleAxes() {
      const vis = axesVisible ? null : "hidden";
      gAxes.attr("visibility", vis);
      xAxisLabel.attr("visibility", vis);
      yAxisLabel.attr("visibility", vis);
      svg.selectAll(".domain, .tick line").style("stroke", "var(--pc-axis-color, #999)");
      svg.selectAll(".tick text").style("fill", "var(--pc-label-color, #555)").attr("font-size", "11px");
      svg.selectAll(".axis-label").style("fill", "var(--pc-label-color, #555)").attr("font-size", "12px");
    }

    function renderGrid(xScale, yScale) {
      gGrid.selectAll("*").remove();
      const { w, h } = plotDims();
      xScale.ticks(6).forEach(t => {
        const sx = xScale(t);
        gGrid.append("line").attr("x1", sx).attr("y1", 0).attr("x2", sx).attr("y2", h);
      });
      yScale.ticks(6).forEach(t => {
        const sy = yScale(t);
        gGrid.append("line").attr("x1", 0).attr("y1", sy).attr("x2", w).attr("y2", sy);
      });
      gGrid.selectAll("line")
        .style("stroke", "var(--pc-axis-color, #999)")
        .style("stroke-dasharray", "4 4")
        .style("stroke-opacity", "0.45")
        .attr("stroke-width", 0.8);
    }

    function syncAxes(xScale, yScale) {
      lastXScale = xScale;
      lastYScale = yScale;
      gXAxis.call(axisBottom(xScale).ticks(6));
      gYAxis.call(axisLeft(yScale).ticks(6));
      styleAxes();
      if (gridVisible) renderGrid(xScale, yScale);
    }

    setAxisPositions();
    setAxisLabelText();

    // --- draw helpers ---
    function buildFilteredPts(x, y, z, zDataType, wArr) {
      const hasFilter = (activeCategories !== null && zDataType === "categorical") ||
                        (numericFilterRange !== null && zDataType === "continuous");
      if (!z || !zDataType || displayMode === "density" || !hasFilter) {
        filteredToOriginal = null;
        displayedX = x;
        displayedY = y;
        return { x, y, z, wArr };
      }
      const n = x.length;
      const keep = new Uint8Array(n);
      let keepCount = 0;
      for (let i = 0; i < n; i++) {
        let pass = false;
        if (zDataType === "categorical") {
          pass = activeCategories === null || activeCategories.has(z[i]);
        } else {
          const [lo, hi] = numericFilterRange;
          pass = z[i] >= lo && z[i] <= hi;
        }
        keep[i] = pass ? 1 : 0;
        if (pass) keepCount++;
      }
      if (keepCount === n) {
        filteredToOriginal = null;
        displayedX = x;
        displayedY = y;
        return { x, y, z, wArr };
      }
      filteredToOriginal = new Int32Array(keepCount);
      const fx = new Float32Array(keepCount);
      const fy = new Float32Array(keepCount);
      const fz = new Float32Array(keepCount);
      const fw = wArr ? new Float32Array(keepCount) : null;
      let j = 0;
      for (let i = 0; i < n; i++) {
        if (keep[i]) {
          filteredToOriginal[j] = i;
          fx[j] = x[i];
          fy[j] = y[i];
          fz[j] = z[i];
          if (fw) fw[j] = wArr[i];
          j++;
        }
      }
      displayedX = fx;
      displayedY = fy;
      return { x: fx, y: fy, z: fz, wArr: fw };
    }

    function applyColor(x, y) {
      // regl-scatterplot expects NDC [-1, 1] coordinates; normalize using the current data domain
      const nx = dataDomainX ? normalizeArr(x, dataDomainX[0], dataDomainX[1]) : x;
      const ny = dataDomainY ? normalizeArr(y, dataDomainY[0], dataDomainY[1]) : y;

      if (displayMode === "density") {
        const result = computeDensity(nx, ny, sigma);
        cachedDensity = result.density;
        densityMax = result.dMax;
        dataExtent = { xMin: result.xMin, xMax: result.xMax, yMin: result.yMin, yMax: result.yMax };
        filteredToOriginal = null;
        displayedX = x;
        displayedY = y;
      } else {
        cachedDensity = null;
        densityMax = 0;
        dataExtent = null;
      }
      const { zArr, colorBy, pointColor, zDataType } = getColorProps(model, displayMode, colormap, cachedDensity, colBank, colCache);
      cachedZ = zArr;
      const { wArr, sizeBy } = getSizeProps(model);

      const filtered = buildFilteredPts(nx, ny, zArr, zDataType, wArr);

      const setProps = { colorBy, pointColor };
      if (!wArr) setProps.pointSize = uiPointSize;
      if (sizeBy) setProps.sizeBy = sizeBy;
      scatter.set(setProps);

      const pts = { x: filtered.x, y: filtered.y };
      if (filtered.z) pts.z = filtered.z;
      if (filtered.wArr) pts.w = filtered.wArr;
      const drawOpts = zDataType ? { zDataType } : {};

      // Guard: if a draw is already in flight, queue this one (latest wins).
      // When the in-flight draw finishes it will run the pending draw automatically.
      const executeDraw = () => {
        _drawInFlight = true;
        return scatter.draw(pts, drawOpts).then(
          result => {
            _drawInFlight = false;
            if (_pendingDrawFn) { const fn = _pendingDrawFn; _pendingDrawFn = null; fn(); }
            return result;
          },
          err => { _drawInFlight = false; _pendingDrawFn = null; throw err; }
        );
      };
      if (_drawInFlight) { _pendingDrawFn = executeDraw; return Promise.resolve(); }
      return executeDraw();
    }

    function redraw() {
      if (!scatter) return;
      if (model.get("_n_points") === 0) return;
      const xDv = model.get("_x_data");
      if (!xDv || xDv.byteLength === 0) return;

      cachedX = toFloat32(xDv);
      cachedY = toFloat32(model.get("_y_data"));

      // Compute data bounds with 5% padding so the scatter scales cover all data
      let xMin = cachedX[0], xMax = cachedX[0], yMin = cachedY[0], yMax = cachedY[0];
      for (let i = 1; i < cachedX.length; i++) {
        if (cachedX[i] < xMin) xMin = cachedX[i];
        if (cachedX[i] > xMax) xMax = cachedX[i];
        if (cachedY[i] < yMin) yMin = cachedY[i];
        if (cachedY[i] > yMax) yMax = cachedY[i];
      }
      const xPad = (xMax - xMin) * 0.05 || 0.5;
      const yPad = (yMax - yMin) * 0.05 || 0.5;
      dataDomainX = [xMin - xPad, xMax + xPad];
      dataDomainY = [yMin - yPad, yMax + yPad];

      // applyColor normalizes x/y to NDC using dataDomainX/Y before passing to regl
      applyColor(cachedX, cachedY).then(() => {
        if (firstDraw) { scatter.reset(); firstDraw = false; }
        // Build data-space D3 scales for the axes (independent of regl's internal scale)
        const { w, h } = plotDims();
        const dxScale = scaleLinear().domain(dataDomainX).range([0, w]);
        const dyScale = scaleLinear().domain([dataDomainY[1], dataDomainY[0]]).range([0, h]);
        syncAxes(dxScale, dyScale);
        renderLegendNow();
      });
    }

    function recolor() {
      if (!scatter || !cachedX) { redraw(); return; }
      applyColor(cachedX, cachedY);
      renderLegendNow();
      if (gridVisible && lastXScale && lastYScale) renderGrid(lastXScale, lastYScale);
    }

    // Debounced recolor: coalesces rapid calls (e.g. fast legend clicks) into one
    // draw per animation frame so scatter.draw() is never called while still in flight.
    function scheduleRecolor() {
      if (_recolorRaf !== null) cancelAnimationFrame(_recolorRaf);
      _recolorRaf = requestAnimationFrame(() => { _recolorRaf = null; recolor(); });
    }

    function renderLegendNow() {
      const callbacks = {
        activeCategories,
        onCategoryToggle(idx, nTotal, detail) {
          if (detail >= 2) {
            activeCategories = (activeCategories !== null && activeCategories.size === 1 && activeCategories.has(idx))
              ? null
              : new Set([idx]);
          } else {
            if (activeCategories === null) {
              activeCategories = new Set(Array.from({ length: nTotal }, (_, i) => i));
              activeCategories.delete(idx);
            } else if (activeCategories.has(idx)) {
              activeCategories.delete(idx);
              if (activeCategories.size === nTotal) activeCategories = null;
            } else {
              activeCategories.add(idx);
              if (activeCategories.size === nTotal) activeCategories = null;
            }
          }
          scheduleRecolor();
        },
        numericFilterRange,
        onNumericRangeChange(range) {
          numericFilterRange = range;
          scheduleRecolor();
        },
      };
      renderLegend(legendDiv, model, displayMode, colormap, colBank, densityMax, callbacks);
    }

    function resize(w, h) {
      widget.style.width = `${w}px`;
      plotDiv.style.width = `${w}px`;
      plotDiv.style.height = `${h}px`;
      svg.attr("width", w).attr("height", h);
      setAxisPositions();
      if (scatter) scatter.set({ width: w, height: h });
    }

    // --- scatter init ---
    function initScatter() {
      if (scatter) scatter.destroy();
      const th = model.get("theme");
      const darkBg = th === "dark" ||
        (th === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);
      const w0 = model.get("width");
      const h0 = model.get("height");
      scatter = createScatterplot({
        canvas,
        width: w0,
        height: h0,
        pointSize: uiPointSize,
        opacity,
        backgroundColor: darkBg ? [30 / 255, 30 / 255, 30 / 255, 1] : [1, 1, 1, 1],
        pointSizeSelected: 0,
        lassoInitiator: true,
        mouseMode: selMode === null ? "panZoom" : "lasso",
        // Ctrl+drag merges with existing selection; other modifier keys keep defaults.
        actionKeyMap: { remove: "alt", rotate: "alt", lasso: "shift", merge: "ctrl" },
        // NDC scales: regl needs xScale/yScale set so the view event reports non-null
        // scales on every pan/zoom. Domain stays [-1,1] (matches our normalized data);
        // regl overrides range itself to [0,width] / [height,0].
        xScale: scaleLinear().domain([-1, 1]).range([0, w0]),
        yScale: scaleLinear().domain([-1, 1]).range([h0, 0]),
      });
      scatter.subscribe("select", ({ points }) => {
        // Buffer selection state; lassoEnd will save atomically with the polygon.
        const wasAdditive = ctrlAtLassoStart && prevSelectedSet.size > 0;
        const newDisplayIndices = Array.from(points).filter(p => !prevSelectedSet.has(p));
        const newOriginalIndices = filteredToOriginal
          ? newDisplayIndices.map(i => filteredToOriginal[i])
          : newDisplayIndices;

        if (newOriginalIndices.length > 0 && cachedX) {
          let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
          for (const i of newOriginalIndices) {
            if (cachedX[i] < xMin) xMin = cachedX[i];
            if (cachedX[i] > xMax) xMax = cachedX[i];
            if (cachedY[i] < yMin) yMin = cachedY[i];
            if (cachedY[i] > yMax) yMax = cachedY[i];
          }
          pendingBboxes = wasAdditive ? [...selectionBboxes, [xMin, yMin, xMax, yMax]] : [[xMin, yMin, xMax, yMax]];
        } else {
          pendingBboxes = wasAdditive ? [...selectionBboxes] : [];
        }
        pendingIsAdditive = wasAdditive;
        prevSelectedSet = new Set(points);
      });
      scatter.subscribe("deselect", () => {
        selectionBboxes = [];
        selectionPolygons = [];
        prevSelectedSet = new Set();
        pendingBboxes = null;
        model.set("selected_indices", []);
        model.set("_selection_bbox", []);
        model.set("_selection_polygons", []);
        model.save_changes();
      });
      scatter.subscribe("lassoEnd", ({ coordinates }) => {
        // select fires before lassoEnd; if no select preceded this, nothing to commit.
        if (pendingBboxes === null) return;
        if (coordinates && coordinates.length > 0 && dataDomainX && dataDomainY) {
          const xSpan = dataDomainX[1] - dataDomainX[0];
          const ySpan = dataDomainY[1] - dataDomainY[0];
          // Convert NDC polygon vertices to data space for server-side PIP filtering.
          const polygon = coordinates.map(([nx, ny]) => [
            dataDomainX[0] + (nx + 1) / 2 * xSpan,
            dataDomainY[0] + (ny + 1) / 2 * ySpan,
          ]);
          selectionPolygons = pendingIsAdditive
            ? [...selectionPolygons, polygon]
            : [polygon];
        } else if (!pendingIsAdditive) {
          selectionPolygons = [];
        }
        selectionBboxes = pendingBboxes;
        pendingBboxes = null;
        const originalIndices = filteredToOriginal
          ? Array.from(prevSelectedSet).map(i => filteredToOriginal[i])
          : Array.from(prevSelectedSet);
        model.set("selected_indices", originalIndices);
        model.set("_selection_bbox", selectionBboxes);
        model.set("_selection_polygons", selectionPolygons);
        model.save_changes();
      });
      scatter.subscribe("view", ({ xScale, yScale }) => {
        if (!xScale || !yScale || !dataDomainX || !dataDomainY) return;
        // regl provides scales with NDC domain at default zoom.
        // xScale: domain = [left_ndc, right_ndc], range = [0, width]
        // yScale: domain = [bottom_ndc, top_ndc],  range = [height, 0]
        //   (domain[0] = bottom of canvas, domain[1] = top of canvas)
        const xSpan = dataDomainX[1] - dataDomainX[0];
        const ySpan = dataDomainY[1] - dataDomainY[0];
        const [nxLo, nxHi] = xScale.domain();
        const [nyBot, nyTop] = yScale.domain(); // bot = domain[0], top = domain[1]
        const { w, h } = plotDims();
        const dxScale = scaleLinear()
          .domain([
            dataDomainX[0] + (nxLo + 1) / 2 * xSpan,
            dataDomainX[0] + (nxHi + 1) / 2 * xSpan,
          ])
          .range([0, w]);
        // D3 dyScale: domain[0] at pixel 0 (top), so put the highest visible data y first
        const dyScale = scaleLinear()
          .domain([
            dataDomainY[0] + (nyTop + 1) / 2 * ySpan,  // top of canvas → highest data y
            dataDomainY[0] + (nyBot + 1) / 2 * ySpan,  // bottom of canvas → lowest data y
          ])
          .range([0, h]);
        syncAxes(dxScale, dyScale);
      });
    }

    // --- toolbar event handlers ---

    expandBtn.addEventListener("click", () => {
      controlsOpen = !controlsOpen;
      controlsPanel.hidden = !controlsOpen;
      expandBtn.innerHTML = controlsOpen ? "⚙▲" : "⚙";
    });

    titleEl.addEventListener("blur", () => {
      const v = titleEl.textContent.trim();
      titleEl.textContent = v;
      model.set("title", v);
      model.save_changes();
    });
    titleEl.addEventListener("keydown", e => {
      if (e.key === "Enter") { e.preventDefault(); titleEl.blur(); }
      if (e.key === "Escape") { titleEl.textContent = model.get("title"); titleEl.blur(); }
    });

    colorModeGroup.addEventListener("click", e => {
      const btn = e.target.closest("[data-val]");
      if (!btn) return;
      displayMode = btn.dataset.val;
      colorModeGroup.querySelectorAll(".pc-btn").forEach(b => b.classList.toggle("pc-active", b.dataset.val === displayMode));
      sigmaGroup.hidden = displayMode !== "density";
      activeCategories = null;
      numericFilterRange = null;
      scheduleRecolor();
    });

    const sigmaSlider = sigmaGroup.querySelector(".pc-sigma-slider");
    const sigmaValEl = sigmaGroup.querySelector(".pc-sigma-val");
    sigmaSlider.addEventListener("input", () => {
      sigma = parseFloat(sigmaSlider.value);
      sigmaValEl.textContent = sigma.toFixed(1);
      if (displayMode === "density") scheduleRecolor();
    });

    cmSelect.addEventListener("change", () => {
      colormap = cmSelect.value;
      if (displayMode === "density" || model.get("_color_mode") === "continuous") scheduleRecolor();
      else renderLegendNow();
    });

    selModeGroup.addEventListener("click", e => {
      const btn = e.target.closest("[data-val]");
      if (!btn) return;
      selMode = btn.dataset.val === selMode ? null : btn.dataset.val;
      selModeGroup.querySelectorAll(".pc-btn").forEach(b => b.classList.toggle("pc-active", b.dataset.val === selMode));
      if (scatter) {
        if (selMode === null) {
          scatter.set({ mouseMode: "panZoom" });
        } else {
          scatter.set({ mouseMode: "lasso", lassoType: selMode === "box" ? "rectangle" : "freeform" });
        }
      }
    });

    clearBtn.addEventListener("click", () => { if (scatter) scatter.deselect(); });

    colPickerSelect.addEventListener("change", () => {
      model.set("_selected_color_col", colPickerSelect.value);
      model.save_changes();
      activeCategories = null;
      numericFilterRange = null;
      scheduleRecolor();
    });

    xInput.addEventListener("input", () => {
      model.set("x_label", xInput.value);
      model.save_changes();
      xAxisLabel.text(xInput.value);
    });

    yInput.addEventListener("input", () => {
      model.set("y_label", yInput.value);
      model.save_changes();
      yAxisLabel.text(yInput.value);
    });

    axesCheck.addEventListener("change", () => {
      axesVisible = axesCheck.checked;
      styleAxes();
    });

    gridCheck.addEventListener("change", () => {
      gridVisible = gridCheck.checked;
      if (gridVisible) {
        const xs = lastXScale || (scatter && scatter.get("xScale"));
        const ys = lastYScale || (scatter && scatter.get("yScale"));
        if (xs && ys) renderGrid(xs, ys);
      } else {
        gGrid.selectAll("*").remove();
      }
    });

    function applyTheme(theme) {
      const dark = theme === "dark" ||
        (theme === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);
      if (dark) {
        widget.setAttribute("data-theme", "dark");
        if (scatter) scatter.set({ backgroundColor: [30 / 255, 30 / 255, 30 / 255, 1] });
      } else {
        widget.removeAttribute("data-theme");
        if (scatter) scatter.set({ backgroundColor: [1, 1, 1, 1] });
      }
    }

    const sizeSlider = sizeGroup.querySelector("input[type=range]");
    const sizeValEl = sizeGroup.querySelector(".pc-sigma-val");
    sizeSlider.addEventListener("input", () => {
      sizeValEl.textContent = sizeSlider.value;
    });
    sizeSlider.addEventListener("change", () => {
      uiPointSize = parseFloat(sizeSlider.value);
      if (scatter) scatter.set({ pointSize: uiPointSize });
    });

    const alphaSlider = alphaGroup.querySelector("input[type=range]");
    const alphaValEl = alphaGroup.querySelector(".pc-sigma-val");
    alphaSlider.addEventListener("input", () => {
      alphaValEl.textContent = parseFloat(alphaSlider.value).toFixed(2);
    });
    alphaSlider.addEventListener("change", () => {
      opacity = parseFloat(alphaSlider.value);
      if (scatter) scatter.set({ opacity });
    });

    // --- model bindings ---
    model.on("change:_x_data change:_y_data change:_color_data change:_size_data change:_color_mode change:_color_palette change:_point_size change:_redraw_trigger", redraw);
    // New dataset: clear all selection state so per-region bboxes/polygons don't carry over.
    model.on("change:_layers", () => {
      selectionBboxes = [];
      selectionPolygons = [];
      prevSelectedSet = new Set();
      pendingBboxes = null;
      redraw();
    });

    model.on("change:width change:height", () => {
      resize(model.get("width"), model.get("height"));
      redraw();
    });

    model.on("change:title", () => { titleEl.textContent = model.get("title"); });

    model.on("change:_color_bank_meta", () => {
      // Refresh metadata only — do NOT clear colCache here.
      // Cached column bytes remain valid; only _frame_columns change (new dataset)
      // triggers a cache clear.
      colBank = loadColBank(model);
    });
    model.on("change:_col_chunk_ts", () => {
      const col = model.get("_col_chunk_col");
      const dv = model.get("_col_chunk_data");
      if (!col || !dv || dv.byteLength === 0) return;
      // Copy the buffer — the DataView is backed by the comm's buffer which may be reused
      colCache.set(col, dv.buffer.slice(dv.byteOffset, dv.byteOffset + dv.byteLength));
      if (col === model.get("_selected_color_col")) {
        activeCategories = null;
        numericFilterRange = null;
        scheduleRecolor();
      }
    });
    model.on("change:_frame_columns", () => {
      // New dataset: clear the column cache and refresh the picker.
      colCache.clear();
      colBank = loadColBank(model);
      refreshColPicker();
    });
    model.on("change:_selected_color_col", () => { colPickerSelect.value = model.get("_selected_color_col"); });

    model.on("change:x_label", () => {
      const v = model.get("x_label");
      xInput.value = v;
      xAxisLabel.text(v);
    });

    model.on("change:y_label", () => {
      const v = model.get("y_label");
      yInput.value = v;
      yAxisLabel.text(v);
    });

    model.on("change:_reset_view_ts", () => { if (scatter) scatter.reset(); });
    model.on("change:theme", () => applyTheme(model.get("theme")));

    // --- init ---
    initScatter();
    applyTheme(model.get("theme"));
    requestAnimationFrame(() => {
      resize(model.get("width"), model.get("height"));
      redraw();
    });

    return () => { if (scatter) { scatter.destroy(); scatter = null; } };
  },
};
