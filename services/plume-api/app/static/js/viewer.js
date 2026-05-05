// viewer.js — Plume side-by-side showcase: ERA5 25 km vs Plume 30 m
//
// Two synchronized MapLibre maps:
//   Left:  ERA5 baseline — satellite + flat terrain + coarse wind (uniform arrows)
//   Right: Plume 30 m — satellite + detailed terrain + GPU wind particles (wind_gl.js)
//
// Camera is synced between both maps. lil-gui controls altitude, particle density.

import { loadCase, loadCaseFromBuffer, speedAtHeight, speedToRGBA } from "/static/js/loader.js";
import { WindGLLayer, encodeWindTexture } from "/static/js/wind_gl.js";

const LILGUI_SRC = "https://cdn.jsdelivr.net/npm/lil-gui@0.19/dist/lil-gui.esm.js";
const { default: GUI } = await import(LILGUI_SRC);

const DEMO_BASE = "/demo";
let currentCase = null;

// ── Shared map style (base) ─────────────────────────────────────────────────

function makeStyle(opts = {}) {
  const style = {
    version: 8,
    glyphs: "https://fonts.openmaptiles.org/{fontstack}/{range}.pbf",
    sources: {
      satellite: {
        type: "raster",
        tiles: [
          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256, attribution: "Imagery © Esri", maxzoom: 19,
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#0a0e1a" } },
      { id: "sat", type: "raster", source: "satellite" },
    ],
  };
  if (opts.terrain) {
    style.sources.terrain = {
      type: "raster-dem",
      tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
      tileSize: 256, encoding: "terrarium", maxzoom: 15,
    };
    style.terrain = { source: "terrain", exaggeration: opts.exaggeration || 1.6 };
  }
  if (opts.sky) {
    style.sky = {
      "sky-color": "#0b1830", "sky-horizon-blend": 0.5,
      "horizon-color": "#1d3a6b", "horizon-fog-blend": 0.5,
    };
  }
  return style;
}

// ── Create maps ──────────────────────────────────────────────────────────────

const defaultCenter = [-7.7386, 39.7125];
const defaultZoom = 13.5;
const defaultPitch = 60;
const defaultBearing = 30;

const mapEra5 = new maplibregl.Map({
  container: "map-era5",
  style: makeStyle({ terrain: true, exaggeration: 0.3, sky: true }),  // flat-ish
  center: defaultCenter, zoom: defaultZoom, pitch: defaultPitch, bearing: defaultBearing,
  antialias: true,
});

const mapPlume = new maplibregl.Map({
  container: "map-plume",
  style: makeStyle({ terrain: true, exaggeration: 1.8, sky: true }),
  center: defaultCenter, zoom: defaultZoom, pitch: defaultPitch, bearing: defaultBearing,
  antialias: true,
});

mapPlume.addControl(new maplibregl.NavigationControl({ visualizePitch: true }));

// ── Sync camera ──────────────────────────────────────────────────────────────

let syncing = false;

function syncCamera(source, target) {
  if (syncing) return;
  syncing = true;
  target.jumpTo({
    center: source.getCenter(),
    zoom: source.getZoom(),
    pitch: source.getPitch(),
    bearing: source.getBearing(),
  });
  syncing = false;
}

["move", "zoom", "pitch", "rotate"].forEach((ev) => {
  mapPlume.on(ev, () => syncCamera(mapPlume, mapEra5));
  mapEra5.on(ev, () => syncCamera(mapEra5, mapPlume));
});

// ── Wind particle layer (right map only) ──────────────────────────────────

const windLayer = new WindGLLayer("plume-wind", {
  numParticles: 8000,
  fadeOpacity: 0.985,
  speedFactor: 1.0,
});

// ── State ────────────────────────────────────────────────────────────────────

const state = {
  targetZ: 60.0,
  showSlice: true,
  opacity: 0.6,
  vmax: 20.0,
  numParticles: 8000,
  speedFactor: 1.0,
  fadeOpacity: 0.985,
};

// ── Loading ──────────────────────────────────────────────────────────────────

function setLoading(msg) {
  const el = document.getElementById("loading");
  if (!msg) { el.classList.add("hidden"); return; }
  el.textContent = msg;
  el.classList.remove("hidden");
}

async function loadDemoCase(caseName) {
  setLoading("Loading CFD field…");
  try {
    currentCase = await loadCase(`${DEMO_BASE}/${caseName}.bin`);
    onCaseReady();
  } catch (e) {
    console.error(e);
    alert(`Failed to load case: ${e.message}`);
  } finally {
    setLoading(null);
  }
}

function onCaseReady() {
  const { latCenter, lonCenter, bounds } = currentCase;
  [mapEra5, mapPlume].forEach((m) =>
    m.flyTo({ center: [lonCenter, latCenter], zoom: 13.5, pitch: 60, bearing: 30, duration: 1500 })
  );
  updateWindOverlay();
  updateWindParticles();
}

// ── ERA5 side: coarse wind overlay ──────────────────────────────────────────

function updateWindOverlay() {
  if (!currentCase) return;
  const { nx, ny, bounds } = currentCase;

  // ERA5 side: very coarse (8×8 subsampled = looks blocky = shows 25km grid)
  const speedEra5 = speedAtHeight(currentCase, state.targetZ);
  // Subsample to ~8×8 to simulate ERA5 resolution
  const era5_n = 8;
  const coarse = new Float32Array(era5_n * era5_n);
  for (let j = 0; j < era5_n; j++) {
    for (let i = 0; i < era5_n; i++) {
      const srcJ = Math.floor(j * ny / era5_n);
      const srcI = Math.floor(i * nx / era5_n);
      coarse[j * era5_n + i] = speedEra5[srcJ * nx + srcI];
    }
  }
  const rgbaEra5 = speedToRGBA(coarse, era5_n, era5_n, 0, state.vmax);

  // Plume side: full resolution
  const speedPlume = speedAtHeight(currentCase, state.targetZ);
  const rgbaPlume = speedToRGBA(speedPlume, nx, ny, 0, state.vmax);

  const coords = [
    [bounds.west, bounds.north], [bounds.east, bounds.north],
    [bounds.east, bounds.south], [bounds.west, bounds.south],
  ];

  updateImageLayer(mapEra5, "wind-slice", rgbaEra5, era5_n, era5_n, coords, state.opacity, state.showSlice);
  updateImageLayer(mapPlume, "wind-slice", rgbaPlume, nx, ny, coords, state.opacity, state.showSlice);
}

function updateImageLayer(map, sourceId, rgba, w, h, coords, opacity, visible) {
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = new ImageData(new Uint8ClampedArray(rgba), w, h);
  ctx.putImageData(img, 0, 0);
  // Flip Y
  const c2 = document.createElement("canvas");
  c2.width = w; c2.height = h;
  const ctx2 = c2.getContext("2d");
  ctx2.translate(0, h); ctx2.scale(1, -1); ctx2.drawImage(canvas, 0, 0);
  const url = c2.toDataURL("image/png");

  if (map.getSource(sourceId)) {
    map.getSource(sourceId).updateImage({ url, coordinates: coords });
  } else {
    map.addSource(sourceId, { type: "image", url, coordinates: coords });
    map.addLayer({
      id: `${sourceId}-layer`, type: "raster", source: sourceId,
      paint: { "raster-opacity": opacity, "raster-fade-duration": 0 },
      layout: { visibility: visible ? "visible" : "none" },
    });
  }
  if (map.getLayer(`${sourceId}-layer`)) {
    map.setPaintProperty(`${sourceId}-layer`, "raster-opacity", opacity);
    map.setLayoutProperty(`${sourceId}-layer`, "visibility", visible ? "visible" : "none");
  }
}

// ── Plume side: GPU wind particles ──────────────────────────────────────────

function updateWindParticles() {
  if (!currentCase) return;
  const { nx, ny, nz, u, v, bounds, zLevels } = currentCase;

  // Extract u/v at target altitude
  const uSlice = new Float32Array(ny * nx);
  const vSlice = new Float32Array(ny * nx);
  let iHi = 0;
  while (iHi < nz && zLevels[iHi] < state.targetZ) iHi++;
  if (iHi === 0) iHi = 1;
  if (iHi >= nz) iHi = nz - 1;
  const iLo = iHi - 1;
  const wHi = (state.targetZ - zLevels[iLo]) / (zLevels[iHi] - zLevels[iLo]);
  const wLo = 1 - wHi;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const base = (j * nx + i) * nz;
      uSlice[j * nx + i] = wLo * u[base + iLo] + wHi * u[base + iHi];
      vSlice[j * nx + i] = wLo * v[base + iLo] + wHi * v[base + iHi];
    }
  }

  const windData = encodeWindTexture(uSlice, vSlice, nx, ny);
  windLayer.setWind(windData, bounds);
  windLayer.numParticles = state.numParticles;
  windLayer.speedFactor = state.speedFactor;
  windLayer.fadeOpacity = state.fadeOpacity;
}

// ── GUI ──────────────────────────────────────────────────────────────────────

const gui = new GUI({ container: document.getElementById("controls"), title: "Plume" });

const fSlice = gui.addFolder("Wind field");
fSlice.add(state, "showSlice").name("Color overlay").onChange(updateWindOverlay);
fSlice.add(state, "targetZ", 5, 500, 5).name("Altitude AGL (m)")
  .onChange(() => { updateWindOverlay(); updateWindParticles(); });
fSlice.add(state, "opacity", 0, 1, 0.05).name("Overlay opacity").onChange(updateWindOverlay);
fSlice.add(state, "vmax", 5, 40, 1).name("Max speed (m/s)").onChange(updateWindOverlay);

const fPart = gui.addFolder("Particles (right panel)");
fPart.add(state, "numParticles", 1000, 20000, 1000).name("Count").onChange(updateWindParticles);
fPart.add(state, "speedFactor", 0.2, 4.0, 0.1).name("Speed").onChange(updateWindParticles);
fPart.add(state, "fadeOpacity", 0.9, 0.999, 0.001).name("Trail length").onChange(updateWindParticles);

// ── Site selector ────────────────────────────────────────────────────────────

document.querySelectorAll(".site-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.disabled) return;
    document.querySelectorAll(".site-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    loadDemoCase(btn.dataset.case);
  });
});

// ── Boot ─────────────────────────────────────────────────────────────────────

Promise.all([
  new Promise((r) => mapEra5.on("load", r)),
  new Promise((r) => mapPlume.on("load", r)),
]).then(async () => {
  mapPlume.addLayer(windLayer);
  await loadDemoCase("synthetic_ridge");
});
