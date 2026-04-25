// LUCiD Event Viewer — main module.
//
// Architecture modeled on the JAXTPC viewer: Three.js WebGL 3D panel +
// Canvas2D unwrapped 2D panel, with h5wasm streaming in a worker. See
// shaders.js (rendering), geometry_layout.js (unwrap math),
// colormaps.js (palettes), h5_worker.js (I/O).

import { PMT_VS, PMT_FS } from './shaders.js';
import {
  PLASMA_STOPS, VIRIDIS_STOPS, VIRIDIS_R_STOPS, INFERNO_R_STOPS,
  hashHue, plasmaRGB, viridisRGB, viridisRRGB, hsl2rgb,
} from './colormaps.js';
import { computeLayout } from './geometry_layout.js';

// ── Globals ─────────────────────────────────────────────────────────────
let worker = null;
let nEvents = 0, nSensors = 0;
let detectorType = '', shape = {};
let sensorPositions = null;        // Float32Array(nSensors * 3)
let layout = null;                 // from computeLayout()

let curEvent = 0;
let evtBundle = null;              // decoded {sensor, inst, seg, labl, t0, srcIdx, ...}

// Per-PMT derived arrays (length nSensors).
let pmtPE = null;                  // summed PE per sensor
let pmtT = null;                   // earliest T per sensor (NaN if no hit)
let pmtDomParticle = null;         // Int32Array  (argmax-over-inst per sensor; -1 if none)
let pmtHasSignal = null;           // Uint8Array
let pmtArrivalT = null;            // Float32Array — same value used for the 3D sweep shader

// inst lookups.
let particleToSensor = null;       // Array(nParticles) of Map<sensor, PE>
let particleTotals = null;         // Float32Array(nParticles)  total PE per particle
let particleAncestor = null;       // Int32Array(nParticles)  derived from per_track
let particleInteraction = null;    // Int32Array(nParticles)

// UI state.
let curView = 'pmts';              // 'pmts' | 'seg'   (exclusive 3D view)
let curField = 'charge';           // 'charge' | 'time'  (continuous field)
let curLabel = 'none';             // 'none' | 'particle' | 'category' | 'track' | 'pdg' | 'beta' | 'ncher'
let logScale = true;
let percMin = 1, percMax = 99;
let manualVmin = null, manualVmax = null;
let cmapName = 'auto';
// Selection state. Only one is ever set at a time.
//   selectedParticle : specific particle index (used when Label = None/Particle)
//   selectedGroup    : { kind: 'category'|'ancestor'|'interaction', id: int }
let selectedParticle = null;
let selectedGroup = null;
let showEmpty = true;
let showMesh = true;
let pmtSize = 10;
let autoRotate = true;

// Time sweep.
let sweepOn = false;
let sweepPlaying = false;
let simTime = 0;
let simTMin = 0, simTMax = 0;              // currently-active-view range
let pmtTRange = [0, 1], segTRange = [0, 1];
let sweepSpeed = 1.0;
// Quantile-T scope: 'off' | 'pmts' | 'seg' | 'both'. Default = PMTs only
// (the event display — where it's most visually useful); segments stay in
// raw ns so the user can see the physical time arithmetic.
let quantileScope = 'pmts';
const quantilePMT = () => quantileScope === 'pmts' || quantileScope === 'both';
const quantileSeg = () => quantileScope === 'seg'  || quantileScope === 'both';
// In 'both' mode, PMT and segment times share one quantile map so the same
// physical time maps to the same rank on both meshes. null otherwise.
let unionQMap = null;

// Three.js.
let renderer, scene, camera, controls;
let pmtGeo, pmtMat, pmtMesh;
let segGeo, segMat, segMesh;
let outlineMesh = null;
let lastFrameTime = 0;

// Colormap textures.
let texPlasma, texViridis, texViridisR, texInfernoR;

// 2D.
let c2d, ctx2d;

// ── Small helpers ───────────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }
function show(id) { $(id).classList.add('visible'); }
function hide(id) { $(id).classList.remove('visible'); }
function setStatus(s) { $('status').textContent = s || ''; }

function showOverlay(m) { $('overlayMsg').textContent = m; show('overlay'); }
function hideOverlay() { hide('overlay'); }

function showToast(msg) {
  let t = $('toast');
  if (!t) { t = document.createElement('div'); t.id = 'toast'; document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.add('visible');
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.classList.remove('visible'), 1500);
}

function currentCmapName() {
  if (cmapName !== 'auto') return cmapName;
  if (curLabel === 'beta' || curLabel === 'ncher') return 'viridis';
  return curField === 'charge' ? 'plasma' : 'viridis_r';
}

function currentCmapRGB(t) {
  const name = currentCmapName();
  if (name === 'plasma') return plasmaRGB(t);
  if (name === 'viridis') return viridisRGB(t);
  if (name === 'viridis_r') return viridisRRGB(t);
  return viridisRGB(t);
}

function currentCmapTex() {
  const name = currentCmapName();
  if (name === 'plasma') return texPlasma;
  if (name === 'viridis') return texViridis;
  if (name === 'viridis_r') return texViridisR;
  if (name === 'inferno_r') return texInfernoR;
  return texViridis;
}

function makeCmapTex(stops) {
  const cv = document.createElement('canvas');
  cv.width = 256; cv.height = 1;
  const cx = cv.getContext('2d');
  const gr = cx.createLinearGradient(0, 0, 256, 0);
  for (const [t, c] of stops) gr.addColorStop(t, typeof c === 'string' ? c : `rgb(${c[0]},${c[1]},${c[2]})`);
  cx.fillStyle = gr; cx.fillRect(0, 0, 256, 1);
  const tx = new THREE.CanvasTexture(cv);
  tx.minFilter = THREE.LinearFilter; tx.magFilter = THREE.LinearFilter;
  return tx;
}

// ── Worker ──────────────────────────────────────────────────────────────
function createWorker() {
  worker = new Worker('h5_worker.js', { type: 'module' });
  worker.addEventListener('error', (e) => {
    console.error('[worker] error:', e.message, e);
    setStatus('worker crash: ' + e.message);
  });
  worker.addEventListener('messageerror', (e) => {
    console.error('[worker] messageerror:', e);
  });
}

function workerCall(action, data, timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (fn, arg) => { if (settled) return; settled = true;
      worker.removeEventListener('message', handler);
      clearTimeout(tid); fn(arg); };
    const handler = (e) => {
      if (e.data.action === 'error') finish(reject, new Error(e.data.message));
      else finish(resolve, e.data);
    };
    const tid = setTimeout(
      () => finish(reject, new Error(`${action} timed out after ${timeoutMs}ms`)),
      timeoutMs);
    worker.addEventListener('message', handler);
    worker.postMessage({ action, ...data });
  });
}

// ── Percentile / normalization ──────────────────────────────────────────
// Returns a normalized [0,1] float per input, following LUCiD conventions:
//   - Optional `mask` (1-byte/bool per index) restricts percentile computation
//     to signal-bearing entries. Masked-out entries still get a normalized
//     value (for continuity), but don't influence vmin/vmax.
//   - Time fields (`isTime=true`): linear scale, percentile over ALL finite
//     values in mask (negatives allowed — v4 stores sensor.T in absolute
//     detector frame = G4 + t0_interaction, so hit times can be large
//     positive or large negative depending on the randomly drawn t0).
//     No-signal rows are excluded upstream via the PE > 0 mask.
//   - Non-time fields: the LUCiD notebook's rule applies — percentile over
//     positive values only, optional log scale.
function normalizeValues(values, opts) {
  const { isTime, pMin, pMax, mVmin, mVmax, mask } = opts;
  const isLog = !!opts.isLog && !isTime;   // log disabled for time
  const n = values.length;
  const out = new Float32Array(n);

  const pool = [];
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (!isTime && v <= 0) continue;        // positive-only for charge-like
    pool.push(v);
  }

  let vmin, vmax;
  if (pool.length === 0) {
    vmin = isLog ? 0.1 : 0;
    vmax = 1;
  } else {
    pool.sort((a, b) => a - b);
    const iMin = Math.max(0, Math.min(pool.length - 1, Math.floor(pool.length * pMin / 100)));
    const iMax = Math.max(0, Math.min(pool.length - 1, Math.ceil(pool.length * pMax / 100) - 1));
    vmin = pool[iMin];
    vmax = pool[Math.max(iMax, iMin)];
  }
  if (mVmin != null && Number.isFinite(mVmin)) vmin = mVmin;
  if (mVmax != null && Number.isFinite(mVmax)) vmax = mVmax;
  if (pMin === 0 && !isTime) vmin = isLog ? 0.1 : 0.001;
  if (vmax <= vmin) vmax = isLog ? vmin * 10 : vmin + 1;

  if (isLog) {
    const logMin = Math.log10(Math.max(vmin, 1e-12));
    const logMax = Math.log10(Math.max(vmax, vmin * 10));
    const r = logMax - logMin || 1;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (!Number.isFinite(v) || v <= 0) { out[i] = 0; continue; }
      const clipped = Math.max(vmin, Math.min(vmax, v));
      out[i] = Math.max(0, Math.min(1, (Math.log10(clipped) - logMin) / r));
    }
  } else {
    const r = vmax - vmin || 1;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) { out[i] = 0; continue; }
      const clipped = Math.max(vmin, Math.min(vmax, v));
      out[i] = Math.max(0, Math.min(1, (clipped - vmin) / r));
    }
  }
  return { norm: out, vmin, vmax };
}

// ── PMT derivation from sensor / inst ──────────────────────────────────
// v4 writes only signal-bearing sensors into the sparse sensor file (the
// save_sensor_event_v3 mask keeps rows where PE > 0 OR T is a finite
// positive time within a reasonable window). Whatever makes it through,
// we still filter here with PE > 0 to guard against edge cases.
function deriveSensorArrays() {
  pmtPE = new Float32Array(nSensors);
  pmtT = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) pmtT[i] = NaN;
  pmtHasSignal = new Uint8Array(nSensors);

  const s = evtBundle.sensor;
  if (s && s.nHits) {
    for (let i = 0; i < s.nHits; i++) {
      const si = s.sensor_idx[i];
      const pe = s.PE[i];
      pmtPE[si] += pe;
      if (pe > 0) {
        const t = s.T[i];
        if (Number.isNaN(pmtT[si]) || t < pmtT[si]) pmtT[si] = t;
        pmtHasSignal[si] = 1;
      }
    }
  }
}

function buildInstLookups() {
  const n_particles = evtBundle.labl.n_particles || 0;
  particleToSensor = [];
  for (let p = 0; p < n_particles; p++) particleToSensor.push(new Map());
  particleTotals = new Float32Array(n_particles);
  particleAncestor = new Int32Array(n_particles);    for (let p = 0; p < n_particles; p++) particleAncestor[p] = -1;
  particleInteraction = new Int32Array(n_particles); for (let p = 0; p < n_particles; p++) particleInteraction[p] = -1;
  pmtDomParticle = new Int32Array(nSensors);
  for (let i = 0; i < nSensors; i++) pmtDomParticle[i] = -1;

  // Derive per-particle ancestor/interaction from per_track (any track of
  // that particle gives the same value; take the first we see).
  const pt = evtBundle.labl.per_track;
  if (pt && pt.particle_idx) {
    for (let t = 0; t < pt.particle_idx.length; t++) {
      const p = pt.particle_idx[t];
      if (p < 0 || p >= n_particles) continue;
      if (particleAncestor[p] < 0 && pt.ancestor) particleAncestor[p] = pt.ancestor[t];
      if (particleInteraction[p] < 0 && pt.interaction) particleInteraction[p] = pt.interaction[t];
    }
  }

  const i_ = evtBundle.inst;
  if (!i_ || !i_.nHits) return;

  const perSensorBest = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) perSensorBest[i] = -Infinity;

  for (let i = 0; i < i_.nHits; i++) {
    const p = i_.particle_idx[i];
    const s = i_.sensor_idx[i];
    const pe = i_.PE[i];
    if (p < 0 || p >= n_particles) continue;
    particleToSensor[p].set(s, (particleToSensor[p].get(s) || 0) + pe);
    particleTotals[p] += pe;
    if (pe > perSensorBest[s]) {
      perSensorBest[s] = pe;
      pmtDomParticle[s] = p;
    }
  }
}

// ── contVal / catVal field builders ─────────────────────────────────────

function pmtContValArray() {
  const isTime = curField === 'time';
  // Quantile transform in time mode: replace each signal PMT's T with its
  // rank fraction in [0, 1]. This gives a uniform visual spread across the
  // viridis_r colormap even when arrival times cluster heavily.
  if (isTime && quantilePMT()) {
    const q = unionQMap || buildQuantileMapMasked(pmtT, pmtHasSignal);
    const norm = new Float32Array(nSensors);
    for (let i = 0; i < nSensors; i++) {
      if (!pmtHasSignal[i]) { norm[i] = 0; continue; }
      const t = pmtT[i];
      norm[i] = (Number.isFinite(t) && q.has(t)) ? q.get(t) : 0;
    }
    return { norm, vmin: 0, vmax: 1 };
  }
  const src = isTime ? pmtT : pmtPE;
  return normalizeValues(src, {
    isTime,
    isLog: logScale,
    pMin: percMin, pMax: percMax,
    mVmin: manualVmin, mVmax: manualVmax,
    mask: pmtHasSignal,
  });
}

function buildQuantileMapMasked(values, mask) {
  const pts = [];
  for (let i = 0; i < values.length; i++) {
    if (mask && !mask[i]) continue;
    const v = values[i];
    if (Number.isFinite(v)) pts.push(v);
  }
  pts.sort((a, b) => a - b);
  const m = new Map();
  const denom = Math.max(1, pts.length - 1);
  for (let i = 0; i < pts.length; i++) m.set(pts[i], i / denom);
  return m;
}

// Union quantile map for 'both' scope: pool signal-PMT T + all seg times
// into one distribution so identical physical times map to identical ranks.
function refreshUnionQMap() {
  if (quantileScope !== 'both' || !evtBundle || !pmtT || !pmtHasSignal) {
    unionQMap = null;
    return;
  }
  const seg = evtBundle.seg;
  const pool = [];
  for (let i = 0; i < pmtT.length; i++) {
    if (pmtHasSignal[i] && Number.isFinite(pmtT[i])) pool.push(pmtT[i]);
  }
  if (seg && seg.time) {
    for (let i = 0; i < seg.time.length; i++) {
      if (Number.isFinite(seg.time[i])) pool.push(seg.time[i]);
    }
  }
  pool.sort((a, b) => a - b);
  const m = new Map();
  const denom = Math.max(1, pool.length - 1);
  for (let i = 0; i < pool.length; i++) {
    // First occurrence wins — ties across meshes resolve to the same rank.
    if (!m.has(pool[i])) m.set(pool[i], i / denom);
  }
  unionQMap = m;
}

// Shared hue for a category id. Used in 3D (shader HSL), 2D (render2D),
// and the sidebar swatch — so colors agree everywhere.
function categoryHue(c) { return ((c | 0) * 0.13 + 0.07) % 1.0; }

// Hue for the "group" label modes (category, ancestor, interaction).
// Category stays on the fixed-palette hue; ancestor/interaction hash by
// golden ratio so distinct values read as distinct colors.
function groupHue(kind, id) {
  if (kind === 'category') return categoryHue(id);
  return hashHue(id);
}

// For a given particle index, return the "group id" for the active label.
function particleGroupId(p) {
  if (p < 0) return -1;
  if (curLabel === 'category') {
    const c = evtBundle.labl.per_particle.category;
    return c ? c[p] : -1;
  }
  if (curLabel === 'ancestor')    return particleAncestor ? particleAncestor[p] : -1;
  if (curLabel === 'interaction') return particleInteraction ? particleInteraction[p] : -1;
  return p;
}

function pmtCatValArray() {
  const out = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) {
    const p = pmtDomParticle[i];
    if (p < 0) { out[i] = 0; continue; }
    if (curLabel === 'particle') {
      out[i] = hashHue(p);
    } else {
      const id = particleGroupId(p);
      out[i] = id < 0 ? 0 : groupHue(curLabel, id);
    }
  }
  return out;
}

function segContValArrays() {
  // Seg continuous source follows Field (since seg-only labels are gone):
  //   charge → edep,  time → time.
  const seg = evtBundle.seg;
  if (!seg || !seg.n) return { contPerSeg: null, vmin: 0, vmax: 1 };
  const isTimeField = curField === 'time';
  const field = isTimeField ? seg.time : seg.edep;
  if (!field) return { contPerSeg: new Float32Array(seg.n), vmin: 0, vmax: 1 };
  const f32 = field instanceof Float32Array ? field : Float32Array.from(field);

  // Quantile for time: same treatment as PMTs; union map when scope=both.
  if (isTimeField && quantileSeg()) {
    const q = unionQMap || buildQuantileMapMasked(f32, null);
    const norm = new Float32Array(seg.n);
    for (let i = 0; i < seg.n; i++) {
      const v = f32[i];
      norm[i] = (Number.isFinite(v) && q.has(v)) ? q.get(v) : 0;
    }
    return { contPerSeg: norm, vmin: 0, vmax: 1 };
  }

  const { norm, vmin, vmax } = normalizeValues(f32, {
    isTime: isTimeField,
    isLog: logScale,
    pMin: percMin, pMax: percMax,
    mVmin: manualVmin, mVmax: manualVmax,
    mask: null,
  });
  return { contPerSeg: norm, vmin, vmax };
}

function segCatValArrays() {
  const seg = evtBundle.seg;
  if (!seg || !seg.n) return null;
  const per_track = evtBundle.labl.per_track;
  const per_particle = evtBundle.labl.per_particle;
  const out = new Float32Array(seg.n);
  for (let i = 0; i < seg.n; i++) {
    const t = seg.track_idx[i];
    if (curLabel === 'particle') {
      const pidx = per_track.particle_idx ? per_track.particle_idx[t] : -1;
      out[i] = pidx >= 0 ? hashHue(pidx) : 0;
    } else if (curLabel === 'category') {
      const pidx = per_track.particle_idx ? per_track.particle_idx[t] : -1;
      const cat = per_particle.category;
      out[i] = (pidx >= 0 && cat) ? categoryHue(cat[pidx]) : 0;
    } else if (curLabel === 'ancestor') {
      const a = per_track.ancestor ? per_track.ancestor[t] : -1;
      out[i] = a >= 0 ? hashHue(a) : 0;
    } else if (curLabel === 'interaction') {
      const k = per_track.interaction ? per_track.interaction[t] : -1;
      out[i] = k >= 0 ? hashHue(k) : 0;
    } else {
      out[i] = 0;
    }
  }
  return out;
}

// ── Build 3D PMT mesh ──────────────────────────────────────────────────
function buildPMTs() {
  if (pmtMesh) {
    scene.remove(pmtMesh);
    pmtGeo.dispose();
    if (pmtMat) pmtMat.dispose();
    pmtMesh = null;
  }

  const pos = sensorPositions;                      // Float32Array(n*3)
  const contVal = new Float32Array(nSensors);
  const catVal = new Float32Array(nSensors);
  const hl = new Float32Array(nSensors);
  const arrivalT = new Float32Array(nSensors);
  const hasSig = new Float32Array(nSensors);

  for (let i = 0; i < nSensors; i++) hasSig[i] = pmtHasSignal[i];

  // arrivalT attribute: per-PMT time (NaN PMTs get +Infinity so sweep never
  // reveals them). Quantile transform replaces T with its rank fraction in
  // [0, 1], which gives the sweep uniform density even when event has a
  // long decay tail. Sweep range tracks whichever space we're in.
  // arrivalT respects the Quantile-T scope:
  //   off:  raw ns
  //   pmts: per-PMT rank
  //   seg:  raw ns (PMT not affected by 'seg' scope)
  //   both: union rank shared with segments
  const qMap = quantilePMT() ? (unionQMap || buildQuantileMap(pmtT)) : null;
  for (let i = 0; i < nSensors; i++) {
    const t = pmtT[i];
    if (!Number.isFinite(t)) { arrivalT[i] = 1e30; continue; }
    arrivalT[i] = qMap ? qMap.get(t) : t;
  }
  pmtTRange = qMap ? [0, 1] : [minFinite(pmtT), maxFinite(pmtT)];
  pmtArrivalT = arrivalT;   // shared with 2D sweep rendering

  pmtGeo = new THREE.BufferGeometry();
  pmtGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const attrs = {
    contVal: new THREE.BufferAttribute(contVal, 1),
    catVal:  new THREE.BufferAttribute(catVal, 1),
    hl:      new THREE.BufferAttribute(hl, 1),
    arrivalT: new THREE.BufferAttribute(arrivalT, 1),
    hasSignal: new THREE.BufferAttribute(hasSig, 1),
  };
  attrs.hl.setUsage(THREE.DynamicDrawUsage);
  attrs.contVal.setUsage(THREE.DynamicDrawUsage);
  attrs.catVal.setUsage(THREE.DynamicDrawUsage);
  for (const k in attrs) pmtGeo.setAttribute(k, attrs[k]);

  pmtMat = new THREE.ShaderMaterial({
    vertexShader: PMT_VS, fragmentShader: PMT_FS,
    uniforms: {
      cmap:      { value: currentCmapTex() },
      colorMode: { value: 0.0 },
      corrOn:    { value: 0.0 },
      sweepOn:   { value: 0.0 },
      simTime:   { value: 0 },
      sweepEps:  { value: 0.5 },
      pmtSize:   { value: pmtSize },
      emptyGray: { value: showEmpty ? 1.0 : 0.0 },
      emptyColor:{ value: new THREE.Color(0x4c4c4c) },
    },
    transparent: true, depthWrite: false,
  });

  pmtMesh = new THREE.Points(pmtGeo, pmtMat);
  pmtMesh.visible = (curView === 'pmts');
  scene.add(pmtMesh);

  updatePMTColors();
  // Do NOT reframe the camera here — that would reset orbit on quantile
  // toggles, pmtSize changes, etc. Camera is (re)framed in loadEvent only.
}

function minFinite(arr) { let m = Infinity; for (const v of arr) if (Number.isFinite(v) && v < m) m = v; return m === Infinity ? 0 : m; }
function maxFinite(arr) { let m = -Infinity; for (const v of arr) if (Number.isFinite(v) && v > m) m = v; return m === -Infinity ? 1 : m; }

function buildQuantileMap(values) {
  // Map t → its quantile rank in [0,1] among finite values.
  const pts = [];
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) pts.push(values[i]);
  pts.sort((a, b) => a - b);
  const m = new Map();
  for (let i = 0; i < pts.length; i++) m.set(pts[i], i / Math.max(1, pts.length - 1));
  return m;
}

// ── Build segment mesh ─────────────────────────────────────────────────
// WebGL lines render at 1px and get drowned out by PMTs. Instead we emit
// K points per segment along its trajectory and render as fat sprites via
// the same PMT shader. Visible and fast (~1000 segs × 6 pts = 6k points).
const SEG_POINTS_PER = 6;

function buildSegments() {
  if (segMesh) {
    scene.remove(segMesh);
    segGeo.dispose();
    if (segMat) segMat.dispose();
    segMesh = null;
  }
  const seg = evtBundle.seg;
  if (!seg || !seg.n) return;

  const n = seg.n;
  const K = SEG_POINTS_PER;
  const N = n * K;
  const pos = new Float32Array(N * 3);
  const contVal = new Float32Array(N);
  const catVal = new Float32Array(N);
  const hl = new Float32Array(N);
  const arrivalT = new Float32Array(N);
  const hasSig = new Float32Array(N);

  // Segment arrivalT respects scope: rank when seg-quantiled (or 'both'),
  // raw ns otherwise.
  const segQMap = quantileSeg() ? (unionQMap || buildQuantileMap(seg.time)) : null;
  if (segQMap) {
    segTRange = [0, 1];
  } else {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = seg.time[i];
      if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
    }
    segTRange = [mn === Infinity ? 0 : mn, mx === -Infinity ? 1 : mx];
  }

  for (let i = 0; i < n; i++) {
    const sx = seg.start_x[i], sy = seg.start_y[i], sz = seg.start_z[i];
    const ex = seg.end_x[i],   ey = seg.end_y[i],   ez = seg.end_z[i];
    const t = seg.time[i];
    const tMapped = segQMap ? segQMap.get(t) : t;
    for (let k = 0; k < K; k++) {
      const f = (k + 0.5) / K;
      const p = i * K + k;
      pos[p*3]     = sx + f * (ex - sx);
      pos[p*3 + 1] = sy + f * (ey - sy);
      pos[p*3 + 2] = sz + f * (ez - sz);
      arrivalT[p] = tMapped;
      hasSig[p] = 1.0;
    }
  }

  segGeo = new THREE.BufferGeometry();
  segGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const contAttr = new THREE.BufferAttribute(contVal, 1);
  contAttr.setUsage(THREE.DynamicDrawUsage);
  segGeo.setAttribute('contVal', contAttr);
  const catAttr = new THREE.BufferAttribute(catVal, 1);
  catAttr.setUsage(THREE.DynamicDrawUsage);
  segGeo.setAttribute('catVal', catAttr);
  const hlAttr = new THREE.BufferAttribute(hl, 1);
  hlAttr.setUsage(THREE.DynamicDrawUsage);
  segGeo.setAttribute('hl', hlAttr);
  segGeo.setAttribute('arrivalT', new THREE.BufferAttribute(arrivalT, 1));
  segGeo.setAttribute('hasSignal', new THREE.BufferAttribute(hasSig, 1));

  segMat = new THREE.ShaderMaterial({
    vertexShader: PMT_VS, fragmentShader: PMT_FS,
    uniforms: {
      cmap:      { value: currentCmapTex() },
      colorMode: { value: 0.0 },
      corrOn:    { value: 0.0 },
      sweepOn:   { value: 0.0 },
      simTime:   { value: 0 },
      sweepEps:  { value: 0.5 },
      pmtSize:   { value: Math.max(3, pmtSize * 0.6) },   // slightly smaller than PMT discs
      emptyGray: { value: 0.0 },                          // segments always have signal
      emptyColor:{ value: new THREE.Color(0x4c4c4c) },
    },
    transparent: true, depthWrite: false,
  });

  segMesh = new THREE.Points(segGeo, segMat);
  segMesh.visible = (curView === 'seg');
  scene.add(segMesh);

  updateSegmentColors();
}

// ── Detector outline (cylinder/box/sphere wireframe) ───────────────────
function buildOutline() {
  if (outlineMesh) { scene.remove(outlineMesh); outlineMesh = null; }
  if (!showMesh) return;
  const t = (detectorType || '').toLowerCase();
  const grp = new THREE.Group();
  const mat = new THREE.LineBasicMaterial({ color: 0x28394a, transparent: true, opacity: 0.5 });
  if (t === 'cylinder') {
    const r = shape.r, hh = shape.halfH;
    const seg = 64;
    const top = new Float32Array(seg * 2 * 3), bot = new Float32Array(seg * 2 * 3);
    for (let i = 0; i < seg; i++) {
      const a0 = 2 * Math.PI * i / seg, a1 = 2 * Math.PI * (i + 1) / seg;
      top[i*6] = r*Math.cos(a0); top[i*6+1] = r*Math.sin(a0); top[i*6+2] = hh;
      top[i*6+3] = r*Math.cos(a1); top[i*6+4] = r*Math.sin(a1); top[i*6+5] = hh;
      bot[i*6] = r*Math.cos(a0); bot[i*6+1] = r*Math.sin(a0); bot[i*6+2] = -hh;
      bot[i*6+3] = r*Math.cos(a1); bot[i*6+4] = r*Math.sin(a1); bot[i*6+5] = -hh;
    }
    const gTop = new THREE.BufferGeometry(); gTop.setAttribute('position', new THREE.BufferAttribute(top, 3));
    const gBot = new THREE.BufferGeometry(); gBot.setAttribute('position', new THREE.BufferAttribute(bot, 3));
    grp.add(new THREE.LineSegments(gTop, mat));
    grp.add(new THREE.LineSegments(gBot, mat));
    // Verticals
    const verts = new Float32Array(8 * 2 * 3);
    for (let i = 0; i < 8; i++) {
      const a = 2 * Math.PI * i / 8;
      verts[i*6] = r*Math.cos(a); verts[i*6+1] = r*Math.sin(a); verts[i*6+2] = -hh;
      verts[i*6+3] = r*Math.cos(a); verts[i*6+4] = r*Math.sin(a); verts[i*6+5] = hh;
    }
    const gV = new THREE.BufferGeometry(); gV.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    grp.add(new THREE.LineSegments(gV, mat));
  } else if (t === 'box') {
    const hL = shape.L/2, hW = shape.W/2, hH = shape.H/2;
    const c = [
      [-hL,-hW,-hH],[hL,-hW,-hH],[hL,hW,-hH],[-hL,hW,-hH],
      [-hL,-hW, hH],[hL,-hW, hH],[hL,hW, hH],[-hL,hW, hH],
    ];
    const e = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const v = new Float32Array(e.length * 2 * 3);
    for (let i = 0; i < e.length; i++) {
      const [a,b] = e[i];
      v[i*6]=c[a][0]; v[i*6+1]=c[a][1]; v[i*6+2]=c[a][2];
      v[i*6+3]=c[b][0]; v[i*6+4]=c[b][1]; v[i*6+5]=c[b][2];
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(v, 3));
    grp.add(new THREE.LineSegments(g, mat));
  } else if (t === 'sphere') {
    const r = shape.r;
    // Three great circles (xy, yz, xz planes)
    const seg = 96;
    const make = (axis) => {
      const v = new Float32Array(seg * 2 * 3);
      for (let i = 0; i < seg; i++) {
        const a0 = 2 * Math.PI * i / seg, a1 = 2 * Math.PI * (i + 1) / seg;
        let p0, p1;
        if (axis === 0) { p0 = [r*Math.cos(a0), r*Math.sin(a0), 0]; p1 = [r*Math.cos(a1), r*Math.sin(a1), 0]; }
        else if (axis === 1) { p0 = [0, r*Math.cos(a0), r*Math.sin(a0)]; p1 = [0, r*Math.cos(a1), r*Math.sin(a1)]; }
        else { p0 = [r*Math.cos(a0), 0, r*Math.sin(a0)]; p1 = [r*Math.cos(a1), 0, r*Math.sin(a1)]; }
        v[i*6]=p0[0]; v[i*6+1]=p0[1]; v[i*6+2]=p0[2];
        v[i*6+3]=p1[0]; v[i*6+4]=p1[1]; v[i*6+5]=p1[2];
      }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(v, 3));
      return new THREE.LineSegments(g, mat);
    };
    grp.add(make(0)); grp.add(make(1)); grp.add(make(2));
  }
  outlineMesh = grp;
  scene.add(outlineMesh);
}

function frameCamera() {
  if (!sensorPositions) return;
  let xmn=Infinity,xmx=-Infinity,ymn=Infinity,ymx=-Infinity,zmn=Infinity,zmx=-Infinity;
  for (let i = 0; i < nSensors; i++) {
    const x = sensorPositions[i*3], y = sensorPositions[i*3+1], z = sensorPositions[i*3+2];
    if (x<xmn)xmn=x; if (x>xmx)xmx=x;
    if (y<ymn)ymn=y; if (y>ymx)ymx=y;
    if (z<zmn)zmn=z; if (z>zmx)zmx=z;
  }
  const cx=(xmn+xmx)/2, cy=(ymn+ymx)/2, cz=(zmn+zmx)/2;
  const rxy = Math.max(xmx - xmn, ymx - ymn) / 2;   // horizontal half-extent
  const rz  = (zmx - zmn) / 2;
  // Camera lives outside the detector in the xy plane, slightly elevated.
  // Looking across the detector's symmetry axis, with +z pointing up.
  // Distance ~4× of the larger half-extent so the whole detector fits
  // comfortably with room to spare (camera FOV is 40°).
  const dist = Math.max(rxy, rz) * 4.0;
  const azim = Math.PI * 0.25;                       // 45° around z
  const elev = 0.3;                                  // ~17° tilt up
  camera.near = Math.max(dist * 0.001, 0.05);
  camera.far = dist * 10;
  camera.position.set(
    cx + dist * Math.cos(elev) * Math.cos(azim),
    cy + dist * Math.cos(elev) * Math.sin(azim),
    cz + dist * Math.sin(elev),
  );
  controls.target.set(cx, cy, cz);
  camera.updateProjectionMatrix();
}

// ── Color update for current label ─────────────────────────────────────
function updatePMTColors() {
  if (!pmtMat || !pmtGeo) return;
  const isCat = (curLabel !== 'none');
  pmtMat.uniforms.colorMode.value = isCat ? 1.0 : 0.0;
  pmtMat.uniforms.cmap.value = currentCmapTex();

  if (isCat) {
    const cat = pmtCatValArray();
    pmtGeo.attributes.catVal.array.set(cat);
    pmtGeo.attributes.catVal.needsUpdate = true;
  } else {
    // PMTs always get continuous color from Charge/Time, even when Label is a
    // segment-only mode (track, pdg, β, ncher).
    const { norm } = pmtContValArrayForField();
    pmtGeo.attributes.contVal.array.set(norm);
    pmtGeo.attributes.contVal.needsUpdate = true;
  }
  drawLegend();
}

function pmtContValArrayForField() {
  // Always keyed to current Charge/Time toggle, not label.
  return pmtContValArray();
}

function updateSegmentColors() {
  if (!segMat || !segGeo) return;
  const seg = evtBundle.seg;
  if (!seg || !seg.n) return;

  const isCat = (curLabel !== 'none');
  segMat.uniforms.colorMode.value = isCat ? 1.0 : 0.0;
  segMat.uniforms.cmap.value = currentCmapTex();
  const K = SEG_POINTS_PER;

  if (isCat) {
    const cat = segCatValArrays();
    const out = segGeo.attributes.catVal.array;
    for (let i = 0; i < seg.n; i++) {
      const v = cat[i];
      for (let k = 0; k < K; k++) out[i * K + k] = v;
    }
    segGeo.attributes.catVal.needsUpdate = true;
  } else {
    const { contPerSeg } = segContValArrays();
    const out = segGeo.attributes.contVal.array;
    for (let i = 0; i < seg.n; i++) {
      const v = contPerSeg[i];
      for (let k = 0; k < K; k++) out[i * K + k] = v;
    }
    segGeo.attributes.contVal.needsUpdate = true;
  }
}

// ── Correspondence: isolate the selected item ─────────────────────────
// A selection is either a specific particle (when label ≠ category) or
// a category (when label = category). Both resolve to a set of particles
// whose inst contributions we union per sensor.
function currentParticleSet() {
  if (!evtBundle) return null;
  if (selectedGroup) {
    const n_particles = evtBundle.labl.n_particles || 0;
    const out = [];
    if (selectedGroup.kind === 'category') {
      const cat = evtBundle.labl.per_particle.category;
      if (!cat) return null;
      for (let p = 0; p < cat.length; p++) if (cat[p] === selectedGroup.id) out.push(p);
    } else if (selectedGroup.kind === 'ancestor') {
      for (let p = 0; p < n_particles; p++) if (particleAncestor[p] === selectedGroup.id) out.push(p);
    } else if (selectedGroup.kind === 'interaction') {
      for (let p = 0; p < n_particles; p++) if (particleInteraction[p] === selectedGroup.id) out.push(p);
    }
    return out;
  }
  if (selectedParticle != null) return [selectedParticle];
  return null;
}

function unionContributions(particleIds) {
  if (!particleToSensor || !particleIds) return null;
  const out = new Map();
  for (const p of particleIds) {
    const m = particleToSensor[p];
    if (!m) continue;
    for (const [s, pe] of m) out.set(s, (out.get(s) || 0) + pe);
  }
  return out;
}

function applyCorrespondence() {
  if (!pmtGeo) return;
  const pmtHL = pmtGeo.attributes.hl.array;
  const segHL = segGeo ? segGeo.attributes.hl.array : null;
  pmtHL.fill(0);
  if (segHL) segHL.fill(0);

  const particleSet = currentParticleSet();
  const corrActive = particleSet != null && particleSet.length > 0;
  pmtMat.uniforms.corrOn.value = corrActive ? 1.0 : 0.0;
  if (segMat) segMat.uniforms.corrOn.value = corrActive ? 1.0 : 0.0;

  if (corrActive) {
    // PMT contributions summed across all particles in the current set.
    const map = unionContributions(particleSet);
    if (map) {
      let maxPE = 0;
      for (const v of map.values()) if (v > maxPE) maxPE = v;
      if (maxPE > 0) {
        for (const [s, pe] of map) {
          if (!(pe > 0)) continue;
          const frac = Math.min(1, pe / maxPE);
          pmtHL[s] = 0.35 + 0.65 * Math.sqrt(frac);
        }
      }
    }
    // Segments: binary highlight if the track belongs to any particle in
    // the selected set.
    const seg = evtBundle.seg;
    const per_track = evtBundle.labl.per_track;
    if (seg && seg.n && segHL && per_track && per_track.particle_idx) {
      const K = SEG_POINTS_PER;
      const setLookup = new Set(particleSet);
      for (let i = 0; i < seg.n; i++) {
        const t = seg.track_idx[i];
        const p = per_track.particle_idx[t];
        const v = setLookup.has(p) ? 1.0 : 0.0;
        for (let k = 0; k < K; k++) segHL[i * K + k] = v;
      }
    }
  } // end corrActive
  pmtGeo.attributes.hl.needsUpdate = true;
  if (segGeo) segGeo.attributes.hl.needsUpdate = true;
}

// ── Sidebar (label-aware) ──────────────────────────────────────────────
function buildSidebar() {
  const list = $('particleList');
  list.innerHTML = '';
  const title = $('sidebarTitle');
  if (title) title.textContent = curLabel === 'category' ? 'CATEGORIES' : 'PARTICLES';
  const n = evtBundle.labl.n_particles || 0;
  if (n === 0) {
    list.innerHTML = '<div class="event-meta-row" style="padding:8px"><span class="k">(none)</span></div>';
  } else if (curLabel === 'category' || curLabel === 'ancestor' || curLabel === 'interaction') {
    buildSidebarGroups(list, curLabel);
  } else {
    buildSidebarParticles(list, n);
  }

  // Event meta.
  const meta = $('eventMeta');
  meta.innerHTML = '';
  const addRow = (k, v) => {
    const r = document.createElement('div');
    r.className = 'event-meta-row';
    r.innerHTML = `<span class="k">${k}</span><span class="v">${v}</span>`;
    meta.appendChild(r);
  };
  addRow('src idx', String(evtBundle.srcIdx ?? curEvent));
  addRow('t0 (ns)', (evtBundle.t0 || 0).toFixed(2));
  addRow('contained', evtBundle.contained ? 'true' : 'false');
  addRow('n hits',      String(evtBundle.sensor.nHits));
  addRow('n particles', String(evtBundle.labl.n_particles));
  addRow('n tracks',    String(evtBundle.labl.n_tracks));
  addRow('n segments',  String(evtBundle.seg.n));

  // v5 per_interaction summary: one row per source interaction (one G4
  // event's primaries bundled). Shows interaction count + per-row source
  // type, primary PDG list, and neutrino probe info for GENIE rows.
  const pi = evtBundle.labl.per_interaction;
  if (pi && pi.t0 && pi.t0.length) {
    addRow('n interactions', String(pi.t0.length));
    for (let i = 0; i < pi.t0.length; i++) {
      const src = pi.source_type[i] === 1 ? 'genie' : 'gun';
      const s0 = pi.primary_pdgs_offsets[i];
      const s1 = pi.primary_pdgs_offsets[i + 1];
      const pdgs = Array.from(pi.primary_pdgs_data.slice(s0, s1)).join(',');
      let line = `[${src}] t0=${pi.t0[i].toFixed(1)} ns  pdgs=[${pdgs}]`;
      if (pi.source_type[i] === 1) {
        line += ` nu=${pi.neutrino_pdg[i]}@${pi.neutrino_energy_MeV[i].toFixed(0)} MeV`;
      }
      if (pi.contained) {
        line += ` contained=${pi.contained[i] ? 'true' : 'false'}`;
      }
      addRow(`int ${i}`, line);
    }
  }

  renderSelectionInfo();
}

function buildSidebarParticles(list, n) {
  for (let p = 0; p < n; p++) {
    const row = document.createElement('div');
    row.className = 'particle-row';
    if (p === selectedParticle) row.classList.add('selected');
    const swatch = document.createElement('span');
    swatch.className = 'particle-swatch';
    swatch.style.background = particleSwatch(p);
    const label = document.createElement('span');
    label.className = 'particle-label';
    label.textContent = particleLabel(p);
    const meta = document.createElement('span');
    meta.className = 'particle-meta';
    meta.textContent = formatPE(particleTotals[p] || 0);
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(meta);
    row.addEventListener('click', () => {
      selectedGroup = null;
      selectedParticle = (selectedParticle === p) ? null : p;
      buildSidebar();
      applyCorrespondence();
      render2D();
    });
    list.appendChild(row);
  }
}

// List categories / ancestors / interactions. One row per distinct group id,
// with the total PE and particle count for that group.
function buildSidebarGroups(list, kind) {
  const n_particles = evtBundle.labl.n_particles || 0;
  const groups = new Map();   // id → { n, pe, parts: [] }
  for (let p = 0; p < n_particles; p++) {
    let id;
    if (kind === 'category') {
      const c = evtBundle.labl.per_particle.category;
      id = c ? c[p] : -1;
    } else if (kind === 'ancestor')    id = particleAncestor ? particleAncestor[p] : -1;
    else if (kind === 'interaction')   id = particleInteraction ? particleInteraction[p] : -1;
    if (id == null || id < 0) continue;
    if (!groups.has(id)) groups.set(id, { n: 0, pe: 0, parts: [] });
    const g = groups.get(id);
    g.n += 1;
    g.pe += particleTotals[p] || 0;
    g.parts.push(p);
  }
  const ordered = [...groups.keys()].sort((a, b) => a - b);
  for (const id of ordered) {
    const row = document.createElement('div');
    row.className = 'particle-row';
    if (selectedGroup && selectedGroup.kind === kind && selectedGroup.id === id) row.classList.add('selected');
    const swatch = document.createElement('span');
    swatch.className = 'particle-swatch';
    const [r,g,b] = hsl2rgb(groupHue(kind, id), 0.78, 0.55);
    swatch.style.background = `rgb(${r},${g},${b})`;
    const label = document.createElement('span');
    label.className = 'particle-label';
    label.textContent = groupLabelText(kind, id, groups.get(id).n);
    const meta = document.createElement('span');
    meta.className = 'particle-meta';
    meta.textContent = formatPE(groups.get(id).pe);
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(meta);
    row.addEventListener('click', () => {
      selectedParticle = null;
      const already = selectedGroup && selectedGroup.kind === kind && selectedGroup.id === id;
      selectedGroup = already ? null : { kind, id };
      buildSidebar();
      applyCorrespondence();
      render2D();
    });
    list.appendChild(row);
  }
}

function groupLabelText(kind, id, n) {
  if (kind === 'category')    return `${CAT_NAMES[id] || 'cat'+id} · n=${n}`;
  if (kind === 'ancestor')    return `ancestor ${id} · n=${n}`;
  if (kind === 'interaction') return `interaction ${id} · n=${n}`;
  return String(id);
}

function renderSelectionInfo() {
  const info = $('selectionInfo');
  if (!info) return;
  info.innerHTML = '';
  if (selectedParticle != null) {
    renderParticleInfo(info, selectedParticle);
  } else if (selectedGroup != null) {
    renderGroupInfo(info, selectedGroup);
  } else {
    info.innerHTML = '<div class="event-meta-row" style="padding:4px 8px"><span class="k" style="color:#444">(nothing selected)</span></div>';
  }
}

function renderParticleInfo(info, p) {
  const labl = evtBundle.labl;
  const cat = labl.per_particle.category;
  const cont = labl.per_particle.contained;
  const per_track = labl.per_track;
  const add = (k, v) => {
    const r = document.createElement('div');
    r.className = 'event-meta-row';
    r.innerHTML = `<span class="k">${k}</span><span class="v">${v}</span>`;
    info.appendChild(r);
  };
  const head = document.createElement('div');
  head.className = 'selection-head';
  head.innerHTML = `<span class="selection-swatch" style="background:${particleSwatch(p)}"></span>
                    <span>${particleLabel(p)}</span>`;
  info.appendChild(head);
  if (cat) add('category', CAT_NAMES[cat[p]] || String(cat[p]));
  if (cont) add('contained', cont[p] ? 'true' : 'false');
  // Tracks belonging to this particle.
  let nTracks = 0, nCher = 0, initE = 0;
  const pdgSet = new Set();
  if (per_track && per_track.particle_idx) {
    for (let t = 0; t < per_track.particle_idx.length; t++) {
      if (per_track.particle_idx[t] === p) {
        nTracks++;
        if (per_track.n_cherenkov) nCher += per_track.n_cherenkov[t];
        if (per_track.initial_energy) initE = Math.max(initE, per_track.initial_energy[t]);
        if (per_track.pdg) pdgSet.add(per_track.pdg[t]);
      }
    }
  }
  add('Σ PE (inst)', formatPE(particleTotals[p] || 0));
  add('sensors hit', String(particleToSensor[p] ? particleToSensor[p].size : 0));
  add('n tracks', String(nTracks));
  add('n Cherenkov', String(nCher));
  if (initE > 0) add('max init E', initE.toFixed(1) + ' MeV');
  if (pdgSet.size > 0) add('PDG', [...pdgSet].map(pdgName).join(', '));
  // Genealogy chain: categorized ancestors.
  const gen = labl.per_particle.genealogy;
  const off = labl.per_particle.genealogy_offsets;
  if (gen && off && off.length > p + 1) {
    const s = off[p], e = off[p + 1];
    if (e > s) {
      const chain = Array.from(gen.slice(s, e)).join(' → ');
      add('genealogy', chain);
    }
  }
}

function renderGroupInfo(info, group) {
  const { kind, id } = group;
  const n_particles = evtBundle.labl.n_particles || 0;
  const parts = [];
  let pe = 0;
  for (let p = 0; p < n_particles; p++) {
    let gid;
    if (kind === 'category')         { const c = evtBundle.labl.per_particle.category; gid = c ? c[p] : -1; }
    else if (kind === 'ancestor')    { gid = particleAncestor[p]; }
    else if (kind === 'interaction') { gid = particleInteraction[p]; }
    if (gid === id) { parts.push(p); pe += particleTotals[p] || 0; }
  }
  const head = document.createElement('div');
  head.className = 'selection-head';
  const [r,g,b] = hsl2rgb(groupHue(kind, id), 0.78, 0.55);
  const title = groupLabelText(kind, id, parts.length);
  head.innerHTML = `<span class="selection-swatch" style="background:rgb(${r},${g},${b})"></span>
                    <span>${title}</span>`;
  info.appendChild(head);
  const add = (k, v) => {
    const row = document.createElement('div');
    row.className = 'event-meta-row';
    row.innerHTML = `<span class="k">${k}</span><span class="v">${v}</span>`;
    info.appendChild(row);
  };
  add('kind', kind);
  add('id', String(id));
  add('n particles', String(parts.length));
  add('particle IDs', parts.join(', '));
  add('Σ PE (inst)', formatPE(pe));
  // Sensors covered and track count from per_track.
  const covered = new Set();
  for (const p of parts) {
    const m = particleToSensor[p];
    if (!m) continue;
    for (const s of m.keys()) covered.add(s);
  }
  add('sensors hit', String(covered.size));
  let nTracks = 0;
  const pt = evtBundle.labl.per_track;
  if (pt && pt.particle_idx) {
    const ps = new Set(parts);
    for (let t = 0; t < pt.particle_idx.length; t++) if (ps.has(pt.particle_idx[t])) nTracks++;
  }
  add('n tracks', String(nTracks));
}

function pdgName(code) {
  return PDG_NAMES.get(code) || String(code);
}

function particleSwatch(p) {
  // Swatch color matches the 3D shader. Category mode uses categoryHue;
  // otherwise each particle gets a hashed-golden hue.
  let h;
  if (curLabel === 'category') {
    const cat = evtBundle.labl.per_particle.category;
    h = categoryHue(cat ? cat[p] : p);
  } else {
    h = hashHue(p);
  }
  const [r,g,b] = hsl2rgb(h, 0.78, 0.55);
  return `rgb(${r},${g},${b})`;
}

const PDG_NAMES = new Map([
  [11,'e⁻'],[-11,'e⁺'],[13,'μ⁻'],[-13,'μ⁺'],[22,'γ'],
  [211,'π⁺'],[-211,'π⁻'],[111,'π⁰'],[2212,'p'],[2112,'n'],
]);
const CAT_NAMES = ['Primary','DecayElec','SecPion','Gamma','—','—','—','—'];

function particleLabel(p) {
  const cat = evtBundle.labl.per_particle.category;
  if (cat) return `${CAT_NAMES[cat[p]] || 'cat'+cat[p]} · P${p}`;
  return `P${p}`;
}

function formatPE(pe) {
  if (pe >= 1000) return (pe/1000).toFixed(1)+'k';
  if (pe >= 10) return pe.toFixed(0);
  return pe.toFixed(1);
}

// ── 2D canvas ──────────────────────────────────────────────────────────
function init2D() {
  c2d = $('canvas2d');
  ctx2d = c2d.getContext('2d');
  resize2D();
  window.addEventListener('resize', () => { onResize(); });
}

function resize2D() {
  const el = $('panel2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  c2d.width = el.clientWidth * dpr;
  c2d.height = el.clientHeight * dpr;
  c2d.style.width = el.clientWidth + 'px';
  c2d.style.height = el.clientHeight + 'px';
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function render2D() {
  if (!layout || !ctx2d) return;
  const el = $('panel2d');
  const W = el.clientWidth, H = el.clientHeight;
  ctx2d.clearRect(0, 0, W, H);
  ctx2d.fillStyle = '#080808';
  ctx2d.fillRect(0, 0, W, H);

  // Fit layout into canvas (preserve aspect ratio, add margin).
  const margin = 30;
  const avW = W - 2 * margin, avH = H - 2 * margin;
  const scale = Math.min(avW / layout.layoutW, avH / layout.layoutH);
  const offX = margin + (avW - layout.layoutW * scale) / 2;
  const offY = margin + (avH - layout.layoutH * scale) / 2;

  // Panel backgrounds (very faint) and labels.
  ctx2d.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx2d.lineWidth = 1;
  ctx2d.font = '10px monospace';
  ctx2d.fillStyle = '#444';
  ctx2d.textAlign = 'left';
  for (const p of layout.panels) {
    const rx = offX + p.rect.x * scale;
    const ry = offY + p.rect.y * scale;
    const rw = p.rect.w * scale, rh = p.rect.h * scale;
    ctx2d.strokeRect(rx, ry, rw, rh);
    ctx2d.fillText(p.label, rx + 4, ry + 11);
  }

  // Seams (box face boundaries).
  if (layout.seams && layout.seams.length) {
    ctx2d.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx2d.setLineDash([4, 3]);
    for (const s of layout.seams) {
      const panel = layout.panels[s.panel];
      const rx = offX + panel.rect.x * scale;
      const ry = offY + panel.rect.y * scale;
      const sx = rx + s.u * scale;
      ctx2d.beginPath();
      ctx2d.moveTo(sx, ry);
      ctx2d.lineTo(sx, ry + panel.rect.h * scale);
      ctx2d.stroke();
    }
    ctx2d.setLineDash([]);
  }

  // Precompute per-sensor colors for this frame.
  const isCat = (curLabel !== 'none');
  const contArr = isCat ? null : pmtContValArrayForField().norm;
  const catArr = isCat ? pmtCatValArray() : null;

  // Selected item (particle or category) influences CPU-side alpha for 2D.
  const particleSet2d = currentParticleSet();
  const corrActive = particleSet2d != null && particleSet2d.length > 0;
  const corrMap = corrActive ? unionContributions(particleSet2d) : null;
  let corrMax = 1;
  if (corrMap) for (const v of corrMap.values()) if (v > corrMax) corrMax = v;

  // Selection hue (categorical mode only): in pile-up / overlap, a sensor can
  // receive PE from the selection AND from another group whose particle
  // dominates the sensor. Painting such a contributor in the dominant
  // particle's hue makes it look like another group's hit is being
  // highlighted. When a selection is active, recolor contributors with the
  // selection's own hue.
  let selectionHue = 0;
  if (corrActive && isCat) {
    if (selectedGroup) {
      selectionHue = groupHue(selectedGroup.kind, selectedGroup.id);
    } else if (selectedParticle != null) {
      selectionHue = hashHue(selectedParticle);
    }
  }

  // 2D PMT fade: pmtArrivalT lives in PMT-time space (rank or raw), but
  // the active simTime might be in seg-time space (SEG view). Map sweep
  // progress (0..1 through the active range) onto PMT range so the 2D
  // panel sweeps in lockstep with the 3D animation, regardless of view.
  const sweepActive = sweepOn && pmtArrivalT;
  const progress = (simTime - simTMin) / Math.max(1e-9, simTMax - simTMin);
  const pmtSimTime = pmtTRange[0] + progress * (pmtTRange[1] - pmtTRange[0]);
  const sweepEps = Math.max(1e-4, (pmtTRange[1] - pmtTRange[0]) / 200);

  // Draw each PMT in layout space. v is already canvas-y-down.
  for (let i = 0; i < nSensors; i++) {
    const p = layout.panel[i];
    const px = offX + layout.u[i] * scale;
    const py = offY + layout.v[i] * scale;
    const d = layout.pmtPitch[p] * scale;
    const r = Math.max(1.2, d * 0.5);

    const hasSig = pmtHasSignal[i];
    let fill;
    if (!hasSig) {
      if (!showEmpty) continue;
      fill = '#333';
    } else if (isCat) {
      // catArr[i] is a hue in [0,1] (categoryHue or hashHue). Render via HSL
      // so 2D agrees with the 3D shader. When a selection is active, paint
      // contributors with the selection's hue instead of the dominant
      // particle's hue — otherwise overlap PMTs read as "wrong group".
      const isSelContrib = corrActive && (corrMap.get(i) || 0) > 0;
      const hue = isSelContrib ? selectionHue : catArr[i];
      const [cr,cg,cb] = hsl2rgb(hue, 0.78, 0.55);
      fill = `rgb(${cr},${cg},${cb})`;
    } else {
      const [cr,cg,cb] = currentCmapRGB(contArr[i]);
      fill = `rgb(${cr},${cg},${cb})`;
    }

    let alpha = hasSig ? 1.0 : 0.35;
    if (corrActive) {
      const contrib = corrMap.get(i) || 0;
      if (contrib > 0) {
        // Contributors: always clearly visible; sqrt curve spreads the
        // brightness across small/medium/large contributions.
        alpha = 0.35 + 0.65 * Math.sqrt(contrib / corrMax);
      } else {
        alpha = hasSig ? 0.15 : 0.08;   // non-contributors dim, but not gone
      }
    }
    // Sweep fade — match the 3D shader's smoothstep (cubic Hermite),
    // using PMT-time-space pmtSimTime so it works in any 3D view.
    if (sweepActive && hasSig) {
      const aT = pmtArrivalT[i];
      if (Number.isFinite(aT) && aT < 1e29) {
        const u = Math.max(0, Math.min(1, (pmtSimTime - (aT - sweepEps)) / (2 * sweepEps)));
        const fade = u * u * (3 - 2 * u);
        alpha *= fade;
      } else {
        alpha = 0;   // sentinel / unreachable
      }
    }
    ctx2d.globalAlpha = alpha;
    ctx2d.fillStyle = fill;
    ctx2d.beginPath();
    ctx2d.arc(px, py, r, 0, Math.PI * 2);
    ctx2d.fill();
  }
  ctx2d.globalAlpha = 1.0;

  drawLegend();
}

function drawLegend() {
  // A small colorbar at the bottom of the 2D panel for continuous mode.
  if (!ctx2d) return;
  const el = $('panel2d');
  const W = el.clientWidth, H = el.clientHeight;
  const isCat = (curLabel !== 'none');
  if (isCat) {
    $('label2dText').textContent = `UNWRAPPED · ${labelText()}`;
    return;
  }
  const bw = 160, bh = 8;
  const bx = W - bw - 16, by = H - 18;
  for (let i = 0; i < bw; i++) {
    const [r,g,b] = currentCmapRGB(i / (bw - 1));
    ctx2d.fillStyle = `rgb(${r},${g},${b})`;
    ctx2d.fillRect(bx + i, by, 1, bh);
  }
  ctx2d.strokeStyle = '#333';
  ctx2d.strokeRect(bx, by, bw, bh);
  ctx2d.fillStyle = '#888';
  ctx2d.font = '10px monospace';
  ctx2d.textAlign = 'left';
  ctx2d.fillText(labelText() + (logScale ? ' (log)' : ''), bx, by - 3);
  $('label2dText').textContent = `UNWRAPPED · ${labelText()}`;
}

function labelText() {
  if (curLabel === 'particle') return 'Particle';
  if (curLabel === 'category') return 'Category';
  if (curLabel === 'track') return 'Track';
  if (curLabel === 'pdg') return 'PDG';
  if (curLabel === 'beta') return 'β';
  if (curLabel === 'ncher') return 'n_Cherenkov';
  return curField === 'charge' ? 'PE' : 'T (ns)';
}

// ── Three.js scene setup ───────────────────────────────────────────────
function initThree() {
  const el = $('panel3d');
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(el.clientWidth, el.clientHeight);
  renderer.setClearColor(0x080808);
  el.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, 0.1, 10000);
  // LUCiD convention: +z is up. Setting this makes OrbitControls orbit
  // around the world z-axis so the top stays at the top during rotation.
  camera.up.set(0, 0, 1);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.autoRotate = autoRotate;
  controls.autoRotateSpeed = 0.15;

  texPlasma = makeCmapTex(PLASMA_STOPS);
  texViridis = makeCmapTex(VIRIDIS_STOPS);
  texViridisR = makeCmapTex(VIRIDIS_R_STOPS);
  texInfernoR = makeCmapTex(INFERNO_R_STOPS);
}

// ── Animate loop ───────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = lastFrameTime ? now - lastFrameTime : 16;
  lastFrameTime = now;

  if (sweepOn && sweepPlaying) {
    const range = Math.max(1e-6, simTMax - simTMin);
    const stepNs = (range / 4000) * sweepSpeed * (dt);  // 4s for full sweep at 1x
    simTime += stepNs;
    if (simTime > simTMax + range * 0.1) simTime = simTMin;
    updateSweepUI();
    if (pmtMat) pmtMat.uniforms.simTime.value = simTime;
    if (segMat) segMat.uniforms.simTime.value = simTime;
    render2D();
  }

  controls.update();
  renderer.render(scene, camera);
}

function applyViewSweepRange() {
  const range = curView === 'seg' ? segTRange : pmtTRange;
  simTMin = range[0]; simTMax = range[1];
  if (!(Number.isFinite(simTMin) && Number.isFinite(simTMax) && simTMax > simTMin)) {
    simTMin = 0; simTMax = 1;
  }
  const scrub = $('sweepScrubber');
  scrub.min = simTMin; scrub.max = simTMax;
  scrub.step = (simTMax - simTMin) / 1000 || 0.01;
  // Always reset simTime when the range changes — applyViewSweepRange is
  // only called when the time *space* changed (event load, view toggle,
  // scope change, reset), so an in-range carry-over (e.g. simTime=0.5 was
  // a quantile rank, now it's 0.5 ns in a different mesh) is meaningless.
  simTime = simTMin;
  scrub.value = simTime;
  const eps = Math.max(1e-4, (simTMax - simTMin) / 200);
  // Re-sync every sweep-related uniform — mesh rebuilds (quantile scope,
  // event load, etc.) create fresh materials with default-0 uniforms.
  if (pmtMat) {
    pmtMat.uniforms.sweepEps.value = eps;
    pmtMat.uniforms.simTime.value = simTime;
    pmtMat.uniforms.sweepOn.value = sweepOn ? 1.0 : 0.0;
  }
  if (segMat) {
    segMat.uniforms.sweepEps.value = eps;
    segMat.uniforms.simTime.value = simTime;
    segMat.uniforms.sweepOn.value = sweepOn ? 1.0 : 0.0;
  }
}

function updateSweepUI() {
  $('sweepScrubber').value = simTime;
  // Label units depend on whether the active mesh's range is rank ([0,1])
  // or raw ns (anything wider).
  const isRank = (simTMax - simTMin) <= 1.5 && simTMin >= -0.001 && simTMax <= 1.001;
  $('sweepTimeLabel').textContent = isRank
    ? (simTime * 100).toFixed(1) + ' %'
    : simTime.toFixed(2) + ' ns';
  const btn = $('sweepPlayPause');
  btn.innerHTML = sweepPlaying ? '&#x23F8;' : '&#x25B6;';
  btn.classList.toggle('active', sweepPlaying);
}

// ── Event load flow ────────────────────────────────────────────────────
async function loadEvent(idx) {
  showOverlay(`Loading event ${idx}...`);
  try {
    const d = await workerCall('loadEvent', { idx });
    evtBundle = d;
    curEvent = idx;
    $('evInput').value = idx;
    if (d.warning) console.warn(d.warning);
    // Transient per-event state. A selection on the previous event is
    // meaningless; sweep resets to a stopped state at the start.
    selectedParticle = null;
    selectedGroup = null;
    sweepPlaying = false;
    simTime = 0;
    deriveSensorArrays();
    buildInstLookups();
    refreshUnionQMap();   // needs pmtT + seg times; before buildPMTs/buildSegments
    buildPMTs();
    buildSegments();
    buildOutline();
    frameCamera();
    buildSidebar();
    applyCorrespondence();
    // Sweep range + shader fade width tied to current view's T spread.
    applyViewSweepRange();
    updateSweepUI();
    render2D();
    setStatus(`event ${idx} · src ${d.srcIdx ?? '?'}`);
  } catch (e) {
    setStatus('error: ' + e.message);
    console.error(e);
  } finally {
    hideOverlay();
  }
}

// ── UI wiring ──────────────────────────────────────────────────────────
function syncSweepBtn() {
  $('sweepBtn').classList.toggle('active', sweepOn);
  $('sweepBar').classList.toggle('visible', sweepOn);
}
function onResize() {
  const el3 = $('panel3d');
  if (renderer) {
    renderer.setSize(el3.clientWidth, el3.clientHeight);
    camera.aspect = el3.clientWidth / el3.clientHeight;
    camera.updateProjectionMatrix();
  }
  resize2D();
  render2D();
}

function setupUI() {
  // Event nav.
  $('evPrev').addEventListener('click', () => { if (curEvent > 0) loadEvent(curEvent - 1); });
  $('evNext').addEventListener('click', () => { if (curEvent < nEvents - 1) loadEvent(curEvent + 1); });
  const commitEvInput = () => {
    const n = Math.max(0, Math.min(nEvents - 1, parseInt($('evInput').value) || 0));
    if (n !== curEvent) loadEvent(n);
    else $('evInput').value = curEvent;
  };
  $('evInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') commitEvInput(); });
  $('evInput').addEventListener('change', commitEvInput);

  // VIEW toggle (PMTs / SEG).
  $('viewGrp').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    curView = b.dataset.v;
    for (const c of $('viewGrp').children) c.classList.toggle('active', c === b);
    if (pmtMesh) pmtMesh.visible = (curView === 'pmts');
    if (segMesh) segMesh.visible = (curView === 'seg');
    applyViewSweepRange();
    updateSweepUI();
  });

  // Field toggle (Charge/Time).
  $('fieldGrp').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    curField = b.dataset.v;
    for (const c of $('fieldGrp').children) c.classList.toggle('active', c === b);
    updatePMTColors();
    updateSegmentColors();
    render2D();
  });

  // Label dropdown.
  $('labelSelect').addEventListener('change', (e) => {
    curLabel = e.target.value;
    // Any existing selection was for a different grouping — clear it so the
    // sidebar + correspondence start fresh in the new label space.
    selectedParticle = null;
    selectedGroup = null;
    updatePMTColors();
    updateSegmentColors();
    buildSidebar();
    applyCorrespondence();
    render2D();
  });

  // Correspondence toggle.
  // Time sweep toggle.
  $('sweepBtn').addEventListener('click', () => {
    sweepOn = !sweepOn;
    syncSweepBtn();
    if (pmtMat) pmtMat.uniforms.sweepOn.value = sweepOn ? 1.0 : 0.0;
    if (segMat) segMat.uniforms.sweepOn.value = sweepOn ? 1.0 : 0.0;
    if (sweepOn) sweepPlaying = true;
    updateSweepUI();
    render2D();
  });
  $('sweepPlayPause').addEventListener('click', () => {
    if (!sweepOn) return;   // no-op when sweep UI is hidden
    sweepPlaying = !sweepPlaying;
    updateSweepUI();
  });
  $('sweepScrubber').addEventListener('input', (e) => {
    simTime = parseFloat(e.target.value);
    sweepPlaying = false;
    updateSweepUI();
    if (pmtMat) pmtMat.uniforms.simTime.value = simTime;
    if (segMat) segMat.uniforms.simTime.value = simTime;
    render2D();
  });

  // Reset — everything visible to the user back to defaults.
  $('resetBtn').addEventListener('click', () => {
    // Toolbar state.
    curView = 'pmts'; curField = 'charge'; curLabel = 'none';
    selectedParticle = null; selectedGroup = null;
    sweepOn = false; sweepPlaying = false; quantileScope = 'pmts';
    simTime = 0;
    // Settings-drawer state.
    logScale = true; percMin = 1; percMax = 99;
    manualVmin = null; manualVmax = null;
    cmapName = 'auto';
    pmtSize = 10;
    showEmpty = true; showMesh = true;
    sweepSpeed = 1.0;
    autoRotate = true;
    // Sync every DOM control.
    $('labelSelect').value = 'none';
    $('logChk').checked = true;
    $('percMin').value = 1; $('percMax').value = 99;
    $('percVal').textContent = '1 – 99';
    $('vminInput').value = ''; $('vmaxInput').value = '';
    $('cmapSelect').value = 'auto';
    $('quantileScope').value = quantileScope;
    $('pmtSizeSlider').value = pmtSize; $('pmtSizeVal').textContent = pmtSize.toFixed(1);
    $('showEmptyChk').checked = showEmpty;
    $('showMeshChk').checked = showMesh;
    $('sweepSpeed').value = sweepSpeed; $('sweepSpeedVal').textContent = sweepSpeed.toFixed(1);
    for (const c of $('viewGrp').children) c.classList.toggle('active', c.dataset.v === 'pmts');
    for (const c of $('fieldGrp').children) c.classList.toggle('active', c.dataset.v === 'charge');
    $('rotBtn').classList.toggle('active', autoRotate);
    if (controls) controls.autoRotate = autoRotate;
    syncSweepBtn();
    // Rebuild meshes (quantile may have been on, arrivalT baked into geometry).
    if (evtBundle) { refreshUnionQMap(); buildPMTs(); buildSegments(); buildOutline(); }
    if (pmtMesh) pmtMesh.visible = true;
    if (segMesh) segMesh.visible = false;
    if (pmtMat) pmtMat.uniforms.sweepOn.value = 0.0;
    if (segMat) segMat.uniforms.sweepOn.value = 0.0;
    applyViewSweepRange();
    updateSweepUI();
    updatePMTColors();
    updateSegmentColors();
    applyCorrespondence();
    buildSidebar();
    render2D();
  });

  $('rotBtn').addEventListener('click', () => {
    autoRotate = !autoRotate;
    controls.autoRotate = autoRotate;
    $('rotBtn').classList.toggle('active', autoRotate);
  });

  // Settings drawer.
  $('settingsBtn').addEventListener('click', () => $('settingsPanel').classList.toggle('visible'));
  $('settingsClose').addEventListener('click', () => hide('settingsPanel'));

  $('logChk').addEventListener('change', (e) => { logScale = e.target.checked; updatePMTColors(); updateSegmentColors(); render2D(); });
  const updatePerc = () => {
    percMin = parseFloat($('percMin').value);
    percMax = parseFloat($('percMax').value);
    if (percMin >= percMax - 1) { percMin = Math.max(0, percMax - 1); $('percMin').value = percMin; }
    $('percVal').textContent = percMin.toFixed(1) + ' – ' + percMax.toFixed(1);
    updatePMTColors(); updateSegmentColors(); render2D();
  };
  $('percMin').addEventListener('input', updatePerc);
  $('percMax').addEventListener('input', updatePerc);
  $('vminInput').addEventListener('change', (e) => { manualVmin = e.target.value.trim() === '' ? null : parseFloat(e.target.value); updatePMTColors(); updateSegmentColors(); render2D(); });
  $('vmaxInput').addEventListener('change', (e) => { manualVmax = e.target.value.trim() === '' ? null : parseFloat(e.target.value); updatePMTColors(); updateSegmentColors(); render2D(); });
  $('cmapSelect').addEventListener('change', (e) => { cmapName = e.target.value; updatePMTColors(); updateSegmentColors(); render2D(); });

  $('pmtSizeSlider').addEventListener('input', (e) => {
    pmtSize = parseFloat(e.target.value);
    $('pmtSizeVal').textContent = pmtSize.toFixed(1);
    if (pmtMat) pmtMat.uniforms.pmtSize.value = pmtSize;
    if (segMat) segMat.uniforms.pmtSize.value = Math.max(3, pmtSize * 0.6);
  });
  $('showEmptyChk').addEventListener('change', (e) => {
    showEmpty = e.target.checked;
    if (pmtMat) pmtMat.uniforms.emptyGray.value = showEmpty ? 1.0 : 0.0;
    render2D();
  });
  $('showMeshChk').addEventListener('change', (e) => {
    showMesh = e.target.checked;
    buildOutline();
  });
  $('sweepSpeed').addEventListener('input', (e) => {
    sweepSpeed = parseFloat(e.target.value);
    $('sweepSpeedVal').textContent = sweepSpeed.toFixed(1);
  });
  $('quantileScope').addEventListener('change', (e) => {
    quantileScope = e.target.value;
    if (evtBundle) {
      // Rebuild meshes (arrivalT is baked in), then refresh colors and
      // correspondence so the effect is visible even with Sweep off.
      refreshUnionQMap();
      buildPMTs();
      buildSegments();
      updatePMTColors();
      updateSegmentColors();
      applyCorrespondence();
      applyViewSweepRange();
      updateSweepUI();
      render2D();
    }
    const label = {off: 'off', pmts: 'PMTs only', seg: 'segments only', both: 'PMTs + segments'}[quantileScope];
    showToast(`quantile T → ${label}`);
  });

  // Save PNG.
  $('save3d').addEventListener('click', async () => {
    const blob = await new Promise(r => renderer.domElement.toBlob(r, 'image/png'));
    downloadBlob(blob, `lucid_event${String(curEvent).padStart(3,'0')}_3d.png`);
  });
  $('save2d').addEventListener('click', async () => {
    const blob = await new Promise(r => c2d.toBlob(r, 'image/png'));
    downloadBlob(blob, `lucid_event${String(curEvent).padStart(3,'0')}_2d.png`);
  });

  window.addEventListener('resize', onResize);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ── Init ───────────────────────────────────────────────────────────────
window.addEventListener('error', (e) => {
  console.error('[viewer] window error:', e.message, e.filename, e.lineno);
  const ls = document.getElementById('loadStatus');
  if (ls) ls.textContent = 'JS error: ' + e.message + ' (see console)';
});
window.addEventListener('unhandledrejection', (e) => {
  console.error('[viewer] unhandled rejection:', e.reason);
  const ls = document.getElementById('loadStatus');
  if (ls) ls.textContent = 'async error: ' + (e.reason && e.reason.message || e.reason);
});

console.log('[viewer] module loaded');
if (typeof THREE === 'undefined') {
  console.error('[viewer] THREE global is undefined — three.js CDN fetch likely blocked');
}

(async function init() {
  const ls = $('loadStatus');
  console.log('[viewer] init() starting');
  try {
    ls.textContent = 'Spawning worker...';
    console.log('[viewer] spawning worker');
    createWorker();
    worker.addEventListener('message', (e) => {
      if (e.data.action === 'error') {
        console.error('[worker error]', e.data.message, e.data.stack);
        setStatus('worker error: ' + e.data.message);
      }
    });

    const params = new URLSearchParams(window.location.search);
    const base = params.get('base') || '';

    ls.textContent = 'Fetching manifest...';
    console.log('[viewer] fetching manifest');
    const mr = await fetch(base + '/manifest.json');
    if (!mr.ok) throw new Error('Cannot fetch manifest.json');
    const manifest = await mr.json();
    console.log('[viewer] manifest:', manifest);

    ls.textContent = 'Mounting HDF5 files...';
    console.log('[viewer] worker init call');
    const cfg = await workerCall('init', { base, manifest });
    console.log('[viewer] worker ready:', cfg);
    nEvents = cfg.nEvents;
    nSensors = cfg.nSensors;
    detectorType = cfg.detectorType;
    shape = cfg.shape;
    sensorPositions = cfg.sensorPositions;

    ls.textContent = 'Computing layout...';
    layout = computeLayout(detectorType, sensorPositions, nSensors, shape);

    $('loading').style.display = 'none';
    $('app').style.display = 'flex';
    $('evInput').max = nEvents - 1;
    $('evTotal').textContent = '/ ' + (nEvents - 1);

    initThree();
    init2D();
    setupUI();

    await loadEvent(0);
    animate();
  } catch (e) {
    ls.textContent = 'Error: ' + e.message;
    console.error(e);
  }
})();
