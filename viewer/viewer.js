// LUCiD Event Viewer — main module.
//
// Architecture modeled on the JAXTPC viewer: Three.js WebGL 3D panel +
// Canvas2D unwrapped 2D panel, with h5wasm streaming in a worker. See
// shaders.js (rendering), geometry_layout.js (unwrap math),
// colormaps.js (palettes), h5_worker.js (I/O).

import { PMT_VS, PMT_FS } from './shaders.js';
import {
  PLASMA_STOPS, VIRIDIS_STOPS, VIRIDIS_R_STOPS, INFERNO_R_STOPS, RDBU_STOPS,
  hashHue, plasmaRGB, viridisRGB, viridisRRGB, rdBuRGB, hsl2rgb,
} from './colormaps.js';
import { computeLayout } from './geometry_layout.js';

// ── Globals ─────────────────────────────────────────────────────────────
let worker = null;
let nEvents = 0, nSensors = 0;
let detectorType = '', shape = {};
let detectorMaterial = 'water';    // string from edep/config.material — drives β_thresh

// Cherenkov β threshold = 1/n. Materials table mirrors lucid/utils.py.
// Used by the BETA field to remap β ∈ [β_thresh, 1] → [0, 1] so the
// colormap covers the physically-meaningful Cherenkov-emitting range.
const REFRACTIVE_INDEX_BY_MATERIAL = {
  water: 1.33,
  ice:   1.31,
};
function cherenkovBetaThreshold(material) {
  const n = REFRACTIVE_INDEX_BY_MATERIAL[(material || '').toLowerCase()] || 1.33;
  return 1.0 / n;
}
let sensorPositions = null;        // Float32Array(nSensors * 3)
let layout = null;                 // from computeLayout()

let curEvent = 0;
let evtBundle = null;              // decoded {sensor, hits, edep, labl, t0, srcIdx, ...}

// Per-PMT derived arrays (length nSensors). pmtPE / pmtT / pmtHasSignal are
// the *active* views — applyEmissionFilter swaps them between the sensor.h5
// derived combined arrays (filter == 'all') and the hits.h5 derived per-
// process arrays (filter == 'cher' / 'scint'). Everything downstream reads
// these three and is filter-agnostic.
let pmtPE = null;                  // summed PE per sensor (active set)
let pmtT = null;                   // earliest T per sensor (NaN if no hit) (active set)
let pmtHasSignal = null;           // Uint8Array (active set)
let pmtBeta = null;                // PE-weighted mean β over contributing segments per sensor (NaN if no contribution)
let pmtDomParticle = null;         // Int32Array  (argmax-over-hits per sensor; -1 if none)
let pmtArrivalT = null;            // Float32Array — same value used for the 3D sweep shader
// Backing arrays per emission slice. Built once per event load by
// derivePMTArrays; applyEmissionFilter selects which set the active
// references point at.
//   pmtPE_all/T_all/HasSignal_all  — from sensor.h5 (combined, smeared,
//     includes orphan-track photons that hits.h5 drops).
//   pmtPE_cher/T_cher/HasSignal_cher — sum of hits.h5 rows with
//     emission_process==0 (Cherenkov), per sensor. Pre-smearing, drops
//     orphan-track photons. Same definition for the scint variant.
let pmtPE_all = null,  pmtT_all = null,  pmtHasSignal_all = null;
let pmtPE_cher = null, pmtT_cher = null, pmtHasSignal_cher = null;
let pmtPE_scint = null, pmtT_scint = null, pmtHasSignal_scint = null;
//   pmtPE_dark/... — sum of hits.h5 rows with emission_process==2 (electronic
//     dark noise, particle_idx==-1). Only populated when dark rows exist.
let pmtPE_dark = null, pmtT_dark = null, pmtHasSignal_dark = null;
// Sentinel dominant-"particle" value for a dark-noise-dominated sensor under
// the LABEL=Particle coloring while EMISSION=Dark (dark has no owning particle).
const DARK_LABEL = -2;
// Reserved categorical hue for the dark-noise category (cyan; distinct from the
// low-index particle hues 0/0.618/0.236/0.854). catVal is a hue in [0,1).
const DARK_HUE = 0.5;
// Per-sensor Cherenkov fraction f = PE_cher / (PE_cher + PE_scint).
// NaN where both are zero. Drives the CHER FRAC continuous field.
let pmtCherFraction = null;

// hits lookups.
let particleToSensor = null;       // Array(nParticles) of Map<sensor, PE>
let particleTotals = null;         // Float32Array(nParticles)  total PE per particle
let particleInteraction = null;    // Int32Array(nParticles)
let particlePdgBucket = null;      // Int8Array(nParticles)   own-PDG bucket, with π⁰ ancestor override
// PDG-mode sidebar rows. One row per particle (mirroring Particle mode),
// just sorted by PDG bucket and colored by the fixed bucket palette.
// Particles with zero PE contribution are excluded entirely.
//   pdgRows[i] = { id, bucket, particleIds: [pid], totalPE }
// (particleIds is always a singleton, kept as a list for code-path
// uniformity with the older shower-grouping shape.)
let pdgRows = null;

// edep/sensor_hits lookups (only populated when LUCiD ran with
// store_segment_sensor_map=true; otherwise all stay null and Label=Segment
// shows nothing).
let pmtDomSegment = null;          // Int32Array(nSensors)  argmax-PE segment per sensor (-1 if none)
let segmentToSensor = null;        // Array(edep.n) of Map<sensor, PE>
let segmentTotals = null;          // Float32Array(edep.n)  total PE per segment

// UI state.
let curView = 'pmts';              // 'pmts' | 'edep'   (exclusive 3D view)
let curField = 'charge';           // 'charge' | 'time' | 'beta' | 'cher_frac'  (continuous field)
let curLabel = 'none';             // 'none' | 'particle' | 'pdg' | 'interaction' | 'segment'
// Per-sensor signal restricted to one emission process (or 'all' for the
// kernel-accumulator combined signal). When the dataset has only a single
// emission process — pure Cherenkov or pure scintillation — the dropdown +
// CHER FRAC field stay hidden and the filter is locked to 'all'.
let emissionFilter = 'all';        // 'all' | 'cher' | 'scint' | 'dark'
let datasetHasBothProcesses = false;
let datasetHasDark = false;         // any emission_process==2 (dark-noise) rows seen
// HIT slice: which recorded hit (digit) to show per sensor. 0 = 'all' (sum
// charge / first-arrival time — the default, == legacy one-hit-per-sensor).
// k>=1 selects the k-th digit (by arrival time) on each sensor; PMTs with
// fewer than k digits go dark in that slice. maxHits = largest per-sensor
// digit count in the current event (drives the dropdown range). Both FIELD
// and LABEL re-aggregate against the active hit slice (via digit_idx).
let hitFilter = 0;
let maxHits = 1;
let sensorDigits = null;    // Array(nSensors): sensor.h5 row indices per sensor, arrival-ordered
let selectedDigit = null;   // Int32Array(nSensors): chosen digit row (== hits.digit_idx) or -1

// TRIGGER WINDOW slice (triggered datasets only). 'all' shows every digit; an
// integer w restricts to readout window w's digits (a contiguous sensor.h5 row
// range from the per_window CSR digit_offsets), and the HIT slice then indexes
// within that window. Outer slice: composes with EMISSION and HIT.
let windowFilter = 'all';
let windows = null;         // { window_start, window_end, digit_offsets } or null

// [lo, hi) sensor.h5 digit rows for the active window ([0, nHits) when 'all' or
// the dataset carries no per_window). digit_idx == the sensor.h5 row index, so
// hits.h5 rows are in-window iff lo <= digit_idx < hi.
function windowRange() {
  const s = evtBundle && evtBundle.sensor;
  const n = s ? s.nHits : 0;
  if (windowFilter === 'all' || !windows || !windows.digit_offsets) return [0, n];
  const off = windows.digit_offsets, w = windowFilter;
  if (w < 0 || w + 1 >= off.length) return [0, n];
  return [off[w], off[w + 1]];
}
let logScale = true;
let percMin = 1, percMax = 99;
let manualVmin = null, manualVmax = null;
let cmapName = 'auto';
// Selection state. Only one is ever set at a time.
//   selectedParticle : specific particle index (used when Label = None/Particle)
//   selectedGroup    : { kind: 'pdg'|'interaction', id: int }
let selectedParticle = null;
let selectedGroup = null;
let showEmpty = true;
let showMesh = true;
let pmtSize = 10;
let outlineWidth = 1.0;
let autoRotate = true;

// Time sweep.
let sweepOn = false;
let sweepPlaying = false;
let simTime = 0;
let simTMin = 0, simTMax = 0;              // currently-active-view range
let pmtTRange = [0, 1], edepTRange = [0, 1];
let sweepSpeed = 1.0;
// Quantile-T scope: 'off' | 'pmts' | 'edep' | 'both'. Default = PMTs only
// (the event display — where it's most visually useful); segments stay in
// raw ns so the user can see the physical time arithmetic.
let quantileScope = 'pmts';
const quantilePMT = () => quantileScope === 'pmts' || quantileScope === 'both';
const quantileEdep = () => quantileScope === 'edep'  || quantileScope === 'both';
// In 'both' mode, PMT and segment times share one quantile map so the same
// physical time maps to the same rank on both meshes. null otherwise.
let unionQMap = null;

// Three.js.
let renderer, scene, camera, controls;
let pmtGeo, pmtMat, pmtMesh;
let edepGeo, edepMat, edepMesh;
let outlineMesh = null;
let outlineMat = null;
let lastFrameTime = 0;

// Colormap textures.
let texPlasma, texViridis, texViridisR, texInfernoR, texRdBu;

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
  if (curField === 'beta') return 'viridis';
  if (curField === 'cher_frac') return 'rdbu';
  return curField === 'charge' ? 'plasma' : 'viridis_r';
}

function currentCmapRGB(t) {
  const name = currentCmapName();
  if (name === 'plasma') return plasmaRGB(t);
  if (name === 'viridis') return viridisRGB(t);
  if (name === 'viridis_r') return viridisRRGB(t);
  if (name === 'rdbu') return rdBuRGB(t);
  return viridisRGB(t);
}

function currentCmapTex() {
  const name = currentCmapName();
  if (name === 'plasma') return texPlasma;
  if (name === 'viridis') return texViridis;
  if (name === 'viridis_r') return texViridisR;
  if (name === 'inferno_r') return texInfernoR;
  if (name === 'rdbu') return texRdBu;
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

// ── PMT derivation from sensor / hits ──────────────────────────────────
// v4 writes only signal-bearing sensors into the sparse sensor file (the
// save_sensor_event_v3 mask keeps rows where PE > 0 OR T is a finite
// positive time within a reasonable window). Whatever makes it through,
// we still filter here with PE > 0 to guard against edge cases.
// Group sensor.h5 digit rows by sensor, arrival-time ordered. sensorDigits[s]
// is the list of sensor.h5 row indices for PMT s (a row index IS that digit's
// hits.digit_idx). Sets maxHits = largest per-sensor digit count.
function buildSensorDigits() {
  sensorDigits = new Array(nSensors);
  maxHits = 1;
  const s = evtBundle && evtBundle.sensor;
  if (!s || !s.nHits) return;
  const [lo, hi] = windowRange();       // restrict to the active trigger window
  const buckets = new Array(nSensors);
  for (let i = lo; i < hi; i++) {
    const si = s.sensor_idx[i];
    (buckets[si] || (buckets[si] = [])).push(i);
  }
  const T = s.T;
  for (let si = 0; si < nSensors; si++) {
    const rows = buckets[si];
    if (!rows) continue;
    rows.sort((a, b) => T[a] - T[b]);   // arrival-time order
    sensorDigits[si] = rows;
    if (rows.length > maxHits) maxHits = rows.length;
  }
  // Navigating to an event with fewer digits invalidates a deeper HIT
  // selection — fall back to 'All' so the display never silently blanks.
  if (hitFilter > maxHits) hitFilter = 0;
}

// For the active hitFilter, record each sensor's chosen digit row (== its
// hits.digit_idx) or -1 when that sensor has fewer than `hitFilter` digits.
function computeSelectedDigit() {
  selectedDigit = new Int32Array(nSensors).fill(-1);
  if (hitFilter <= 0 || !sensorDigits) return;
  const k = hitFilter - 1;
  for (let si = 0; si < nSensors; si++) {
    const rows = sensorDigits[si];
    if (rows && k < rows.length) selectedDigit[si] = rows[k];
  }
}

function deriveSensorArrays() {
  buildSensorDigits();
  computeSelectedDigit();
  const [wLo, wHi] = windowRange();   // active trigger-window digit-row range

  // 'All' slice: from sensor.h5. hitFilter==0 sums every digit per sensor
  // (charge) at the first-arrival time (== legacy). hitFilter==k shows only
  // the selected digit's charge/time on each PMT that has >=k digits.
  pmtPE_all = new Float32Array(nSensors);
  pmtT_all = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) pmtT_all[i] = NaN;
  pmtHasSignal_all = new Uint8Array(nSensors);
  const s = evtBundle.sensor;
  if (s && s.nHits) {
    if (hitFilter === 0) {
      for (let i = wLo; i < wHi; i++) {          // window-restricted
        const si = s.sensor_idx[i];
        const pe = s.PE[i];
        pmtPE_all[si] += pe;
        if (pe > 0) {
          const t = s.T[i];
          if (Number.isNaN(pmtT_all[si]) || t < pmtT_all[si]) pmtT_all[si] = t;
          pmtHasSignal_all[si] = 1;
        }
      }
    } else {
      for (let si = 0; si < nSensors; si++) {
        const d = selectedDigit[si];
        if (d < 0) continue;
        const pe = s.PE[d];
        pmtPE_all[si] = pe;
        pmtT_all[si] = s.T[d];
        if (pe > 0) pmtHasSignal_all[si] = 1;
      }
    }
  }

  // Per-process slices: sum hits.h5 rows by emission_process. These drop
  // orphan-track photons (rows where particle_idx < 0 don't reach hits.h5
  // — but those tracks' Cherenkov / scintillation contributions still show
  // up in the 'All' slice via sensor.h5). Pre-smearing.
  pmtPE_cher = new Float32Array(nSensors);
  pmtPE_scint = new Float32Array(nSensors);
  pmtPE_dark = new Float32Array(nSensors);
  pmtT_cher = new Float32Array(nSensors);
  pmtT_scint = new Float32Array(nSensors);
  pmtT_dark = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) { pmtT_cher[i] = NaN; pmtT_scint[i] = NaN; pmtT_dark[i] = NaN; }
  pmtHasSignal_cher = new Uint8Array(nSensors);
  pmtHasSignal_scint = new Uint8Array(nSensors);
  pmtHasSignal_dark = new Uint8Array(nSensors);
  const h = evtBundle.hits;
  if (h && h.nHits && h.emission_process) {
    const sIdx = h.sensor_idx, pe = h.PE, t = h.T, ep = h.emission_process;
    const dg = h.digit_idx;
    for (let i = 0; i < h.nHits; i++) {
      const si = sIdx[i];
      // TRIGGER WINDOW slice: keep only rows whose digit is in the window range.
      if (windowFilter !== 'all' && dg && (dg[i] < wLo || dg[i] >= wHi)) continue;
      // HIT slice: keep only rows belonging to the selected digit on each PMT.
      if (hitFilter !== 0 && (!dg || dg[i] !== selectedDigit[si])) continue;
      const p = pe[i];
      if (!(p > 0)) continue;
      if (ep[i] === 1) {
        pmtPE_scint[si] += p;
        if (Number.isNaN(pmtT_scint[si]) || t[i] < pmtT_scint[si]) pmtT_scint[si] = t[i];
        pmtHasSignal_scint[si] = 1;
      } else if (ep[i] === 2) {
        // emission_process == 2 (electronic dark noise).
        pmtPE_dark[si] += p;
        if (Number.isNaN(pmtT_dark[si]) || t[i] < pmtT_dark[si]) pmtT_dark[si] = t[i];
        pmtHasSignal_dark[si] = 1;
      } else {
        // emission_process == 0 (Cherenkov); also catches any pre-change
        // dataset that lacks the column (h5_worker defaults to all-zeros).
        pmtPE_cher[si] += p;
        if (Number.isNaN(pmtT_cher[si]) || t[i] < pmtT_cher[si]) pmtT_cher[si] = t[i];
        pmtHasSignal_cher[si] = 1;
      }
    }
  }

  // Per-sensor Cherenkov fraction: f = PE_cher / (PE_cher + PE_scint).
  // NaN where neither process contributes (no signal sensor) so the
  // percentile / colormap path masks them out — same convention as pmtBeta.
  pmtCherFraction = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) {
    const num = pmtPE_cher[i];
    const den = num + pmtPE_scint[i];
    pmtCherFraction[i] = (den > 0) ? (num / den) : NaN;
  }

  // applyEmissionFilter() wires the active views (pmtPE / pmtT /
  // pmtHasSignal) to the slice selected by the current dropdown state.
  applyEmissionFilter();
}

// Detect whether the loaded event carries both Cherenkov and scintillation
// rows. Used at first-event-load time to decide whether to expose the
// EMISSION dropdown + CHER FRAC button. Locked across the dataset so the
// toolbar doesn't shape-shift when navigating events with different
// process composition (e.g. an event where every scintillation photon
// happened to QE-fail would otherwise hide the dropdown for that one).
function detectDualEmission(bundle) {
  const ep = bundle && bundle.hits && bundle.hits.emission_process;
  if (!ep || !ep.length) return false;
  // Single-pass any-mismatch check — cheap, terminates on first 1.
  const first = ep[0];
  for (let i = 1; i < ep.length; i++) if (ep[i] !== first) return true;
  return false;
}

// Detect whether the loaded event carries dark-noise rows (emission_process==2).
// Locked across the dataset (like detectDualEmission) so the Dark slice stays
// available even on events where dark noise happened to land nowhere.
function detectDark(bundle) {
  const ep = bundle && bundle.hits && bundle.hits.emission_process;
  if (!ep) return false;
  for (let i = 0; i < ep.length; i++) if (ep[i] === 2) return true;
  return false;
}

// Repoint the active per-sensor arrays to the slice selected by
// emissionFilter. Cheap (alias swap, no copy). Call after derivePMTArrays
// (event load) and after the dropdown changes.
function applyEmissionFilter() {
  if (emissionFilter === 'cher') {
    pmtPE = pmtPE_cher;
    pmtT  = pmtT_cher;
    pmtHasSignal = pmtHasSignal_cher;
  } else if (emissionFilter === 'scint') {
    pmtPE = pmtPE_scint;
    pmtT  = pmtT_scint;
    pmtHasSignal = pmtHasSignal_scint;
  } else if (emissionFilter === 'dark') {
    pmtPE = pmtPE_dark;
    pmtT  = pmtT_dark;
    pmtHasSignal = pmtHasSignal_dark;
  } else {
    pmtPE = pmtPE_all;
    pmtT  = pmtT_all;
    pmtHasSignal = pmtHasSignal_all;
  }
}

// Project per-segment β_start onto sensors via the edep/sensor_hits map.
// For each sensor s:
//   pmtBeta[s] = Σ_rows[sensor==s] (β_start[seg_idx] · PE) / Σ_rows[sensor==s] PE
// PE-weighting is the natural choice — a row's PE is the number of detected
// photons from that segment at that sensor, so a segment with stronger
// photon yield carries proportionally more weight in the average.
// pmtBeta[s] = NaN when no row references s (rendered as 0 / dark via the
// pmtHasSignal mask).
function deriveBetaProjection() {
  pmtBeta = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) pmtBeta[i] = NaN;
  const edep = evtBundle && evtBundle.edep;
  if (!edep|| !edep.sensor_hits || !edep.beta_start || !edep.n) return;
  const sh = edep.sensor_hits;
  const segIdx = sh.segment_idx, sensorIdx = sh.sensor_idx, peArr = sh.PE;
  const beta = edep.beta_start;
  if (!segIdx || !sensorIdx || !peArr) return;
  const wsum = new Float64Array(nSensors);
  const bsum = new Float64Array(nSensors);
  for (let r = 0; r < segIdx.length; r++) {
    const sg = segIdx[r], sn = sensorIdx[r], w = peArr[r];
    if (sg < 0 || sn < 0 || sg >= edep.n || sn >= nSensors) continue;
    if (!(w > 0)) continue;
    const b = beta[sg];
    if (!Number.isFinite(b)) continue;
    bsum[sn] += b * w;
    wsum[sn] += w;
  }
  for (let i = 0; i < nSensors; i++) {
    if (wsum[i] > 0) pmtBeta[i] = bsum[i] / wsum[i];
  }
}

function buildHitsLookups() {
  const n_particles = evtBundle.labl.n_particles || 0;
  particleToSensor = [];
  for (let p = 0; p < n_particles; p++) particleToSensor.push(new Map());
  particleTotals = new Float32Array(n_particles);
  particleInteraction = new Int32Array(n_particles); for (let p = 0; p < n_particles; p++) particleInteraction[p] = -1;
  pmtDomParticle = new Int32Array(nSensors);
  for (let i = 0; i < nSensors; i++) pmtDomParticle[i] = -1;

  // Derive per-particle interaction from per_track (any track of that
  // particle gives the same value; take the first we see).
  const pt = evtBundle.labl.per_track;
  if (pt && pt.particle_idx) {
    for (let t = 0; t < pt.particle_idx.length; t++) {
      const p = pt.particle_idx[t];
      if (p < 0 || p >= n_particles) continue;
      if (particleInteraction[p] < 0 && pt.interaction) particleInteraction[p] = pt.interaction[t];
    }
  }

  particlePdgBucket = computePdgBuckets(evtBundle.labl);

  const i_ = evtBundle.hits;
  if (!i_ || !i_.nHits) return;

  const perSensorBest = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) perSensorBest[i] = -Infinity;

  // EMISSION filter applies to LABEL coloring too: when the user has
  // restricted to one process, the per-particle contribution maps + the
  // sidebar totals reflect that process only. The filter value lives on
  // emissionFilter and the per-row tag on i_.emission_process (always
  // present — h5_worker defaults to all-zeros on pre-Phase-0 datasets).
  const ep = i_.emission_process;
  const wantCher  = (emissionFilter === 'cher');
  const wantScint = (emissionFilter === 'scint');
  const wantDark  = (emissionFilter === 'dark');
  const dg = i_.digit_idx;
  const [wLo, wHi] = windowRange();
  for (let i = 0; i < i_.nHits; i++) {
    const s = i_.sensor_idx[i];
    // TRIGGER WINDOW slice: keep only rows whose digit is in the window range.
    if (windowFilter !== 'all' && dg && (dg[i] < wLo || dg[i] >= wHi)) continue;
    // HIT slice: keep only rows belonging to the selected digit on each PMT.
    if (hitFilter !== 0 && (!dg || dg[i] !== selectedDigit[s])) continue;
    const e = ep ? ep[i] : 0;
    if (wantCher && e !== 0) continue;
    if (wantScint && e !== 1) continue;
    if (wantDark && e !== 2) continue;
    const pe = i_.PE[i];
    if (wantDark) {
      // Dark-noise rows have no owning particle (particle_idx == -1); colour
      // the dominated sensor with the dedicated DARK category instead.
      if (pe > perSensorBest[s]) { perSensorBest[s] = pe; pmtDomParticle[s] = DARK_LABEL; }
      continue;
    }
    const p = i_.particle_idx[i];
    if (p < 0 || p >= n_particles) continue;   // dark/orphan rows under non-dark slices
    particleToSensor[p].set(s, (particleToSensor[p].get(s) || 0) + pe);
    particleTotals[p] += pe;
    if (pe > perSensorBest[s]) {
      perSensorBest[s] = pe;
      pmtDomParticle[s] = p;
    }
  }
  rebuildPdgRows();
}

// Build the PDG-mode sidebar rows. Called after particleTotals is
// populated. One row per particle (matches Particle-mode rows 1:1, just
// re-coloured by PDG bucket). Particles with zero PE contribution are
// excluded so the sidebar only lists rows that actually correspond to a
// detector signature.
//   pdgRows[i] = { id, bucket, particleIds: [pid], totalPE }
function rebuildPdgRows() {
  pdgRows = [];
  if (!evtBundle || !particlePdgBucket || !particleTotals) return;
  const n_p = (evtBundle.labl.n_particles || 0);
  for (let p = 0; p < n_p; p++) {
    if ((particleTotals[p] || 0) <= 0) continue;     // hide zero-PE particles
    pdgRows.push({
      bucket: particlePdgBucket[p],
      particleIds: [p],
      totalPE: particleTotals[p],
    });
  }
  // Order: by bucket asc (μ⁻ first … "other" last), then by PE desc within
  // each bucket so the most luminous of, say, three π⁺ shows first.
  pdgRows.sort((a, b) =>
    a.bucket - b.bucket || b.totalPE - a.totalPE);
  for (let i = 0; i < pdgRows.length; i++) pdgRows[i].id = i;
}

// Mirror of buildHitsLookups for the edep/sensor_hits subgroup. Populates
// pmtDomSegment / segmentToSensor / segmentTotals from the flat
// (segment, sensor) PE rows. All three stay null when sensor_hits is
// absent — Label=Segment renders a blank in that case.
function buildSegmentLookups() {
  pmtDomSegment = null;
  segmentToSensor = null;
  segmentTotals = null;

  const edep = evtBundle && evtBundle.edep;
  const hits = edep && edep.sensor_hits;
  const nSeg = edep ? edep.n : 0;
  if (!hits || !nSeg) return;

  pmtDomSegment = new Int32Array(nSensors);
  for (let i = 0; i < nSensors; i++) pmtDomSegment[i] = -1;
  const perSensorBest = new Float32Array(nSensors);
  for (let i = 0; i < nSensors; i++) perSensorBest[i] = -Infinity;

  segmentToSensor = new Array(nSeg);
  for (let i = 0; i < nSeg; i++) segmentToSensor[i] = new Map();
  segmentTotals = new Float32Array(nSeg);

  const segIdx = hits.segment_idx, senIdx = hits.sensor_idx, pe = hits.PE;
  const ep = hits.emission_process;
  const n = pe ? pe.length : 0;
  // EMISSION filter applies to the LABEL=Segment side too: sensors get
  // colored by the segment that dominates *the selected process's*
  // contribution. The segment table itself stays untouched (segments
  // don't have an intrinsic process — they emit both Cherenkov and
  // scintillation simultaneously; only the per-(segment, sensor) hits
  // carry the process tag).
  const wantCher  = (emissionFilter === 'cher');
  const wantScint = (emissionFilter === 'scint');
  const wantDark  = (emissionFilter === 'dark');   // sensor_hits carries no dark rows
  for (let i = 0; i < n; i++) {
    if (wantDark) continue;   // dark slice has no segment contributions
    if (ep) {
      const e = ep[i];
      if (wantCher && e !== 0) continue;
      if (wantScint && e !== 1) continue;
    }
    const sg = segIdx[i];
    const s = senIdx[i];
    const v = pe[i];
    if (sg < 0 || sg >= nSeg) continue;
    segmentToSensor[sg].set(s, (segmentToSensor[sg].get(s) || 0) + v);
    segmentTotals[sg] += v;
    if (v > perSensorBest[s]) {
      perSensorBest[s] = v;
      pmtDomSegment[s] = sg;
    }
  }
}

// ── contVal / catVal field builders ─────────────────────────────────────

function pmtContValArray() {
  const isTime = curField === 'time';
  const isBeta = curField === 'beta';
  const isCherFrac = curField === 'cher_frac';
  // Cherenkov fraction: f = PE_cher / (PE_cher + PE_scint). Bounded in
  // [0, 1] with a natural anchor at 0.5 → use the diverging rdbu map and
  // disable log scale (same handling as β). Sensors with no signal in
  // either process get NaN and are masked out of the percentile pool —
  // same convention as pmtBeta. The cher/scint masks are unioned because
  // a sensor lit by either process is a valid sample of the fraction.
  if (isCherFrac) {
    const mask = new Uint8Array(nSensors);
    for (let i = 0; i < nSensors; i++) {
      mask[i] = (pmtHasSignal_cher && pmtHasSignal_scint
                 && (pmtHasSignal_cher[i] || pmtHasSignal_scint[i])) ? 1 : 0;
    }
    return normalizeValues(pmtCherFraction || new Float32Array(nSensors), {
      isTime: false,
      isLog: false,
      pMin: 0, pMax: 100,   // no percentile clip — fraction is already in [0,1]
      mVmin: 0, mVmax: 1,   // anchor the diverging map symmetrically
      mask,
    });
  }
  // β is intrinsically bounded in [0, 1] but only β > 1/n produces
  // Cherenkov in the active medium. Remap β ∈ [β_thresh, 1] → [0, 1]
  // so the viridis ramp covers the physically meaningful range; sub-
  // threshold β clamps to 0. Sensors without a β projection (no
  // edep/sensor_hits row, or segmap not stored) get NaN and are
  // excluded from the percentile pool by normalizeValues' finite check.
  if (isBeta) {
    const remapped = new Float32Array(nSensors);
    const bt = cherenkovBetaThreshold(detectorMaterial);
    const denom = Math.max(1e-6, 1.0 - bt);
    if (pmtBeta) {
      for (let i = 0; i < nSensors; i++) {
        const b = pmtBeta[i];
        remapped[i] = (pmtHasSignal[i] && Number.isFinite(b))
          ? Math.max(0, Math.min(1, (b - bt) / denom))
          : NaN;
      }
    } else {
      for (let i = 0; i < nSensors; i++) remapped[i] = NaN;
    }
    // Route through normalizeValues so the panel's vmin/vmax + percentile
    // sliders work in remapped-β space. log is forced off — log on a
    // bounded ratio doesn't carry useful structure.
    return normalizeValues(remapped, {
      isTime: false,
      isLog: false,
      pMin: percMin, pMax: percMax,
      mVmin: manualVmin, mVmax: manualVmax,
      mask: pmtHasSignal,
    });
  }
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

// Union quantile map for 'both' scope: pool signal-PMT T + all edep times
// into one distribution so identical physical times map to identical ranks.
function refreshUnionQMap() {
  if (quantileScope !== 'both' || !evtBundle || !pmtT || !pmtHasSignal) {
    unionQMap = null;
    return;
  }
  const edep = evtBundle.edep;
  const pool = [];
  for (let i = 0; i < pmtT.length; i++) {
    if (pmtHasSignal[i] && Number.isFinite(pmtT[i])) pool.push(pmtT[i]);
  }
  if (edep && edep.time) {
    for (let i = 0; i < edep.time.length; i++) {
      if (Number.isFinite(edep.time[i])) pool.push(edep.time[i]);
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

// PDG-bucket coloring. Fixed palette so a given bucket reads with the
// same hue across every event and label-mode toggle.
//   0=μ⁻ 1=μ⁺ 2=π⁺ 3=π⁻ 4=π⁰ 5=e⁻ 6=e⁺ 7=p 8=n 9=γ 10=ν 11=other meson 12=other
const PDG_BUCKET_NAMES = [
  'μ⁻','μ⁺','π⁺','π⁻','π⁰','e⁻','e⁺','p','n','γ','ν','other meson','other'
];
const PDG_BUCKET_HUE = [
  0.61,  // μ⁻ — blue
  0.78,  // μ⁺ — purple
  0.00,  // π⁺ — red
  0.08,  // π⁻ — orange
  0.16,  // π⁰ — yellow
  0.34,  // e⁻ — green
  0.50,  // e⁺ — cyan
  0.10,  // p  — brown-ish (low S/L applied where used)
  0.55,  // n  — steel teal (neutral hadron)
  0.18,  // γ  — pale gold (visually close-but-distinct from π⁰)
  0.95,  // ν  — faded violet (S/L low → "almost invisible")
  0.90,  // other meson — magenta
  0.00,  // other — gray (S=0 below)
];
const PDG_BUCKET_SAT = [
  0.78, 0.78, 0.78, 0.78, 0.85, 0.78, 0.78, 0.45, 0.45, 0.55, 0.20, 0.78, 0.00,
];
const PDG_BUCKET_LIT = [
  0.55, 0.55, 0.55, 0.55, 0.55, 0.50, 0.55, 0.40, 0.50, 0.65, 0.45, 0.55, 0.55,
];
// Other-meson PDG codes (|pdg|).
const OTHER_MESON_ABS_PDG = new Set([
  130, 310, 311, 321,            // K⁰_L, K⁰_S, K⁰, K±
  411, 421, 431,                 // D±, D⁰, D_s±
  511, 521, 531,                 // B⁰, B±, B_s
  221, 331, 333,                 // η, η′, φ
  443, 553,                      // J/ψ, Υ
]);
// Neutrinos (|pdg|).
const NEUTRINO_ABS_PDG = new Set([12, 14, 16]);
function pdgBucket(pdg) {
  switch (pdg) {
    case 13:   return 0;   // μ⁻
    case -13:  return 1;   // μ⁺
    case 211:  return 2;   // π⁺
    case -211: return 3;   // π⁻
    case 111:  return 4;   // π⁰
    case 11:   return 5;   // e⁻
    case -11:  return 6;   // e⁺
    case 2212: return 7;   // p
    case 2112: return 8;   // n
    case 22:   return 9;   // γ
  }
  if (NEUTRINO_ABS_PDG.has(Math.abs(pdg))) return 10;
  if (OTHER_MESON_ABS_PDG.has(Math.abs(pdg))) return 11;
  return 12;
}
function pdgBucketHue(b) { return PDG_BUCKET_HUE[(b|0) % PDG_BUCKET_HUE.length]; }
function pdgBucketRGB(b) {
  const i = (b|0) % PDG_BUCKET_HUE.length;
  return hsl2rgb(PDG_BUCKET_HUE[i], PDG_BUCKET_SAT[i], PDG_BUCKET_LIT[i]);
}

// Compute per-particle bucket from genealogy + per_track:
//   own PDG = pdg of last (leaf) track in genealogy chain
//   override to π⁰ bucket if any track in the chain has pdg==111
function computePdgBuckets(labl) {
  const n_p = labl ? (labl.n_particles || 0) : 0;
  const buckets = new Int8Array(n_p);
  buckets.fill(9);   // 'other' default
  if (n_p === 0) return buckets;
  const pp = labl.per_particle || {};
  const pt = labl.per_track || {};
  if (!pt.track_id || !pt.pdg || !pp.genealogy || !pp.genealogy_offsets) return buckets;
  const pdgByTrackId = new Map();
  for (let i = 0; i < pt.track_id.length; i++) pdgByTrackId.set(pt.track_id[i], pt.pdg[i]);
  const gen = pp.genealogy, off = pp.genealogy_offsets;
  for (let p = 0; p < n_p; p++) {
    const s = off[p], e = off[p + 1];
    if (e <= s) continue;
    let hasPi0 = false;
    for (let k = s; k < e; k++) {
      if (pdgByTrackId.get(gen[k]) === 111) { hasPi0 = true; break; }
    }
    if (hasPi0) { buckets[p] = 4; continue; }
    const leafPdg = pdgByTrackId.get(gen[e - 1]);
    if (leafPdg !== undefined) buckets[p] = pdgBucket(leafPdg);
  }
  return buckets;
}

// Hue for the "group" label modes (pdg, interaction).
// PDG uses the fixed palette keyed on bucket index; interaction hashes
// by golden ratio.
function groupHue(kind, id) {
  if (kind === 'pdg') return pdgBucketHue(id);
  return hashHue(id);
}

// Hue for the currently-selected group. PDG selection IDs are row indices
// into pdgRows (not buckets), so we dereference here. Other kinds use the
// id directly.
function selectedGroupHue() {
  if (!selectedGroup) return 0;
  if (selectedGroup.kind === 'pdg') {
    if (!pdgRows || selectedGroup.id < 0 || selectedGroup.id >= pdgRows.length) return 0;
    return pdgBucketHue(pdgRows[selectedGroup.id].bucket);
  }
  return groupHue(selectedGroup.kind, selectedGroup.id);
}

// For a given particle index, return the "group id" for the active label.
function particleGroupId(p) {
  if (p < 0) return -1;
  if (curLabel === 'pdg') {
    return particlePdgBucket ? particlePdgBucket[p] : -1;
  }
  if (curLabel === 'interaction') return particleInteraction ? particleInteraction[p] : -1;
  return p;
}

function pmtCatValArray() {
  const out = new Float32Array(nSensors);
  if (curLabel === 'segment') {
    if (!pmtDomSegment) return out;   // sensor_hits absent — blank panel
    for (let i = 0; i < nSensors; i++) {
      const sg = pmtDomSegment[i];
      out[i] = sg < 0 ? 0 : hashHue(sg);
    }
    return out;
  }
  for (let i = 0; i < nSensors; i++) {
    const p = pmtDomParticle[i];
    if (p === DARK_LABEL) { out[i] = DARK_HUE; continue; }   // dark-noise category
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

function edepContValArrays() {
  // Edep continuous source follows Field (since edep-only labels are gone):
  //   charge → edep,  time → time,  beta → beta_start.
  const edep = evtBundle.edep;
  if (!edep|| !edep.n) return { contPerEdep: null, vmin: 0, vmax: 1 };
  const isTimeField = curField === 'time';
  const isBetaField = curField === 'beta';
  // β remap to [β_thresh, 1] → [0, 1]; matches the PMT β projection
  // so a sensor and the segments contributing to it read with the same
  // hue when both are above threshold. Routed through normalizeValues
  // (with log forced off) so panel vmin/vmax/percentile sliders work.
  if (isBetaField) {
    const remapped = new Float32Array(edep.n);
    const b = edep.beta_start;
    if (b) {
      const bt = cherenkovBetaThreshold(detectorMaterial);
      const denom = Math.max(1e-6, 1.0 - bt);
      for (let i = 0; i < edep.n; i++) {
        remapped[i] = Number.isFinite(b[i])
          ? Math.max(0, Math.min(1, (b[i] - bt) / denom))
          : NaN;
      }
    } else {
      for (let i = 0; i < edep.n; i++) remapped[i] = NaN;
    }
    const { norm, vmin, vmax } = normalizeValues(remapped, {
      isTime: false,
      isLog: false,
      pMin: percMin, pMax: percMax,
      mVmin: manualVmin, mVmax: manualVmax,
      mask: null,
    });
    return { contPerEdep: norm, vmin, vmax };
  }
  const field = isTimeField ? edep.time : edep.edep;
  if (!field) return { contPerEdep: new Float32Array(edep.n), vmin: 0, vmax: 1 };
  const f32 = field instanceof Float32Array ? field : Float32Array.from(field);

  // Quantile for time: same treatment as PMTs; union map when scope=both.
  if (isTimeField && quantileEdep()) {
    const q = unionQMap || buildQuantileMapMasked(f32, null);
    const norm = new Float32Array(edep.n);
    for (let i = 0; i < edep.n; i++) {
      const v = f32[i];
      norm[i] = (Number.isFinite(v) && q.has(v)) ? q.get(v) : 0;
    }
    return { contPerEdep: norm, vmin: 0, vmax: 1 };
  }

  const { norm, vmin, vmax } = normalizeValues(f32, {
    isTime: isTimeField,
    isLog: logScale,
    pMin: percMin, pMax: percMax,
    mVmin: manualVmin, mVmax: manualVmax,
    mask: null,
  });
  return { contPerEdep: norm, vmin, vmax };
}

function edepCatValArrays() {
  const edep = evtBundle.edep;
  if (!edep|| !edep.n) return null;
  const per_track = evtBundle.labl.per_track;
  const per_particle = evtBundle.labl.per_particle;
  const out = new Float32Array(edep.n);
  for (let i = 0; i < edep.n; i++) {
    const t = edep.track_idx[i];
    if (curLabel === 'particle') {
      const pidx = per_track.particle_idx ? per_track.particle_idx[t] : -1;
      out[i] = pidx >= 0 ? hashHue(pidx) : 0;
    } else if (curLabel === 'pdg') {
      const pidx = per_track.particle_idx ? per_track.particle_idx[t] : -1;
      out[i] = (pidx >= 0 && particlePdgBucket) ? pdgBucketHue(particlePdgBucket[pidx]) : 0;
    } else if (curLabel === 'interaction') {
      const k = per_track.interaction ? per_track.interaction[t] : -1;
      out[i] = k >= 0 ? hashHue(k) : 0;
    } else if (curLabel === 'segment') {
      out[i] = hashHue(i);
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
  //   edep:  raw ns (PMT not affected by 'edep' scope)
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
const EDEP_POINTS_PER = 6;

function buildEdeps() {
  if (edepMesh) {
    scene.remove(edepMesh);
    edepGeo.dispose();
    if (edepMat) edepMat.dispose();
    edepMesh = null;
  }
  const edep = evtBundle.edep;
  if (!edep|| !edep.n) return;

  const n = edep.n;
  const K = EDEP_POINTS_PER;
  const N = n * K;
  const pos = new Float32Array(N * 3);
  const contVal = new Float32Array(N);
  const catVal = new Float32Array(N);
  const hl = new Float32Array(N);
  const arrivalT = new Float32Array(N);
  const hasSig = new Float32Array(N);

  // Segment arrivalT respects scope: rank when edep-quantiled (or 'both'),
  // raw ns otherwise.
  const edepQMap = quantileEdep() ? (unionQMap || buildQuantileMap(edep.time)) : null;
  if (edepQMap) {
    edepTRange = [0, 1];
  } else {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = edep.time[i];
      if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
    }
    edepTRange = [mn === Infinity ? 0 : mn, mx === -Infinity ? 1 : mx];
  }

  for (let i = 0; i < n; i++) {
    const sx = edep.start_x[i], sy = edep.start_y[i], sz = edep.start_z[i];
    const ex = edep.end_x[i],   ey = edep.end_y[i],   ez = edep.end_z[i];
    const t = edep.time[i];
    const tMapped = edepQMap ? edepQMap.get(t) : t;
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

  edepGeo = new THREE.BufferGeometry();
  edepGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const contAttr = new THREE.BufferAttribute(contVal, 1);
  contAttr.setUsage(THREE.DynamicDrawUsage);
  edepGeo.setAttribute('contVal', contAttr);
  const catAttr = new THREE.BufferAttribute(catVal, 1);
  catAttr.setUsage(THREE.DynamicDrawUsage);
  edepGeo.setAttribute('catVal', catAttr);
  const hlAttr = new THREE.BufferAttribute(hl, 1);
  hlAttr.setUsage(THREE.DynamicDrawUsage);
  edepGeo.setAttribute('hl', hlAttr);
  edepGeo.setAttribute('arrivalT', new THREE.BufferAttribute(arrivalT, 1));
  edepGeo.setAttribute('hasSignal', new THREE.BufferAttribute(hasSig, 1));

  edepMat = new THREE.ShaderMaterial({
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

  edepMesh = new THREE.Points(edepGeo, edepMat);
  edepMesh.visible = (curView === 'edep');
  scene.add(edepMesh);

  updateEdepColors();
}

// ── Detector outline (cylinder/box/sphere wireframe) ───────────────────
// Uses three.js fat lines (LineSegments2 + LineMaterial) so thickness
// works across browsers. WebGL/ANGLE silently caps the regular
// LineBasicMaterial.linewidth at 1px on Chrome/Safari.
function buildOutline() {
  if (outlineMesh) { scene.remove(outlineMesh); outlineMesh = null; }
  outlineMat = null;
  if (!showMesh) return;

  const segs = []; // flat [x0,y0,z0, x1,y1,z1, ...] — pairs of endpoints.
  const push = (x0,y0,z0, x1,y1,z1) => segs.push(x0,y0,z0, x1,y1,z1);

  const t = (detectorType || '').toLowerCase();
  if (t === 'cylinder') {
    const r = shape.r, hh = shape.halfH, seg = 64;
    for (let i = 0; i < seg; i++) {
      const a0 = 2*Math.PI*i/seg, a1 = 2*Math.PI*(i+1)/seg;
      push(r*Math.cos(a0), r*Math.sin(a0),  hh, r*Math.cos(a1), r*Math.sin(a1),  hh);
      push(r*Math.cos(a0), r*Math.sin(a0), -hh, r*Math.cos(a1), r*Math.sin(a1), -hh);
    }
    for (let i = 0; i < 8; i++) {
      const a = 2*Math.PI*i/8;
      push(r*Math.cos(a), r*Math.sin(a), -hh, r*Math.cos(a), r*Math.sin(a),  hh);
    }
  } else if (t === 'box') {
    const hL = shape.L/2, hW = shape.W/2, hH = shape.H/2;
    const c = [
      [-hL,-hW,-hH],[hL,-hW,-hH],[hL,hW,-hH],[-hL,hW,-hH],
      [-hL,-hW, hH],[hL,-hW, hH],[hL,hW, hH],[-hL,hW, hH],
    ];
    for (const [a,b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]) {
      push(c[a][0],c[a][1],c[a][2], c[b][0],c[b][1],c[b][2]);
    }
  } else if (t === 'sphere') {
    const r = shape.r, seg = 96;
    for (let axis = 0; axis < 3; axis++) {
      for (let i = 0; i < seg; i++) {
        const a0 = 2*Math.PI*i/seg, a1 = 2*Math.PI*(i+1)/seg;
        let p0, p1;
        if (axis === 0)      { p0 = [r*Math.cos(a0), r*Math.sin(a0), 0]; p1 = [r*Math.cos(a1), r*Math.sin(a1), 0]; }
        else if (axis === 1) { p0 = [0, r*Math.cos(a0), r*Math.sin(a0)]; p1 = [0, r*Math.cos(a1), r*Math.sin(a1)]; }
        else                 { p0 = [r*Math.cos(a0), 0, r*Math.sin(a0)]; p1 = [r*Math.cos(a1), 0, r*Math.sin(a1)]; }
        push(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]);
      }
    }
  } else {
    return;
  }

  const fat = (typeof THREE.LineSegments2 !== 'undefined') &&
              (typeof THREE.LineMaterial !== 'undefined') &&
              (typeof THREE.LineSegmentsGeometry !== 'undefined');

  if (fat) {
    const g = new THREE.LineSegmentsGeometry();
    g.setPositions(segs);
    const mat = new THREE.LineMaterial({
      color: 0x28394a, transparent: true, opacity: 0.5, linewidth: outlineWidth,
    });
    // LineMaterial linewidth is in screen pixels by default and needs the
    // renderer resolution to compute the thickness in clip space.
    const sz = renderer.getSize(new THREE.Vector2());
    mat.resolution.set(sz.x, sz.y);
    outlineMat = mat;
    const ls = new THREE.LineSegments2(g, mat);
    ls.computeLineDistances();
    outlineMesh = ls;
  } else {
    // Fallback: plain LineSegments. Thickness slider will visibly affect
    // Firefox only.
    const arr = new Float32Array(segs);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x28394a, transparent: true, opacity: 0.5, linewidth: outlineWidth });
    outlineMat = mat;
    outlineMesh = new THREE.LineSegments(g, mat);
  }
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
  // Re-apply any active selection so the catVal override survives — this
  // canonical write would otherwise wipe the selection-hue paint.
  applyCorrespondence();
}

function pmtContValArrayForField() {
  // Always keyed to current Charge/Time toggle, not label.
  return pmtContValArray();
}

function updateEdepColors() {
  if (!edepMat || !edepGeo) return;
  const edep = evtBundle.edep;
  if (!edep|| !edep.n) return;

  const isCat = (curLabel !== 'none');
  edepMat.uniforms.colorMode.value = isCat ? 1.0 : 0.0;
  edepMat.uniforms.cmap.value = currentCmapTex();
  const K = EDEP_POINTS_PER;

  if (isCat) {
    const cat = edepCatValArrays();
    const out = edepGeo.attributes.catVal.array;
    for (let i = 0; i < edep.n; i++) {
      const v = cat[i];
      for (let k = 0; k < K; k++) out[i * K + k] = v;
    }
    edepGeo.attributes.catVal.needsUpdate = true;
  } else {
    const { contPerEdep } = edepContValArrays();
    const out = edepGeo.attributes.contVal.array;
    for (let i = 0; i < edep.n; i++) {
      const v = contPerEdep[i];
      for (let k = 0; k < K; k++) out[i * K + k] = v;
    }
    edepGeo.attributes.contVal.needsUpdate = true;
  }
  // Re-apply the selection override (it lives in the same buffer we
  // just rewrote canonically).
  applyCorrespondence();
}

// ── Correspondence: isolate the selected item ─────────────────────────
// A selection is either a specific particle (when label ≠ pdg) or
// a PDG-mode row (when label = pdg). Both resolve to a set of particles
// whose inst contributions we union per sensor.
function currentParticleSet() {
  if (!evtBundle) return null;
  if (selectedGroup) {
    if (selectedGroup.kind === 'segment') return [];   // not particle-decomposed
    const n_particles = evtBundle.labl.n_particles || 0;
    if (selectedGroup.kind === 'pdg') {
      // PDG selection id is a row index into pdgRows. Each row carries an
      // explicit particleIds list (singleton for non-shower buckets, the
      // full shower for e±/π⁰ groupings).
      if (!pdgRows || selectedGroup.id < 0 || selectedGroup.id >= pdgRows.length) return null;
      return pdgRows[selectedGroup.id].particleIds.slice();
    }
    const out = [];
    if (selectedGroup.kind === 'interaction') {
      for (let p = 0; p < n_particles; p++) if (particleInteraction[p] === selectedGroup.id) out.push(p);
    }
    return out;
  }
  if (selectedParticle != null) return [selectedParticle];
  return null;
}

// Sensor-PE Map for the active selection. Returns null when nothing is
// selected OR the selection has zero contributions (so callers can use
// "is this null?" as the single corrActive predicate). Segment selection
// routes through segmentToSensor; everything else is the union of
// per-particle inst contributions.
function selectionContributions() {
  if (!evtBundle) return null;
  if (selectedGroup && selectedGroup.kind === 'segment') {
    if (!segmentToSensor) return null;
    const m = segmentToSensor[selectedGroup.id];
    return (m && m.size > 0) ? m : null;
  }
  const ps = currentParticleSet();
  if (ps == null || ps.length === 0) return null;
  const m = unionContributions(ps);
  return (m && m.size > 0) ? m : null;
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
  const segHL = edepGeo ? edepGeo.attributes.hl.array : null;
  pmtHL.fill(0);
  if (segHL) segHL.fill(0);

  const particleSet = currentParticleSet();
  const segSelected = selectedGroup && selectedGroup.kind === 'segment';
  const map = selectionContributions();
  const corrActive = map != null;
  pmtMat.uniforms.corrOn.value = corrActive ? 1.0 : 0.0;
  if (edepMat) edepMat.uniforms.corrOn.value = corrActive ? 1.0 : 0.0;

  // Reset PMT catVal to the canonical (dominant-particle) hue. We do
  // this unconditionally so a prior selection-hue override is wiped
  // before we (maybe) write a new one — keeps applyCorrespondence
  // idempotent across selection toggles.
  const isCat = (curLabel !== 'none');
  const pmtCV = pmtGeo.attributes.catVal.array;
  if (isCat) {
    pmtCV.set(pmtCatValArray());
    pmtGeo.attributes.catVal.needsUpdate = true;
  }
  // Same for edep catVal: a per-track binary highlight + a hue override
  // for tracks belonging to the selection so the 3D edep points read as
  // the selection's color, mirroring the PMT path. updatePMTColors can
  // chain in here before buildEdeps has rebuilt edepGeo for the new
  // event, so guard against a buffer-size mismatch — buildEdeps will
  // eventually call updateEdepColors → applyCorrespondence with the
  // correct geometry.
  let segCV = null;
  let segOK = false;
  const K = EDEP_POINTS_PER;
  if (edepGeo && isCat && evtBundle && evtBundle.edep) {
    const edep = evtBundle.edep;
    const buf = edepGeo.attributes.catVal.array;
    if (edep && edep.n && buf.length === edep.n * K) {
      segCV = buf;
      segOK = true;
      const segCat = edepCatValArrays();
      if (segCat) {
        for (let i = 0; i < edep.n; i++) {
          const v = segCat[i];
          for (let k = 0; k < K; k++) segCV[i * K + k] = v;
        }
        edepGeo.attributes.catVal.needsUpdate = true;
      }
    }
  }

  if (corrActive) {
    // Selection hue override: in pile-up overlap, a sensor can receive PE
    // from the selection AND from another group whose particle dominates.
    // Painting such a contributor in the dominant particle's hue makes it
    // look like another group's hit is being highlighted. Recolor
    // contributors with the selection's own hue instead. (The 2D panel
    // does the same trick at render time; here we bake it into catVal.)
    let selectionHue = 0;
    let useSelHue = false;
    if (isCat) {
      if (selectedGroup) {
        selectionHue = selectedGroupHue();
        useSelHue = true;
      } else if (selectedParticle != null) {
        selectionHue = hashHue(selectedParticle);
        useSelHue = true;
      }
    }

    // PMT contributions: per-segment when a segment row is selected;
    // otherwise the per-particle union. Already computed at the top of
    // applyCorrespondence — reuse.
    if (map) {
      let maxPE = 0;
      for (const v of map.values()) if (v > maxPE) maxPE = v;
      if (maxPE > 0) {
        for (const [s, pe] of map) {
          if (!(pe > 0)) continue;
          const frac = Math.min(1, pe / maxPE);
          pmtHL[s] = 0.35 + 0.65 * Math.sqrt(frac);
          if (useSelHue) pmtCV[s] = selectionHue;
        }
        if (useSelHue) pmtGeo.attributes.catVal.needsUpdate = true;
      }
    }
    // Segments: binary highlight if the track belongs to any particle in
    // the selected set (or, when a segment row is selected, just that
    // single segment), and — in categorical mode — override the hue to
    // the selection's color so the bright edep points read consistently.
    const edep = evtBundle.edep;
    const per_track = evtBundle.labl.per_track;
    const segHLOK = segHL && edep && edep.n && segHL.length === edep.n * K;
    if (segHLOK && segSelected) {
      const sid = selectedGroup.id;
      for (let i = 0; i < edep.n; i++) {
        const v = (i === sid) ? 1.0 : 0.0;
        for (let k = 0; k < K; k++) segHL[i * K + k] = v;
        if (i === sid && useSelHue && segOK) {
          for (let k = 0; k < K; k++) segCV[i * K + k] = selectionHue;
        }
      }
      if (useSelHue && segOK) edepGeo.attributes.catVal.needsUpdate = true;
    } else if (segHLOK && per_track && per_track.particle_idx) {
      const setLookup = new Set(particleSet);
      for (let i = 0; i < edep.n; i++) {
        const t = edep.track_idx[i];
        const p = per_track.particle_idx[t];
        const inSel = setLookup.has(p);
        const v = inSel ? 1.0 : 0.0;
        for (let k = 0; k < K; k++) segHL[i * K + k] = v;
        if (inSel && useSelHue && segOK) {
          for (let k = 0; k < K; k++) segCV[i * K + k] = selectionHue;
        }
      }
      if (useSelHue && segOK) edepGeo.attributes.catVal.needsUpdate = true;
    }
  } // end corrActive
  pmtGeo.attributes.hl.needsUpdate = true;
  if (edepGeo) edepGeo.attributes.hl.needsUpdate = true;
}

// ── Sidebar (label-aware) ──────────────────────────────────────────────
function buildSidebar() {
  const list = $('particleList');
  list.innerHTML = '';
  const title = $('sidebarTitle');
  if (title) {
    title.textContent =
      curLabel === 'pdg'     ? 'PDG'      :
      curLabel === 'segment' ? 'SEGMENTS' :
      'PARTICLES';
  }
  const n = evtBundle.labl.n_particles || 0;
  if (curLabel === 'segment') {
    buildSidebarSegments(list);
  } else if (n === 0) {
    list.innerHTML = '<div class="event-meta-row" style="padding:8px"><span class="k">(none)</span></div>';
  } else if (curLabel === 'pdg') {
    buildSidebarPdgRows(list);
  } else if (curLabel === 'interaction') {
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
  addRow('n segments',  String(evtBundle.edep.n));

  // v5 per_interaction summary. Split into labeled rows so the source,
  // probe, vertex, and per-primary energies are individually readable
  // (the .v cell ellipsis-clips at ~140 px wide so a single dense line
  // gets cut off).
  const pi = evtBundle.labl.per_interaction;
  if (pi && pi.t0 && pi.t0.length) {
    const nInt = pi.t0.length;
    if (nInt > 1) addRow('n interactions', String(nInt));
    for (let i = 0; i < nInt; i++) {
      const prefix = (nInt > 1) ? `[int ${i}] ` : '';
      const isGenie = pi.source_type[i] === 1;
      addRow(prefix + 'source', isGenie ? 'GENIE' : 'particle gun');
      if (isGenie) {
        const nuName = pdgName(pi.neutrino_pdg[i]);
        const eMeV = pi.neutrino_energy_MeV[i];
        const eStr = eMeV >= 1000
          ? `${(eMeV / 1000).toFixed(2)} GeV`
          : `${eMeV.toFixed(0)} MeV`;
        addRow(prefix + 'probe', `${nuName} @ ${eStr}`);
      }
      // Per-primary rows: PDG name + energy, one per primary, so each
      // line stays short. Energy comes from primary_energies_data when
      // present (v5+); falls back to "—" if missing.
      const s0 = pi.primary_pdgs_offsets[i];
      const s1 = pi.primary_pdgs_offsets[i + 1];
      const hasE = !!(pi.primary_energies_data && pi.primary_energies_offsets);
      const e0 = hasE ? pi.primary_energies_offsets[i] : 0;
      for (let k = s0; k < s1; k++) {
        const pdg = pi.primary_pdgs_data[k];
        const eMeV = hasE ? pi.primary_energies_data[e0 + (k - s0)] : null;
        const eStr = eMeV == null ? '—' :
          (eMeV >= 1000
            ? `${(eMeV / 1000).toFixed(2)} GeV`
            : `${eMeV.toFixed(0)} MeV`);
        addRow(prefix + `prim ${k - s0}`, `${pdgName(pdg)} · ${eStr}`);
      }
      if (pi.contained) {
        addRow(prefix + 'contained', pi.contained[i] ? 'true' : 'false');
      }
      if (nInt > 1) addRow(prefix + 't0', `${pi.t0[i].toFixed(1)} ns`);
    }
  }

  renderSelectionInfo();
}

function buildSidebarParticles(list, n) {
  let shown = 0;
  for (let p = 0; p < n; p++) {
    // Hide particles that contribute no PE in the active EMISSION/HIT slice,
    // mirroring the PDG rows — so the list tracks the selected hit (a particle
    // whose light all landed in an earlier digit drops out of the Nth-hit view).
    if ((particleTotals[p] || 0) <= 0) continue;
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
    shown++;
  }
  if (shown === 0) {
    list.innerHTML = '<div class="event-meta-row" style="padding:8px"><span class="k">(no particles with hits)</span></div>';
  }
}

// PDG rows: one entry per particle (or per shower for e±/π⁰). Driven
// from precomputed `pdgRows` so zero-PE particles are absent and shower
// grouping is consistent with the selection / hue lookups.
function buildSidebarPdgRows(list) {
  if (!pdgRows || pdgRows.length === 0) {
    list.innerHTML = '<div class="event-meta-row" style="padding:8px"><span class="k">(no particles with hits)</span></div>';
    return;
  }
  for (const row of pdgRows) {
    const r = document.createElement('div');
    r.className = 'particle-row';
    if (selectedGroup && selectedGroup.kind === 'pdg' && selectedGroup.id === row.id) r.classList.add('selected');
    const swatch = document.createElement('span');
    swatch.className = 'particle-swatch';
    const [rr,gg,bb] = pdgBucketRGB(row.bucket);
    swatch.style.background = `rgb(${rr},${gg},${bb})`;
    const label = document.createElement('span');
    label.className = 'particle-label';
    label.textContent = pdgRowLabel(row);
    const meta = document.createElement('span');
    meta.className = 'particle-meta';
    meta.textContent = formatPE(row.totalPE);
    r.appendChild(swatch);
    r.appendChild(label);
    r.appendChild(meta);
    r.addEventListener('click', () => {
      selectedParticle = null;
      const already = selectedGroup && selectedGroup.kind === 'pdg' && selectedGroup.id === row.id;
      selectedGroup = already ? null : { kind: 'pdg', id: row.id };
      buildSidebar();
      applyCorrespondence();
      render2D();
    });
    list.appendChild(r);
  }
}

function pdgRowLabel(row) {
  const name = PDG_BUCKET_NAMES[row.bucket] || ('pdg' + row.bucket);
  return `${name} · P${row.particleIds[0]}`;
}

// List interactions. One row per distinct group id, with the total PE
// and particle count for that group.
function buildSidebarGroups(list, kind) {
  const n_particles = evtBundle.labl.n_particles || 0;
  const groups = new Map();   // id → { n, pe, parts: [] }
  for (let p = 0; p < n_particles; p++) {
    let id;
    if (kind === 'interaction') id = particleInteraction ? particleInteraction[p] : -1;
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
  if (kind === 'pdg') {
    if (!pdgRows || id < 0 || id >= pdgRows.length) return 'pdg row ' + id;
    return pdgRowLabel(pdgRows[id]);
  }
  if (kind === 'interaction') return `interaction ${id} · n=${n}`;
  if (kind === 'segment')     return `segment ${id}`;
  return String(id);
}

// One row per segment that hit at least one PMT, sorted by descending
// total PE. Shows track id and n_sensors hit. Click toggles isolation.
function buildSidebarSegments(list) {
  if (!segmentToSensor) {
    list.innerHTML = '<div class="event-meta-row" style="padding:8px"><span class="k">(edep/sensor_hits absent — re-run with store_segment_sensor_map=true)</span></div>';
    return;
  }
  const edep = evtBundle.edep;
  const per_track = evtBundle.labl.per_track;
  const trackPdg = per_track ? per_track.pdg : null;
  const ordered = [];
  for (let i = 0; i < edep.n; i++) {
    if (segmentTotals[i] > 0) ordered.push(i);
  }
  ordered.sort((a, b) => segmentTotals[b] - segmentTotals[a]);
  if (ordered.length === 0) {
    list.innerHTML = '<div class="event-meta-row" style="padding:8px"><span class="k">(no segments hit any sensor)</span></div>';
    return;
  }
  for (const id of ordered) {
    const row = document.createElement('div');
    row.className = 'particle-row';
    if (selectedGroup && selectedGroup.kind === 'segment' && selectedGroup.id === id) row.classList.add('selected');
    const swatch = document.createElement('span');
    swatch.className = 'particle-swatch';
    const [r,g,b] = hsl2rgb(hashHue(id), 0.78, 0.55);
    swatch.style.background = `rgb(${r},${g},${b})`;
    const label = document.createElement('span');
    label.className = 'particle-label';
    const t = edep.track_idx ? edep.track_idx[id] : -1;
    const pdg = (t >= 0 && trackPdg) ? trackPdg[t] : null;
    const pdgStr = pdg != null ? (PDG_NAMES.get(pdg) || `pdg${pdg}`) : '';
    label.textContent = `S${id} · T${t}${pdgStr ? ' · ' + pdgStr : ''}`;
    const meta = document.createElement('span');
    meta.className = 'particle-meta';
    const nSen = segmentToSensor[id] ? segmentToSensor[id].size : 0;
    meta.textContent = `${formatPE(segmentTotals[id])} · ${nSen} pmt`;
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(meta);
    row.addEventListener('click', () => {
      selectedParticle = null;
      const already = selectedGroup && selectedGroup.kind === 'segment' && selectedGroup.id === id;
      selectedGroup = already ? null : { kind: 'segment', id };
      buildSidebar();
      applyCorrespondence();
      render2D();
    });
    list.appendChild(row);
  }
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
  if (particlePdgBucket) add('pdg bucket', PDG_BUCKET_NAMES[particlePdgBucket[p]] || String(particlePdgBucket[p]));
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
  add('Σ PE (hits)', formatPE(particleTotals[p] || 0));
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
  let parts = [];
  let pe = 0;
  let bucketForHue = id;
  if (kind === 'pdg') {
    if (pdgRows && id >= 0 && id < pdgRows.length) {
      parts = pdgRows[id].particleIds.slice();
      pe = pdgRows[id].totalPE;
      bucketForHue = pdgRows[id].bucket;
    }
  } else {
    for (let p = 0; p < n_particles; p++) {
      let gid;
      if (kind === 'interaction') gid = particleInteraction[p];
      if (gid === id) { parts.push(p); pe += particleTotals[p] || 0; }
    }
  }
  const head = document.createElement('div');
  head.className = 'selection-head';
  const [r,g,b] = hsl2rgb(groupHue(kind, bucketForHue), 0.78, 0.55);
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
  add('Σ PE (hits)', formatPE(pe));
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
  // Swatch color matches the 3D shader. PDG mode uses the fixed bucket
  // palette (so e.g. μ⁻ is always blue); otherwise each particle gets a
  // hashed-golden hue.
  if (curLabel === 'pdg' && particlePdgBucket) {
    const [r,g,b] = pdgBucketRGB(particlePdgBucket[p]);
    return `rgb(${r},${g},${b})`;
  }
  const [r,g,b] = hsl2rgb(hashHue(p), 0.78, 0.55);
  return `rgb(${r},${g},${b})`;
}

const PDG_NAMES = new Map([
  [11,'e⁻'],[-11,'e⁺'],[13,'μ⁻'],[-13,'μ⁺'],[22,'γ'],
  [211,'π⁺'],[-211,'π⁻'],[111,'π⁰'],[2212,'p'],[2112,'n'],
  [12,'νe'],[-12,'ν̄e'],[14,'νμ'],[-14,'ν̄μ'],[16,'ντ'],[-16,'ν̄τ'],
  [321,'K⁺'],[-321,'K⁻'],[130,'K⁰_L'],[310,'K⁰_S'],
]);

function particleLabel(p) {
  if (particlePdgBucket) {
    return `${PDG_BUCKET_NAMES[particlePdgBucket[p]] || 'pdg'+particlePdgBucket[p]} · P${p}`;
  }
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

  // Panel backgrounds (very faint) and labels. Labels render OUTSIDE the
  // rect so they never overlap sensors: 'top' anchor sits above the rect
  // (used for wide strips); 'left' anchor reads bottom→top along the
  // rect's left margin (used for cap squares, where there isn't enough
  // empty corner space inside the disc-of-sensors).
  ctx2d.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx2d.lineWidth = 1;
  ctx2d.font = '10px monospace';
  ctx2d.fillStyle = '#444';
  ctx2d.textAlign = 'left';
  ctx2d.textBaseline = 'alphabetic';
  for (const p of layout.panels) {
    const rx = offX + p.rect.x * scale;
    const ry = offY + p.rect.y * scale;
    const rw = p.rect.w * scale, rh = p.rect.h * scale;
    ctx2d.strokeRect(rx, ry, rw, rh);
    if (p.labelAnchor === 'left') {
      ctx2d.save();
      ctx2d.translate(rx - 4, ry + rh);
      ctx2d.rotate(-Math.PI / 2);
      ctx2d.fillText(p.label, 0, 0);
      ctx2d.restore();
    } else {
      ctx2d.fillText(p.label, rx + 4, ry - 4);
    }
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

  // Selected item (particle, group or segment) influences CPU-side alpha for 2D.
  const corrMap = selectionContributions();
  const corrActive = corrMap != null;
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
      selectionHue = selectedGroupHue();
    } else if (selectedParticle != null) {
      selectionHue = hashHue(selectedParticle);
    }
  }

  // 2D PMT fade: pmtArrivalT lives in PMT-time space (rank or raw), but
  // the active simTime might be in edep-time space (EDEP view). Map sweep
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
      // catArr[i] is a hue in [0,1] (pdgBucketHue or hashHue). Render via HSL
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
  // Log applies only to the charge field — time normalises linear, β
  // is bounded and routed with isLog=false. Suppress the "(log)" tag
  // unless it's actually in effect.
  const logActive = logScale && curField === 'charge';
  ctx2d.fillText(labelText() + (logActive ? ' (log)' : ''), bx, by - 3);
  $('label2dText').textContent = `UNWRAPPED · ${labelText()}`;
}

function labelText() {
  if (curLabel === 'particle') return 'Particle';
  if (curLabel === 'pdg') return 'PDG';
  if (curLabel === 'interaction') return 'Interaction';
  if (curLabel === 'segment') return 'Segment';
  if (curField === 'beta') return 'β (Č-remapped)';
  if (curField === 'cher_frac') return 'Č fraction';
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
  texRdBu = makeCmapTex(RDBU_STOPS);
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
    if (edepMat) edepMat.uniforms.simTime.value = simTime;
    render2D();
  }

  controls.update();
  renderer.render(scene, camera);
}

function applyViewSweepRange() {
  const range = curView === 'edep' ? edepTRange : pmtTRange;
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
  if (edepMat) {
    edepMat.uniforms.sweepEps.value = eps;
    edepMat.uniforms.simTime.value = simTime;
    edepMat.uniforms.sweepOn.value = sweepOn ? 1.0 : 0.0;
  }
}

// Reflect emission-process eligibility in the toolbar:
//   - EMISSION dropdown + CHER FRAC button only present when the dataset
//     carries rows from both Cherenkov and scintillation. Pure datasets
//     keep the legacy toolbar.
//   - CHER FRAC button disabled / faded when emissionFilter != 'all'
//     (single-process slices have trivial f = 0 or 1 — see option (α) in
//     the design notes).
function syncEmissionUI() {
  const sel  = $('emissionSelect');
  const lbl  = $('emissionLabel');
  const sep  = $('emissionSep');
  const cher = $('fieldCherFrac');
  if (!sel || !lbl || !sep || !cher) return;
  const showDual = !!datasetHasBothProcesses;          // both photon processes -> cher/scint + CHER FRAC
  const hasSlices = showDual || !!datasetHasDark;      // any emission slicing (incl. dark)
  sel.style.display  = hasSlices ? '' : 'none';
  lbl.style.display  = hasSlices ? '' : 'none';
  sep.style.display  = hasSlices ? '' : 'none';
  cher.style.display = showDual ? '' : 'none';         // CHER FRAC only when both photon processes exist
  // Per-option visibility: cher/scint only under dual photon processes; Dark
  // only when dark rows exist.
  for (const v of ['cher', 'scint']) {
    const o = sel.querySelector(`option[value="${v}"]`);
    if (o) o.style.display = showDual ? '' : 'none';
  }
  const darkOpt = sel.querySelector('option[value="dark"]');
  if (darkOpt) darkOpt.style.display = datasetHasDark ? '' : 'none';
  sel.value = emissionFilter;
  const cherUsable = showDual && emissionFilter === 'all';
  cher.disabled = !cherUsable;
  cher.style.opacity = cherUsable ? '' : '0.4';
  cher.title = (!showDual)
    ? 'CHER FRAC only available on datasets with both Cherenkov and scintillation'
    : (emissionFilter === 'all'
        ? ''
        : 'CHER FRAC is meaningless under a single-process EMISSION filter — set EMISSION=All to enable.');
}

// HIT dropdown: rebuild the option list for the current event's maxHits and
// show it only when some PMT recorded more than one digit. Called on event
// load (after deriveSensorArrays sets maxHits). Options: All, 1st, 2nd, …
function syncHitUI() {
  const sel = $('hitSelect'), lbl = $('hitLabel'), sep = $('hitSep');
  if (!sel || !lbl || !sep) return;
  const show = maxHits > 1;
  sel.style.display = show ? '' : 'none';
  lbl.style.display = show ? '' : 'none';
  sep.style.display = show ? '' : 'none';
  // Rebuild options only when the count changed (cheap guard against churn).
  if (sel.options.length !== maxHits + 1) {
    const ordinal = (k) => {
      const s = ['th', 'st', 'nd', 'rd'], v = k % 100;
      return k + (s[(v - 20) % 10] || s[v] || s[0]);
    };
    sel.innerHTML = '';
    const optAll = document.createElement('option');
    optAll.value = '0'; optAll.textContent = 'All';
    sel.appendChild(optAll);
    for (let k = 1; k <= maxHits; k++) {
      const o = document.createElement('option');
      o.value = String(k); o.textContent = ordinal(k);
      sel.appendChild(o);
    }
  }
  sel.value = String(hitFilter);
}

// TRIGGER WINDOW dropdown: shown on triggered datasets (per_window present).
// `All` = every stored digit; `Wk` restricts to readout window k, and the HIT
// dropdown then indexes within that window. Labeled by each gate's time range.
function syncWindowUI() {
  const sel = $('windowSelect'), lbl = $('windowLabel'), sep = $('windowSep');
  if (!sel || !lbl || !sep) return;
  const nwin = windows && windows.window_start ? windows.window_start.length : 0;
  const show = nwin >= 1;
  sel.style.display = show ? '' : 'none';
  lbl.style.display = show ? '' : 'none';
  sep.style.display = show ? '' : 'none';
  if (!show) return;
  if (sel.options.length !== nwin + 1) {
    sel.innerHTML = '';
    const optAll = document.createElement('option');
    optAll.value = 'all'; optAll.textContent = 'All';
    sel.appendChild(optAll);
    for (let w = 0; w < nwin; w++) {
      const o = document.createElement('option');
      o.value = String(w);
      o.textContent = `W${w + 1} (${windows.window_start[w].toFixed(0)}–${windows.window_end[w].toFixed(0)}ns)`;
      sel.appendChild(o);
    }
  }
  sel.value = String(windowFilter);
}

// Reflect field-dependent control eligibility. Disables the log-scale
// toggle for the bounded-ratio fields (β and Cherenkov fraction).
function syncFieldDependentControls() {
  const chk = $('logChk');
  if (!chk) return;
  if (curField === 'beta' || curField === 'cher_frac') {
    chk.disabled = true;
    chk.parentElement.style.opacity = '0.4';
    chk.parentElement.title = (curField === 'beta')
      ? 'Log scale not applicable to β'
      : 'Log scale not applicable to Cherenkov fraction';
  } else {
    chk.disabled = false;
    chk.parentElement.style.opacity = '';
    chk.parentElement.title = '';
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
    // Dataset-level emission introspection. Latch on (OR-accumulate): once
    // we've seen an event with both Cherenkov and scintillation rows,
    // expose the EMISSION dropdown for the rest of the session even if
    // later events happen to have only one process represented (rare edge
    // case where all of one process QE-failed on a given event).
    datasetHasBothProcesses = datasetHasBothProcesses || detectDualEmission(d);
    datasetHasDark = datasetHasDark || detectDark(d);
    // Trigger windows (triggered datasets only). Clamp a stale selection when
    // navigating to an event with fewer windows.
    windows = (d.labl && d.labl.per_window) || null;
    const _nwin = windows && windows.window_start ? windows.window_start.length : 0;
    if (windowFilter !== 'all' && windowFilter >= _nwin) windowFilter = 'all';
    syncEmissionUI();
    deriveSensorArrays();   // sets maxHits (+ clamps a stale hitFilter)
    syncHitUI();
    syncWindowUI();
    buildHitsLookups();
    buildSegmentLookups();
    deriveBetaProjection();
    refreshUnionQMap();   // needs pmtT + edep times; before buildPMTs/buildEdeps
    buildPMTs();
    buildEdeps();
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
    if (outlineMat && outlineMat.resolution) {
      outlineMat.resolution.set(el3.clientWidth, el3.clientHeight);
    }
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

  // VIEW toggle (PMTs / EDEP).
  $('viewGrp').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    curView = b.dataset.v;
    for (const c of $('viewGrp').children) c.classList.toggle('active', c === b);
    if (pmtMesh) pmtMesh.visible = (curView === 'pmts');
    if (edepMesh) edepMesh.visible = (curView === 'edep');
    applyViewSweepRange();
    updateSweepUI();
  });

  // Field toggle (Charge / Time / Beta / Cher Frac).
  $('fieldGrp').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    // CHER FRAC is only meaningful when both processes are unfiltered. The
    // button is hidden / disabled in those cases, but guard anyway so a
    // programmatic click is a no-op rather than a wrong-looking render.
    if (b.dataset.v === 'cher_frac'
        && (!datasetHasBothProcesses || emissionFilter !== 'all')) {
      return;
    }
    curField = b.dataset.v;
    for (const c of $('fieldGrp').children) c.classList.toggle('active', c === b);
    syncFieldDependentControls();
    updatePMTColors();
    updateEdepColors();
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
    updateEdepColors();
    buildSidebar();
    applyCorrespondence();
    render2D();
  });

  // Emission dropdown (only present when the dataset has both processes;
  // syncEmissionUI hides the controls + locks emissionFilter to 'all' for
  // single-process datasets).
  $('emissionSelect').addEventListener('change', (e) => {
    emissionFilter = e.target.value;
    // CHER FRAC is only meaningful unfiltered (per the design decision —
    // single-process slices have trivial f = 0 or 1 everywhere). Snap the
    // FIELD back to CHARGE if the user filters down while CHER FRAC is on.
    if (curField === 'cher_frac' && emissionFilter !== 'all') {
      curField = 'charge';
      for (const c of $('fieldGrp').children)
        c.classList.toggle('active', c.dataset.v === 'charge');
      syncFieldDependentControls();
    }
    syncEmissionUI();
    applyEmissionFilter();
    // Selection state stays valid (it indexes particles, not sensors), but
    // the per-particle and per-segment contribution maps depend on which
    // emission_process rows were summed in. Rebuild both lookups so the
    // LABEL coloring + sidebar totals match the active slice.
    buildHitsLookups();
    buildSegmentLookups();
    refreshUnionQMap();
    buildPMTs();
    updatePMTColors();
    applyCorrespondence();
    buildSidebar();
    render2D();
  });

  // HIT dropdown (only present when some PMT has >1 digit). Re-derives every
  // per-sensor slice against the selected digit and rebuilds the label + sidebar
  // so FIELD and LABEL stay coherent on the same hit.
  $('hitSelect').addEventListener('change', (e) => {
    hitFilter = parseInt(e.target.value, 10) || 0;
    deriveSensorArrays();
    applyEmissionFilter();
    buildHitsLookups();
    buildSegmentLookups();
    refreshUnionQMap();
    buildPMTs();
    updatePMTColors();
    applyCorrespondence();
    buildSidebar();
    render2D();
  });

  // TRIGGER WINDOW dropdown: restricts the digit set to one readout window;
  // re-derives every slice and rebuilds the HIT options (maxHits is per-window).
  $('windowSelect').addEventListener('change', (e) => {
    windowFilter = e.target.value === 'all' ? 'all' : parseInt(e.target.value, 10);
    deriveSensorArrays();
    syncHitUI();               // window changes the per-sensor digit count
    applyEmissionFilter();
    buildHitsLookups();
    buildSegmentLookups();
    refreshUnionQMap();
    buildPMTs();
    updatePMTColors();
    applyCorrespondence();
    buildSidebar();
    render2D();
  });

  // Correspondence toggle.
  // Time sweep toggle.
  $('sweepBtn').addEventListener('click', () => {
    sweepOn = !sweepOn;
    syncSweepBtn();
    if (pmtMat) pmtMat.uniforms.sweepOn.value = sweepOn ? 1.0 : 0.0;
    if (edepMat) edepMat.uniforms.sweepOn.value = sweepOn ? 1.0 : 0.0;
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
    if (edepMat) edepMat.uniforms.simTime.value = simTime;
    render2D();
  });

  // Reset — everything visible to the user back to defaults.
  $('resetBtn').addEventListener('click', () => {
    // Toolbar state.
    curView = 'pmts'; curField = 'charge'; curLabel = 'none';
    emissionFilter = 'all';
    hitFilter = 0;
    windowFilter = 'all';
    selectedParticle = null; selectedGroup = null;
    sweepOn = false; sweepPlaying = false; quantileScope = 'pmts';
    simTime = 0;
    // Settings-drawer state.
    logScale = true; percMin = 1; percMax = 99;
    manualVmin = null; manualVmax = null;
    cmapName = 'auto';
    pmtSize = 10;
    outlineWidth = 1.0;
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
    $('outlineWidthSlider').value = outlineWidth; $('outlineWidthVal').textContent = outlineWidth.toFixed(1);
    $('showEmptyChk').checked = showEmpty;
    $('showMeshChk').checked = showMesh;
    $('sweepSpeed').value = sweepSpeed; $('sweepSpeedVal').textContent = sweepSpeed.toFixed(1);
    for (const c of $('viewGrp').children) c.classList.toggle('active', c.dataset.v === 'pmts');
    for (const c of $('fieldGrp').children) c.classList.toggle('active', c.dataset.v === 'charge');
    syncEmissionUI();
    // hitFilter/emissionFilter both reset above — re-derive the per-sensor
    // slices for the 'All'/'All-hits' state and rebuild the label lookups.
    if (evtBundle) { deriveSensorArrays(); buildHitsLookups(); buildSegmentLookups(); }
    syncHitUI();
    syncWindowUI();
    syncFieldDependentControls();
    $('rotBtn').classList.toggle('active', autoRotate);
    if (controls) controls.autoRotate = autoRotate;
    syncSweepBtn();
    // Rebuild meshes (quantile may have been on, arrivalT baked into geometry).
    if (evtBundle) { refreshUnionQMap(); buildPMTs(); buildEdeps(); buildOutline(); }
    if (pmtMesh) pmtMesh.visible = true;
    if (edepMesh) edepMesh.visible = false;
    if (pmtMat) pmtMat.uniforms.sweepOn.value = 0.0;
    if (edepMat) edepMat.uniforms.sweepOn.value = 0.0;
    applyViewSweepRange();
    updateSweepUI();
    updatePMTColors();
    updateEdepColors();
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

  $('logChk').addEventListener('change', (e) => { logScale = e.target.checked; updatePMTColors(); updateEdepColors(); render2D(); });
  const updatePerc = () => {
    percMin = parseFloat($('percMin').value);
    percMax = parseFloat($('percMax').value);
    if (percMin >= percMax - 1) { percMin = Math.max(0, percMax - 1); $('percMin').value = percMin; }
    $('percVal').textContent = percMin.toFixed(1) + ' – ' + percMax.toFixed(1);
    updatePMTColors(); updateEdepColors(); render2D();
  };
  $('percMin').addEventListener('input', updatePerc);
  $('percMax').addEventListener('input', updatePerc);
  $('vminInput').addEventListener('change', (e) => { manualVmin = e.target.value.trim() === '' ? null : parseFloat(e.target.value); updatePMTColors(); updateEdepColors(); render2D(); });
  $('vmaxInput').addEventListener('change', (e) => { manualVmax = e.target.value.trim() === '' ? null : parseFloat(e.target.value); updatePMTColors(); updateEdepColors(); render2D(); });
  $('cmapSelect').addEventListener('change', (e) => { cmapName = e.target.value; updatePMTColors(); updateEdepColors(); render2D(); });

  $('pmtSizeSlider').addEventListener('input', (e) => {
    pmtSize = parseFloat(e.target.value);
    $('pmtSizeVal').textContent = pmtSize.toFixed(1);
    if (pmtMat) pmtMat.uniforms.pmtSize.value = pmtSize;
    if (edepMat) edepMat.uniforms.pmtSize.value = Math.max(3, pmtSize * 0.6);
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
  $('outlineWidthSlider').addEventListener('input', (e) => {
    outlineWidth = parseFloat(e.target.value);
    $('outlineWidthVal').textContent = outlineWidth.toFixed(1);
    if (outlineMat) { outlineMat.linewidth = outlineWidth; outlineMat.needsUpdate = true; }
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
      buildEdeps();
      updatePMTColors();
      updateEdepColors();
      applyCorrespondence();
      applyViewSweepRange();
      updateSweepUI();
      render2D();
    }
    const label = {off: 'off', pmts: 'PMTs only', edep: 'segments only', both: 'PMTs + segments'}[quantileScope];
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
    detectorMaterial = cfg.material || 'water';
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
    syncFieldDependentControls();

    await loadEvent(0);
    animate();
  } catch (e) {
    ls.textContent = 'Error: ' + e.message;
    console.error(e);
  }
})();
