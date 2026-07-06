// LUCiD HDF5 worker.
//
// Streams four v3 files (sensor, hits, step, labl) via h5wasm + HTTP Range,
// and decodes per-event bundles for the main viewer thread.
//
// Schema reference: docs/LUCID_DATASET.md (v3, post-migration).

import h5wasm from 'https://cdn.jsdelivr.net/npm/h5wasm@0.10.1/dist/esm/hdf5_hl.js';

let mod;
let sensorF, hitsF, edepF, lablF;
let nEvents = 0, nSensors = 0;
let detectorType = '';
let sensorPositions = null;        // Float32Array(nSensors * 3)
let shape = {};                    // { r, halfH } or { L, W, H } or { r }
let sourceEventIdxPerFile = {};    // { sensor: Uint32Array, hits: ..., edep: ..., labl: ... }

// ── Attr / dataset helpers ──────────────────────────────────────────────

function readAttr(grp, name) {
  if (!grp) return undefined;
  const a = grp.attrs[name];
  if (!a) return undefined;
  const v = a.value;
  if (typeof v === 'bigint') return Number(v);
  if (v instanceof Uint8Array || v instanceof Int8Array) {
    // Treat single-byte arrays as strings if printable.
    try { return new TextDecoder().decode(v); } catch { return v; }
  }
  return v;
}

function readString(grp, name) {
  const v = readAttr(grp, name);
  if (typeof v === 'string') return v;
  if (Array.isArray(v) && typeof v[0] === 'string') return v[0];
  return v;
}

// Always copy (detach-safe for transferables).
function readDsFloat32(grp, name) {
  const d = grp.get(name); if (!d) return null;
  return new Float32Array(d.value);
}
function readDsInt32(grp, name) {
  const d = grp.get(name); if (!d) return null;
  return new Int32Array(d.value);
}
function readDsUint16(grp, name) {
  const d = grp.get(name); if (!d) return null;
  return new Uint16Array(d.value);
}
function readDsUint32(grp, name) {
  const d = grp.get(name); if (!d) return null;
  return new Uint32Array(d.value);
}
function readDsUint8(grp, name) {
  const d = grp.get(name); if (!d) return null;
  return new Uint8Array(d.value);
}
function readDsInt16(grp, name) {
  const d = grp.get(name); if (!d) return null;
  return new Int16Array(d.value);
}
function readDsInt8(grp, name) {
  const d = grp.get(name); if (!d) return null;
  return new Int8Array(d.value);
}
// HDF5 bool → Uint8Array (0/1). Callers use !!arr[i] to consume as JS bool.
function readDsBool(grp, name) {
  const d = grp.get(name); if (!d) return null;
  const v = d.value;
  if (v instanceof Uint8Array) return v;
  // h5wasm may surface bool as a normal Array of booleans / numbers.
  const out = new Uint8Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] ? 1 : 0;
  return out;
}
function readScalarBool(grp, name) {
  const ds = grp.get(name);
  if (!ds) return false;
  const v = ds.value;
  if (typeof v === 'boolean') return v;
  if (typeof v === 'number') return !!v;
  if (v && v.length !== undefined) return !!v[0];
  return !!v;
}

// ── Mount HDF5 files in memory ─────────────────────────────────────────
// Production files are small (≤ few MB each). Fetching them whole into a
// Uint8Array and serving HDF5 reads from that buffer is dramatically faster
// than the sync-XHR-per-chunk pattern: no per-read network round-trip and
// no cache thrash from HDF5's non-sequential b-tree traversal.

async function mountUrl(url, wasmName) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  const node = mod.FS.createFile('/', wasmName, {}, true, false);
  node.usedBytes = buf.length;
  Object.defineProperty(node, 'size', { get: () => buf.length });
  node.stream_ops = {
    read(s, dst, off, len, pos) {
      const n = Math.max(0, Math.min(len, buf.length - pos));
      for (let i = 0; i < n; i++) dst[off + i] = buf[pos + i];
      return n;
    },
    llseek(s, off, w) { return w === 1 ? s.position + off : w === 2 ? buf.length + off : off; },
  };
}

// ── Event decoding ──────────────────────────────────────────────────────

function eventKey(idx) { return 'event_' + String(idx).padStart(3, '0'); }

function decodeEvent(idx) {
  const k = eventKey(idx);

  const sEvt = sensorF.get(k);
  const hEvt = hitsF.get(k);
  const eEvt = edepF.get(k);
  const lEvt = lablF.get(k);
  if (!sEvt || !hEvt || !eEvt || !lEvt) {
    throw new Error(`event ${idx} missing from one or more files`);
  }

  // Cross-file sanity check: all four should agree on source_event_idx.
  const srcIdxS = readAttr(sEvt, 'source_event_idx');
  const srcIdxH = readAttr(hEvt, 'source_event_idx');
  const srcIdxE = readAttr(eEvt, 'source_event_idx');
  const srcIdxL = readAttr(lEvt, 'source_event_idx');
  let warning = null;
  if (srcIdxS !== undefined && srcIdxH !== undefined &&
      srcIdxE !== undefined && srcIdxL !== undefined) {
    if (!(srcIdxS === srcIdxH && srcIdxH === srcIdxE && srcIdxE === srcIdxL)) {
      warning = `source_event_idx mismatch: sensor=${srcIdxS} hits=${srcIdxH} step=${srcIdxE} labl=${srcIdxL}`;
    }
  }

  // Sensor file.
  const sensorSIdx = readDsUint16(sEvt, 'sensor_idx');
  const sensorPE = readDsFloat32(sEvt, 'PE');
  const sensorT = readDsFloat32(sEvt, 'T');

  // Hits file.
  const hitsParticle = readDsInt32(hEvt, 'particle_idx');
  const hitsSIdx = readDsUint16(hEvt, 'sensor_idx');
  const hitsPE = readDsFloat32(hEvt, 'PE');
  const hitsT = readDsFloat32(hEvt, 'T');
  // digit_idx: FK into sensor.h5's per-event digit list (which recorded hit
  // each decomposition row belongs to). Absent on pre-digitizer datasets —
  // synthesize all-zeros (one digit per sensor) so the HIT slice degrades to
  // the single-hit case.
  let hitsDigit = readDsInt32(hEvt, 'digit_idx');
  if (!hitsDigit && hitsSIdx) hitsDigit = new Int32Array(hitsSIdx.length);
  // emission_process column: per-row int8 tag (0=Cherenkov, 1=scintillation).
  // Added in the LUCiD Phase 0 schema delta — readers default to all-zeros on
  // pre-change datasets via the Python reader, but the worker has its own
  // h5wasm path so we synthesize the same fallback here when the column is
  // absent (Cherenkov-only legacy datasets).
  let hitsEmission = readDsInt8(hEvt, 'emission_process');
  if (!hitsEmission && hitsSIdx) hitsEmission = new Int8Array(hitsSIdx.length);
  const nParticles = readAttr(hEvt, 'n_particles') || 0;

  // Step file.
  const edepTrackIdx = readDsInt32(eEvt, 'track_idx');
  const edepStartX = readDsFloat32(eEvt, 'start_x');
  const edepStartY = readDsFloat32(eEvt, 'start_y');
  const edepStartZ = readDsFloat32(eEvt, 'start_z');
  const edepEndX = readDsFloat32(eEvt, 'end_x');
  const edepEndY = readDsFloat32(eEvt, 'end_y');
  const edepEndZ = readDsFloat32(eEvt, 'end_z');
  const edepTime = readDsFloat32(eEvt, 'time');
  const edepEdep = readDsFloat32(eEvt, 'edep');
  const edepBeta = readDsFloat32(eEvt, 'beta_start');
  const edepNCh = readDsInt32(eEvt, 'n_cherenkov');
  const nTracksEdep = readAttr(eEvt, 'n_tracks') || 0;
  const nSegments = readAttr(eEvt, 'n_segments') || (edepEdep ? edepEdep.length : 0);

  // Optional sensor_hits subgroup (present when LUCiD ran with
  // store_segment_sensor_map=true). Flat parallel arrays — one row per
  // (segment, sensor) pair — used to colour PMTs by the dominant segment
  // and to drive segment-row selection in the sidebar.
  let edepSensorHits = null;
  const shGrp = eEvt.get('sensor_hits');
  if (shGrp) {
    const shSIdx = readDsUint16(shGrp, 'sensor_idx');
    let shEmission = readDsInt8(shGrp, 'emission_process');
    if (!shEmission && shSIdx) shEmission = new Int8Array(shSIdx.length);
    edepSensorHits = {
      segment_idx:      readDsInt32(shGrp,   'segment_idx'),
      sensor_idx:       shSIdx,
      PE:               readDsFloat32(shGrp, 'PE'),
      T:                readDsFloat32(shGrp, 'T'),
      emission_process: shEmission,
    };
  }

  // Labl file.
  const perEvtGrp = lEvt.get('per_event');
  const perIntGrp = lEvt.get('per_interaction');
  const perPartGrp = lEvt.get('per_particle');
  const perTrkGrp = lEvt.get('per_track');

  const t0 = perEvtGrp ? readAttrOrScalar(perEvtGrp, 't0') : 0;
  const contained = perEvtGrp ? readScalarBool(perEvtGrp, 'contained') : false;

  // per_window (triggered datasets only): readout gates + CSR digit_offsets
  // into sensor.h5 (window w = sensor digit rows [off[w], off[w+1])).
  const perWinGrp = lEvt.get('per_window');
  let perWindow = null;
  if (perWinGrp) {
    perWindow = {
      window_start:  readDsFloat32(perWinGrp, 'window_start'),
      window_end:    readDsFloat32(perWinGrp, 'window_end'),
      digit_offsets: readDsInt32(perWinGrp,   'digit_offsets'),
    };
  }

  // v5 per_interaction: one row per source interaction (one G4 event
  // worth of primaries + their descendants). CSR primary_{track_ids,
  // pdgs, energies} lists carry each interaction's primary particles.
  let perInteraction = null;
  if (perIntGrp) {
    perInteraction = {
      source_type:             readDsUint8(perIntGrp,   'source_type'),
      t0:                      readDsFloat32(perIntGrp, 't0'),
      vertex_x:                readDsFloat32(perIntGrp, 'vertex_x'),
      vertex_y:                readDsFloat32(perIntGrp, 'vertex_y'),
      vertex_z:                readDsFloat32(perIntGrp, 'vertex_z'),
      n_primaries:             readDsInt32(perIntGrp,   'n_primaries'),
      n_particles:             readDsInt32(perIntGrp,   'n_particles'),
      neutrino_pdg:            readDsInt16(perIntGrp,   'neutrino_pdg'),
      neutrino_energy_MeV:     readDsFloat32(perIntGrp, 'neutrino_energy_MeV'),
      contained:               readDsBool(perIntGrp,    'contained'),
      primary_track_ids_offsets: readDsUint32(perIntGrp, 'primary_track_ids_offsets'),
      primary_track_ids_data:    readDsInt32(perIntGrp,  'primary_track_ids_data'),
      primary_pdgs_offsets:      readDsUint32(perIntGrp, 'primary_pdgs_offsets'),
      primary_pdgs_data:         readDsInt16(perIntGrp,  'primary_pdgs_data'),
      primary_energies_offsets:  readDsUint32(perIntGrp, 'primary_energies_offsets'),
      primary_energies_data:     readDsFloat32(perIntGrp, 'primary_energies_data'),
    };
  }

  let containedPerParticle = null, genealogy = null, genealogyOffsets = null;
  if (perPartGrp) {
    containedPerParticle = readDsBool(perPartGrp, 'contained');
    genealogy = readDsInt32(perPartGrp, 'genealogy_data');
    genealogyOffsets = readDsUint32(perPartGrp, 'genealogy_offsets');
  }

  let trackId = null, trackPdg = null, trackParticleIdx = null, trackInitE = null, trackNCh = null;
  let trackInteraction = null;
  if (perTrkGrp) {
    trackId = readDsInt32(perTrkGrp, 'track_id');
    trackPdg = readDsInt16(perTrkGrp, 'pdg');
    trackParticleIdx = readDsInt32(perTrkGrp, 'particle_idx');
    trackInitE = readDsFloat32(perTrkGrp, 'initial_energy');
    trackNCh = readDsInt32(perTrkGrp, 'n_cherenkov');
    trackInteraction = readDsInt32(perTrkGrp, 'interaction');
  }

  const n_particles = nParticles || (genealogyOffsets ? Math.max(0, genealogyOffsets.length - 1) : 0);
  const n_tracks = (trackPdg ? trackPdg.length : 0) || nTracksEdep;

  const edepContained = readDsBool(eEvt, 'contained');
  return {
    warning,
    srcIdx: srcIdxS,
    t0,
    contained,
    sensor: { sensor_idx: sensorSIdx, PE: sensorPE, T: sensorT,
              nHits: sensorSIdx ? sensorSIdx.length : 0 },
    hits: { particle_idx: hitsParticle, sensor_idx: hitsSIdx, PE: hitsPE, T: hitsT,
            emission_process: hitsEmission, digit_idx: hitsDigit,
            nHits: hitsSIdx ? hitsSIdx.length : 0 },
    edep: { track_idx: edepTrackIdx,
           start_x: edepStartX, start_y: edepStartY, start_z: edepStartZ,
           end_x: edepEndX, end_y: edepEndY, end_z: edepEndZ,
           time: edepTime, edep: edepEdep, beta_start: edepBeta, n_cherenkov: edepNCh,
           contained: edepContained,
           sensor_hits: edepSensorHits,
           n: nSegments },
    labl: { n_particles, n_tracks,
            per_window: perWindow,
            per_interaction: perInteraction,
            per_particle: { contained: containedPerParticle, genealogy, genealogy_offsets: genealogyOffsets },
            per_track: { track_id: trackId, pdg: trackPdg, particle_idx: trackParticleIdx,
                         initial_energy: trackInitE, n_cherenkov: trackNCh,
                         interaction: trackInteraction } },
  };
}

function readAttrOrScalar(grp, name) {
  // t0 / contained are scalar datasets per schema, not attrs.
  const ds = grp.get(name);
  if (ds) {
    const v = ds.value;
    if (typeof v === 'number') return v;
    if (v && v.length !== undefined) return v[0];
    return v;
  }
  // Fallback: some older writers may store as attr.
  const a = readAttr(grp, name);
  return typeof a === 'number' ? a : (Array.isArray(a) ? a[0] : 0);
}

// ── Transferables (avoid copies when sending to main) ──────────────────

function collectTransfers(obj, seen) {
  if (!seen) seen = new Set();
  const t = [];
  (function walk(o) {
    if (!o) return;
    if (o instanceof ArrayBuffer) {
      if (!seen.has(o)) { seen.add(o); t.push(o); }
    } else if (ArrayBuffer.isView(o)) {
      if (!seen.has(o.buffer)) { seen.add(o.buffer); t.push(o.buffer); }
    } else if (typeof o === 'object') {
      for (const v of Object.values(o)) walk(v);
    }
  })(obj);
  return t;
}

// ── Message dispatch ────────────────────────────────────────────────────

self.onerror = (err) => {
  console.error('[h5_worker] onerror:', err);
  self.postMessage({ action: 'error', message: 'worker onerror: ' + (err.message || err) });
};

console.log('[h5_worker] module loaded');

self.onmessage = async function (e) {
  const { action } = e.data;
  console.log('[h5_worker] action:', action);
  try {
    if (action === 'init') {
      console.log('[h5_worker] awaiting h5wasm.ready...');
      mod = await h5wasm.ready;
      console.log('[h5_worker] h5wasm ready, fetching files...');
      const base = e.data.base;
      const manifest = e.data.manifest;
      // Prefetch all four files in parallel — 5 MB total for the sample.
      await Promise.all([
        mountUrl(base + '/' + manifest.sensor, 'sensor.h5'),
        mountUrl(base + '/' + manifest.hits,   'hits.h5'),
        mountUrl(base + '/' + manifest.step,    'edep.h5'),
        mountUrl(base + '/' + manifest.labl,   'labl.h5'),
      ]);
      console.log('[h5_worker] files fetched, opening HDF5');
      sensorF = new h5wasm.File('/sensor.h5', 'r');
      hitsF = new h5wasm.File('/hits.h5', 'r');
      edepF = new h5wasm.File('/edep.h5', 'r');
      lablF = new h5wasm.File('/labl.h5', 'r');

      const sCfg = sensorF.get('config');
      const gCfg = edepF.get('config');
      nEvents = readAttr(sCfg, 'n_events') || 0;
      nSensors = readAttr(sCfg, 'n_sensors') || 0;
      detectorType = readString(sCfg, 'detector_type') || '';
      const posDs = sCfg.get('sensor_positions');
      sensorPositions = posDs ? new Float32Array(posDs.value) : null;
      if (!nSensors && sensorPositions) nSensors = Math.floor(sensorPositions.length / 3);

      // Geometry shape params (step/config; sensor/config usually lacks them).
      if (gCfg) {
        const shapeAttr = readString(gCfg, 'detector_shape') ||
                          readString(gCfg, 'detector_type') || detectorType;
        const dR = readAttr(gCfg, 'detector_radius');
        const dHH = readAttr(gCfg, 'detector_half_height');
        const bboxDs = gCfg.get('detector_bbox');
        const bbox = bboxDs ? new Float32Array(bboxDs.value) : null;
        if ((shapeAttr || detectorType || '').toLowerCase().includes('cyl')) {
          shape = { r: dR, halfH: dHH };
        } else if ((shapeAttr || detectorType || '').toLowerCase().includes('sphere')) {
          shape = { r: dR };
        } else if ((shapeAttr || detectorType || '').toLowerCase().includes('box')) {
          if (bbox) shape = { L: bbox[3] - bbox[0], W: bbox[4] - bbox[1], H: bbox[5] - bbox[2] };
          else shape = { L: 2 * dR, W: 2 * dR, H: 2 * dHH };
        }
      }
      // Fallback — infer from sensor_positions extent if shape params absent.
      if ((!shape.r && !shape.L) && sensorPositions) {
        let xmn=Infinity,xmx=-Infinity,ymn=Infinity,ymx=-Infinity,zmn=Infinity,zmx=-Infinity;
        for (let i = 0; i < nSensors; i++) {
          const x = sensorPositions[i*3], y = sensorPositions[i*3+1], z = sensorPositions[i*3+2];
          if (x<xmn)xmn=x; if (x>xmx)xmx=x;
          if (y<ymn)ymn=y; if (y>ymx)ymx=y;
          if (z<zmn)zmn=z; if (z>zmx)zmx=z;
        }
        const t = (detectorType || '').toLowerCase();
        if (t === 'cylinder') shape = { r: Math.max(xmx, ymx), halfH: zmx };
        else if (t === 'sphere') shape = { r: Math.max(xmx, ymx, zmx) };
        else if (t === 'box') shape = { L: xmx - xmn, W: ymx - ymn, H: zmx - zmn };
      }

      // source_event_idx arrays per file (for the integrity check on event load).
      for (const [key, f] of [['sensor', sensorF], ['hits', hitsF],
                              ['edep', edepF], ['labl', lablF]]) {
        const ds = f.get('config/source_event_idx');
        if (ds) sourceEventIdxPerFile[key] = new Uint32Array(ds.value);
      }

      // Provenance snippet.
      const provenance = {
        dataset_name: readString(sCfg, 'dataset_name'),
        run_id: readString(sCfg, 'run_id'),
        git_commit: readString(sCfg, 'git_commit'),
        source_file: readString(sCfg, 'source_file'),
        format_version: readAttr(sCfg, 'format_version'),
      };
      // Material name (used by the viewer to derive the Cherenkov β
      // threshold for the BETA field). Stored on step/config in v5; fall
      // back to sensor/config; default 'water' if absent.
      const material = readString(gCfg, 'material')
                    || readString(sCfg, 'material')
                    || 'water';

      // Send a copy of sensor positions so we retain the local reference.
      const posCopy = sensorPositions ? new Float32Array(sensorPositions) : null;
      self.postMessage({
        action: 'ready',
        nEvents, nSensors,
        detectorType,
        shape,
        material,
        sensorPositions: posCopy,
        provenance,
      }, posCopy ? [posCopy.buffer] : []);
    } else if (action === 'loadEvent') {
      const d = decodeEvent(e.data.idx);
      self.postMessage({ action: 'eventLoaded', idx: e.data.idx, ...d },
                       collectTransfers(d));
    }
  } catch (err) {
    self.postMessage({ action: 'error', message: err.message, stack: err.stack });
  }
};
