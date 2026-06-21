// Geometry-specific 2D unwrapping for the LUCiD viewer.
//
// Client-side compute — takes sensor_positions (from sensor.h5/config) plus
// detector shape metadata (from step.h5/config) and produces a layout that
// the 2D panel renderer can use unchanged for any geometry.
//
// Convention: output (u, v) are in *layout* space, where u is horizontal
// (x-right) and v is vertical with **v-down** (so larger v = further down
// the screen). The renderer places a circle at screen (offX + u·scale,
// offY + v·scale). Every panel's rect.(x, y) is its top-left in the same
// coordinate system.

// ── Cylinder ───────────────────────────────────────────────────────────
// Matches good_notebooks/cylinder_2D_displays.ipynb cell 3 conceptually:
// top cap on top, barrel in the middle, bottom cap on bottom. Bottom cap
// is mirrored (as the notebook does), so a sensor at +y_world on the
// bottom cap renders at the bottom of its subpanel.
export function layoutCylinder(positions, nSensors, r, halfH) {
  const u = new Float32Array(nSensors);
  const v = new Float32Array(nSensors);
  const panel = new Int16Array(nSensors);
  const case_ = new Int16Array(nSensors);

  const stripH = 2 * halfH;
  const capW = 2 * r, capH = 2 * r;
  const gap = 0.05 * halfH;

  const topY    = 0;
  const barrelY = capH + gap;
  const botY    = barrelY + stripH + gap;
  const layoutH = botY + capH;
  const capX    = Math.PI * r - r;     // cap centered on the barrel's midpoint

  for (let i = 0; i < nSensors; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    // Classify by nearest surface so a barrel ring near the top doesn't
    // get misread as a cap ring. Cap sensors sit on the disc with rho < r;
    // barrel sensors sit on the wall with rho ≈ r — using only |z| pulls
    // the topmost barrel rows onto the cap and renders them on top of the
    // real outer cap ring.
    const rho = Math.sqrt(x * x + y * y);
    const dBarrel = Math.abs(rho - r);
    const dTopCap = Math.abs(z - halfH);
    const dBotCap = Math.abs(z + halfH);
    const dMin = Math.min(dBarrel, dTopCap, dBotCap);
    if (dMin === dTopCap) {
      // Top cap: physical +y = up on screen, so v = (r - y), offset by topY.
      case_[i] = 1; panel[i] = 1;
      u[i] = (x + r) + capX;      // x_world + r so it sits in rect.x = capX..capX+capW
      v[i] = (r - y) + topY;
    } else if (dMin === dBotCap) {
      // Bottom cap: mirrored. Physical -y_world = up on screen.
      case_[i] = 2; panel[i] = 2;
      u[i] = (x + r) + capX;
      v[i] = (y + r) + botY;
    } else {
      // Barrel: matches the notebook's u = ((θ + 3π/2) mod 2π) · r.
      // v = halfH - z so z=+halfH (physical top) renders at v=0.
      case_[i] = 0; panel[i] = 0;
      const theta = Math.atan2(y, x);
      const thetaNorm = ((theta + 1.5 * Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
      u[i] = thetaNorm * r;
      v[i] = (halfH - z) + barrelY;
    }
  }

  const panels = [
    { id: 0, label: 'Barrel',     labelAnchor: 'top',  width: 2 * Math.PI * r, height: stripH,
      rect: { x: 0,    y: barrelY, w: 2 * Math.PI * r, h: stripH } },
    { id: 1, label: 'Top Cap',    labelAnchor: 'left', width: capW, height: capH,
      rect: { x: capX, y: topY,    w: capW, h: capH } },
    { id: 2, label: 'Bottom Cap', labelAnchor: 'left', width: capW, height: capH,
      rect: { x: capX, y: botY,    w: capW, h: capH } },
  ];

  const pmtPitch = computePitches(panel, panels.length, [
    2 * Math.PI * r * (2 * halfH),
    Math.PI * r * r,
    Math.PI * r * r,
  ], nSensors);

  return { panels, u, v, panel, case_, layoutW: 2 * Math.PI * r, layoutH, pmtPitch, seams: [] };
}

// ── Box ────────────────────────────────────────────────────────────────
// Four side faces unrolled as a strip (back → right → front → left,
// CCW walking around the box), plus top/bottom cap rectangles. Cap
// placement matches the cylinder T-shape; the bottom cap is mirrored.
export function layoutBox(positions, nSensors, L, W, H) {
  const halfL = L * 0.5, halfW = W * 0.5, halfH = H * 0.5;
  const u = new Float32Array(nSensors);
  const v = new Float32Array(nSensors);
  const panel = new Int16Array(nSensors);
  const case_ = new Int16Array(nSensors);

  const perim = 2 * L + 2 * W;
  const stripH = H;
  const capW = L, capH = W;
  const gap = 0.05 * Math.max(halfH, halfW);
  const topY    = 0;
  const stripY  = capH + gap;
  const botY    = stripY + stripH + gap;
  const layoutH = botY + capH;
  const capX = (L + W) - L / 2;   // cap centered on strip midpoint (u = perim/2 = L+W)

  for (let i = 0; i < nSensors; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const dFront  = Math.abs(y - halfW);
    const dBack   = Math.abs(y + halfW);
    const dLeft   = Math.abs(x + halfL);
    const dRight  = Math.abs(x - halfL);
    const dTop    = Math.abs(z - halfH);
    const dBottom = Math.abs(z + halfH);
    let minD = dFront, face = 0;
    if (dBack < minD)   { minD = dBack; face = 1; }
    if (dLeft < minD)   { minD = dLeft; face = 2; }
    if (dRight < minD)  { minD = dRight; face = 3; }
    if (dTop < minD)    { minD = dTop; face = 4; }
    if (dBottom < minD) { minD = dBottom; face = 5; }
    case_[i] = face;

    // Strip: v = (halfH - z) + stripY so physical-top maps to top of rect.
    if (face === 1)       { u[i] = x + halfL;                    v[i] = (halfH - z) + stripY; panel[i] = 0; }
    else if (face === 3)  { u[i] = L + (y + halfW);              v[i] = (halfH - z) + stripY; panel[i] = 0; }
    else if (face === 0)  { u[i] = L + W + (halfL - x);          v[i] = (halfH - z) + stripY; panel[i] = 0; }
    else if (face === 2)  { u[i] = 2 * L + W + (halfW - y);      v[i] = (halfH - z) + stripY; panel[i] = 0; }
    else if (face === 4)  { u[i] = (x + halfL) + capX;           v[i] = (halfW - y) + topY;   panel[i] = 1; }
    else                  { u[i] = (x + halfL) + capX;           v[i] = (y + halfW) + botY;   panel[i] = 2; }
  }

  const panels = [
    { id: 0, label: 'Sides',       labelAnchor: 'top',  width: perim, height: stripH,
      rect: { x: 0,    y: stripY, w: perim, h: stripH } },
    { id: 1, label: 'Top Cap',     labelAnchor: 'left', width: capW,  height: capH,
      rect: { x: capX, y: topY,   w: capW,  h: capH } },
    { id: 2, label: 'Bottom Cap',  labelAnchor: 'left', width: capW,  height: capH,
      rect: { x: capX, y: botY,   w: capW,  h: capH } },
  ];

  const seams = [
    { panel: 0, u: L,          label: 'back|right' },
    { panel: 0, u: L + W,      label: 'right|front' },
    { panel: 0, u: 2 * L + W,  label: 'front|left' },
  ];

  const pmtPitch = computePitches(panel, panels.length, [
    perim * H,
    L * W,
    L * W,
  ], nSensors);

  return { panels, u, v, panel, case_, layoutW: perim, layoutH, pmtPitch, seams };
}

// ── Sphere — equirectangular (single panel) ────────────────────────────
export function layoutSphere(positions, nSensors, r) {
  const u = new Float32Array(nSensors);
  const v = new Float32Array(nSensors);
  const panel = new Int16Array(nSensors);      // always 0
  const case_ = new Int16Array(nSensors);

  for (let i = 0; i < nSensors; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const theta = Math.atan2(y, x);
    const phi   = Math.asin(Math.max(-1, Math.min(1, z / r)));
    u[i] = (theta + Math.PI) * r;                 // [0, 2πr]
    v[i] = (Math.PI / 2 - phi) * r;               // [0, πr] with north pole at top
    case_[i] = z >= 0 ? 0 : 1;
  }

  const panels = [
    { id: 0, label: 'Equirectangular', labelAnchor: 'top', width: 2 * Math.PI * r, height: Math.PI * r,
      rect: { x: 0, y: 0, w: 2 * Math.PI * r, h: Math.PI * r } },
  ];

  const pmtPitch = computePitches(panel, 1, [4 * Math.PI * r * r], nSensors);

  return { panels, u, v, panel, case_,
           layoutW: 2 * Math.PI * r, layoutH: Math.PI * r, pmtPitch, seams: [] };
}

// ── Dispatch ───────────────────────────────────────────────────────────

export function computeLayout(detectorType, positions, nSensors, shape) {
  const t = (detectorType || '').toLowerCase();
  if (t === 'cylinder') return layoutCylinder(positions, nSensors, shape.r, shape.halfH);
  if (t === 'box')      return layoutBox(positions, nSensors, shape.L, shape.W, shape.H);
  if (t === 'sphere')   return layoutSphere(positions, nSensors, shape.r);
  throw new Error('Unsupported detector_type: ' + detectorType);
}

// ── Per-panel disc pitch ───────────────────────────────────────────────
// Analytical estimate — sqrt(panel_area / nSensorsInPanel) divided by a
// small safety factor so circles don't quite touch. This matches the
// intent of the notebook's "min pdist" but is O(n) rather than O(n²).
function computePitches(panel, nPanels, areas, nSensors) {
  const counts = new Int32Array(nPanels);
  for (let i = 0; i < nSensors; i++) counts[panel[i]]++;
  const out = new Float32Array(nPanels);
  const SAFETY = 1.05;
  for (let p = 0; p < nPanels; p++) {
    const n = counts[p];
    if (n <= 1) { out[p] = Math.sqrt(areas[p] || 1) * 0.02; continue; }
    out[p] = Math.sqrt(areas[p] / n) / SAFETY;
  }
  return out;
}
