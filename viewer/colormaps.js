// Colormap definitions for the LUCiD viewer.
//
// LUCiD convention (from good_notebooks/cylinder_2D_displays.ipynb):
//   - Charge (PE)  → plasma
//   - Time         → viridis_r
// Continuous edep colormaps: plasma (edep), viridis_r (time), viridis (β, n_cherenkov).
// Plus a fixed palette for particle categories and a hashed-hue helper for
// generic categorical IDs (particle_idx, track_idx, pdg).

// ── Stops (piecewise-linear) ────────────────────────────────────────────

// Matplotlib plasma (7-stop coarse), dark→bright.
export const PLASMA_STOPS = [
  [0.00, '#0d0887'], [0.17, '#5c02a3'], [0.33, '#9a179b'],
  [0.50, '#cb4679'], [0.67, '#ed7953'], [0.83, '#fbb32f'],
  [1.00, '#f0f921'],
];

// Matplotlib viridis, dark→bright.
export const VIRIDIS_STOPS = [
  [0.00, '#440154'], [0.17, '#482777'], [0.33, '#3f4a8a'],
  [0.50, '#31678e'], [0.67, '#26838f'], [0.83, '#6cce5a'],
  [1.00, '#fde725'],
];

// Reverse of viridis (used for "Time" — bright at early, dark at late).
export const VIRIDIS_R_STOPS = VIRIDIS_STOPS.map(([t, c]) => [1 - t, c]).reverse();

// "Inferno" reversed — kept for optional light-mode charge.
export const INFERNO_R_STOPS = [
  [0.00, '#fcffa4'], [0.20, '#fca50a'], [0.40, '#dd513a'],
  [0.60, '#932667'], [0.80, '#420a68'], [1.00, '#0d0829'],
];

// ── Parsing / interpolation ─────────────────────────────────────────────

export function parseHex(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}

function toRGB(c) { return typeof c === 'string' ? parseHex(c) : c; }

export function interpolate(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      const a = toRGB(c0), b = toRGB(c1);
      return [
        Math.round(a[0] + f * (b[0] - a[0])),
        Math.round(a[1] + f * (b[1] - a[1])),
        Math.round(a[2] + f * (b[2] - a[2])),
      ];
    }
  }
  return toRGB(stops[stops.length - 1][1]);
}

export function plasmaRGB(t)    { return interpolate(PLASMA_STOPS, t); }
export function viridisRGB(t)   { return interpolate(VIRIDIS_STOPS, t); }
export function viridisRRGB(t)  { return interpolate(VIRIDIS_R_STOPS, t); }

// Registry for name-based lookup.
export const NAMED_STOPS = {
  plasma:    PLASMA_STOPS,
  viridis:   VIRIDIS_STOPS,
  viridis_r: VIRIDIS_R_STOPS,
  inferno_r: INFERNO_R_STOPS,
};

// ── HSL → RGB ───────────────────────────────────────────────────────────

export function hsl2rgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h * 6) % 2 - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  const i = Math.floor(h * 6) % 6;
  if (i === 0) { r = c; g = x; }
  else if (i === 1) { r = x; g = c; }
  else if (i === 2) { g = c; b = x; }
  else if (i === 3) { g = x; b = c; }
  else if (i === 4) { r = x; b = c; }
  else { r = c; b = x; }
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

// Golden-ratio hue for hashed categorical colors.
export function hashHue(id) {
  return (Math.abs(id | 0) * 0.618033988749895) % 1.0;
}

// ── Fixed palette for particle categories ───────────────────────────────
// Indices correspond to labl/per_particle/category (uint8). The common LUCiD
// mapping is: 0=Primary, 1=DecayElectron, 2=SecondaryPion, 3=Gamma — though
// this is a data-dependent convention. Keeping it as a list so future
// category schemes can extend.
export const CATEGORY_PALETTE = [
  '#ff9f1c', // Primary       — orange
  '#00b4d8', // DecayElectron — cyan
  '#9d4edd', // SecondaryPion — purple
  '#ffd23f', // Gamma         — yellow
  '#06d6a0', // extra 1       — teal
  '#ef476f', // extra 2       — pink
  '#8d99ae', // extra 3       — gray-blue
  '#3a86ff', // extra 4       — blue
];

export function categoryRGB(cat) {
  const hex = CATEGORY_PALETTE[((cat | 0) % CATEGORY_PALETTE.length + CATEGORY_PALETTE.length) % CATEGORY_PALETTE.length];
  return parseHex(hex);
}
