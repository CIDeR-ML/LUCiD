// GLSL shaders for the LUCiD 3D renderer.
//
// Single point-sprite material PMT_VS / PMT_FS — used for both the PMT
// discs and for segment-trajectory points (K dots along each segment).
// WebGL's 1px line limit made LineSegments invisible against the PMT
// cloud; a point cloud sidesteps that at the cost of a few kB of geometry.
//
// Color mode is unified:
//   colorMode = 0  → continuous (sample cmap texture at `contVal`)
//   colorMode = 1  → categorical (compute HSL from `catVal` hue)
//
// Time sweep animation: when `sweepOn = 1`, alpha fades from 0 → 1 as
// `simTime` crosses each primitive's `arrivalT`.
//
// Correspondence: when `corrOn = 1`, non-highlighted primitives are dimmed.

export const PMT_VS = `
attribute float contVal;
attribute float catVal;
attribute float hl;
attribute float arrivalT;
attribute float hasSignal;

uniform float colorMode;   // 0 = continuous, 1 = categorical
uniform float corrOn;      // 1 = isolate highlighted
uniform float sweepOn;     // 1 = time sweep active
uniform float simTime;
uniform float sweepEps;    // fade width around arrivalT
uniform float pmtSize;     // base disc size in screen pixels
uniform float emptyGray;   // 0 = hide empty PMTs, 1 = show as gray

varying float vT;
varying float vHL;
varying float vMode;
varying float vEmpty;
varying float vA;

void main() {
  vMode = colorMode;
  vHL = hl;
  vEmpty = (hasSignal < 0.5) ? 1.0 : 0.0;

  // Base alpha. Empty PMTs get the silhouette alpha, signal PMTs full.
  float a = vEmpty > 0.5 ? 0.35 * emptyGray : 0.9;

  // Time-sweep fade.
  if (sweepOn > 0.5 && vEmpty < 0.5) {
    float fade = smoothstep(arrivalT - sweepEps, arrivalT + sweepEps, simTime);
    a *= fade;
  }

  // Correspondence dimming — non-contributors fade hard; contributors pop.
  if (corrOn > 0.5 && vEmpty < 0.5) {
    a *= mix(0.08, 1.0, hl);
  }
  if (corrOn > 0.5 && vEmpty > 0.5) {
    a *= 0.15;
  }

  vT = (vMode > 0.5) ? catVal : contVal;
  vA = a;

  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float sizeBoost = 1.0 + hl * 1.0 * corrOn;
  gl_PointSize = pmtSize * sizeBoost;
}`;

export const PMT_FS = `
uniform sampler2D cmap;
uniform vec3 emptyColor;

varying float vT;
varying float vHL;
varying float vMode;
varying float vEmpty;
varying float vA;

vec3 hsl2rgb(vec3 hsl) {
  float h = hsl.x, s = hsl.y, l = hsl.z;
  vec3 rgb;
  float c = (1.0 - abs(2.0 * l - 1.0)) * s;
  float x = c * (1.0 - abs(mod(h * 6.0, 2.0) - 1.0));
  float m = l - c * 0.5;
  if (h < 1.0/6.0) rgb = vec3(c, x, 0.0);
  else if (h < 2.0/6.0) rgb = vec3(x, c, 0.0);
  else if (h < 3.0/6.0) rgb = vec3(0.0, c, x);
  else if (h < 4.0/6.0) rgb = vec3(0.0, x, c);
  else if (h < 5.0/6.0) rgb = vec3(x, 0.0, c);
  else rgb = vec3(c, 0.0, x);
  return rgb + vec3(m);
}

void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = length(d);
  if (r > 0.5) discard;
  float edge = 1.0 - smoothstep(0.42, 0.5, r);

  vec3 c;
  if (vEmpty > 0.5) {
    c = emptyColor;
  } else if (vMode > 0.5) {
    c = hsl2rgb(vec3(vT, 0.78, 0.55));
  } else {
    c = texture2D(cmap, vec2(clamp(vT, 0.005, 0.995), 0.5)).rgb;
  }
  // Highlight tint when correspondence is active.
  c = mix(c, vec3(1.0), vHL * 0.5);

  gl_FragColor = vec4(c, vA * edge);
}`;
