/**
 * Deformed-lattice showcase engine.
 *
 * The lattice is no longer a selector that paints clip IDs; it is the geometry
 * that gets drawn. Undeformed lattice positions are texture coordinates,
 * deformed positions are vertex positions, and the render is a textured mesh.
 *
 * Ported from two offline probes:
 *   scripts/probe-f-force-field.py   vertices settle onto a glyph distance
 *                                    field under spring constraints (default)
 *   scripts/probe-g-global-warp.py   one harmonic displacement field, plus the
 *                                    Jacobian shading term and the strength
 *                                    schedule, which are taken from G verbatim
 *
 * The deformation is solved ONCE per word and lattice. Per frame the engine
 * evaluates the wobble, adds amplitude times the displacement, derives the
 * shading term from the deformed cell areas, and draws.
 */

export type Algorithm = 'settle' | 'warp';
/** The two correspondence families are kept separate so the late-G reference
 * cannot silently replace the original offset experiment. */
export type WarpCorrespondence = 'offset' | 'late-g-lens';
export type ReleaseShape = 'disperse' | 'rewind' | 'asymmetric';
export type Schedule = 'const' | 'overshoot' | 'inverse' | 'lead' | 'lag';
/** Signed exponent meaning is explicit in the UI and in the shader. The old
 * names remain accepted for saved links made by the first port. */
export type Polarity = 'compression-bright' | 'compression-dark' | 'positive' | 'negative' | 'debossed' | 'embossed';
export type FootageMode = 'single' | 'jumble';
export type IdleField = 'jumble' | 'single';
export type CommitTransition = 'cut' | 'squeeze' | 'push';
export type JumbleShading = 'on' | 'off';
/** What the two commit pools are drawn from. `luminance` is the original
 *  round-4 split and stays the default so the recorded studies still
 *  reproduce; `origin` commits the background to one clip — the project the
 *  word names — and leaves the letters to everything else. `origin-auto` is
 *  the same pooling with the home clip matched from the word instead of
 *  chosen by hand; the match itself happens in the panel, which is the only
 *  layer that knows the clips' names, and arrives here as `homeClip`. */
export type FamilyMode = 'luminance' | 'origin' | 'origin-auto';

export type MeshParams = {
  // word
  text: string;
  fontSize: number;
  fontAuto: boolean;
  offsetX: number;
  offsetY: number;
  drift: boolean;
  driftAmount: number;
  // lattice
  columns: number;
  breath: number;
  breathSpeed: number;
  // deformation
  algorithm: Algorithm;
  warpCorrespondence: WarpCorrespondence;
  gain: number;
  band: number;
  spread: number;
  amplitudeAuto: boolean;
  amplitude: number;
  // attenuation cycle
  idle: number;
  settle: number;
  hold: number;
  release: number;
  releaseShape: ReleaseShape;
  // shading
  strength: number;
  schedule: Schedule;
  flarePeak: number;
  flareCentre: number;
  flareWidth: number;
  polarity: Polarity;
  lumaAdaptive: boolean;
  smoothing: number;
  smoothingPx: number;
  clampIdleTail: boolean;
  jumbleShading: JumbleShading;
  // footage
  footage: FootageMode;
  commit: boolean;
  commitTransition: CommitTransition;
  idleField: IdleField;
  stagger: number;
  clip: number;
  familySplit: boolean;
  familyMode: FamilyMode;
  /** Resolved ground clip for the origin split. The panel sets it from the
   *  Clip select, or from the word when the mode is `origin-auto`. */
  homeClip: number;
  seed: number;
  // view
  view: number;
};

export const MESH_DEFAULTS: MeshParams = {
  text: 'Yope3D',
  fontSize: 240,
  fontAuto: false,
  offsetX: 0,
  offsetY: 0,
  drift: false,
  driftAmount: 60,
  columns: 88,
  breath: 0.3,
  breathSpeed: 1,
  algorithm: 'settle',
  warpCorrespondence: 'offset',
  gain: 1,
  band: 0.85,
  spread: 1,
  amplitudeAuto: true,
  amplitude: 1,
  idle: 2,
  settle: 4.5,
  hold: 3,
  release: 3,
  releaseShape: 'disperse',
  strength: 0.55,
  schedule: 'overshoot',
  flarePeak: 1.7,
  flareCentre: 0.45,
  flareWidth: 0.28,
  polarity: 'compression-dark',
  lumaAdaptive: false,
  smoothing: 2,
  smoothingPx: 3,
  clampIdleTail: false,
  jumbleShading: 'on',
  footage: 'jumble',
  commit: false,
  commitTransition: 'cut',
  idleField: 'jumble',
  stagger: 0.72,
  clip: 0,
  familySplit: true,
  familyMode: 'luminance',
  homeClip: 0,
  seed: 11,
  view: 0,
};

export const CANVAS_W = 1280;
export const CANVAS_H = 720;

/** Return the actual signed exponent sent to the fragment shader. */
export function signedExponent(polarity: Polarity, magnitude: number): number {
  const positive = polarity === 'compression-bright' || polarity === 'positive' || polarity === 'embossed';
  return (positive ? 1 : -1) * Math.abs(magnitude);
}

// Force-field configuration, from probe-f-force-field.py DEFAULT_CFG.
const CFG = {
  kAtt: 6.0,
  kSpring: 0.42,
  kShape: 0.1,
  kHome: 0.012,
  cohPow: 3.0,
  kCurv: 1.0,
  projMin: 0.14,
  projRelax: 0.85,
  projSub: 6,
  stepCap: 0.2,
  dt: 0.25,
  coolFrac: 0.18,
  rampFrac: 0.55,
};

// probe-g-global-warp.py render_word(width_frac=0.80, height_frac=0.32) and its
// 8..900 binary-search bracket, used by the auto font fit.
const FIT_WIDTH = 0.8;
const FIT_HEIGHT = 0.32;
const FIT_MIN_PX = 8;
const FIT_MAX_PX = 900;

const OVERSCAN_CELLS = 3;
const SDF_PAD = 224; // canvas px of padding around the field
const SDF_SCALE = 0.5; // field samples per canvas px
// Jacobian clamp, from probe-g-global-warp.py shade_base().
const DET_LO = 0.4;
const DET_HI = 2.5;

/* ---------------------------------------------------------------------- */
/* 1. glyph raster and signed distance field                               */
/* ---------------------------------------------------------------------- */

const INF = 1e20;

function edt1d(f: Float64Array, d: Float64Array, v: Int32Array, z: Float64Array, n: number) {
  let k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const dist = q - v[k];
    d[q] = dist * dist + f[v[k]];
  }
}

/** Exact Euclidean distance transform (Felzenszwalb & Huttenlocher). */
function edt2d(seed: Uint8Array, w: number, h: number, want: number): Float32Array {
  const grid = new Float64Array(w * h);
  for (let p = 0; p < grid.length; p++) grid[p] = seed[p] === want ? 0 : INF;
  const n = Math.max(w, h);
  const f = new Float64Array(n);
  const d = new Float64Array(n);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = grid[y * w + x];
    edt1d(f, d, v, z, h);
    for (let y = 0; y < h; y++) grid[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) f[x] = grid[row + x];
    edt1d(f, d, v, z, w);
    for (let x = 0; x < w; x++) grid[row + x] = d[x];
  }
  const out = new Float32Array(w * h);
  for (let p = 0; p < out.length; p++) out[p] = Math.sqrt(grid[p]);
  return out;
}

/** Separable [1,2,1]/4 passes; stands in for a small Gaussian. */
function blur3(field: Float32Array, w: number, h: number, passes: number) {
  if (passes <= 0) return field;
  const tmp = new Float32Array(field.length);
  for (let pass = 0; pass < passes; pass++) {
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const a = field[row + Math.max(0, x - 1)];
        const b = field[row + x];
        const c = field[row + Math.min(w - 1, x + 1)];
        tmp[row + x] = (a + 2 * b + c) * 0.25;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const a = tmp[Math.max(0, y - 1) * w + x];
        const b = tmp[y * w + x];
        const c = tmp[Math.min(h - 1, y + 1) * w + x];
        field[y * w + x] = (a + 2 * b + c) * 0.25;
      }
    }
  }
  return field;
}

export type Glyph = {
  w: number;
  h: number;
  sdf: Float32Array; // canvas px, positive outside the ink
  gx: Float32Array; // unit gradient
  gy: Float32Array;
  coh: Float32Array; // |grad|, collapses on medial axes
  curv: Float32Array; // |div of the unit gradient|, 1/px
  strokePx: number;
  fontPx: number;
  /** The size actually rasterised, in canvas px. Equals `fontSize` unless the
   *  auto fit or the overflow squeeze changed it. */
  fontPx: number;
  inkWidth: number;
  inkHeight: number;
  /** Approximate per-letter centers used only by the late-G correspondence. */
  letterCenters: Array<{ x: number; y: number }>;
  empty: boolean;
};

/**
 * Rasterise the word with canvas fillText, then build the field the settle
 * solver reads. Equivalent to render_word_mask + build_sdf in probe F, except
 * that the browser supplies the type.
 */
export function buildGlyph(text: string, fontSize: number, fontAuto = false): Glyph {
  const fw = Math.round((CANVAS_W + 2 * SDF_PAD) * SDF_SCALE);
  const fh = Math.round((CANVAS_H + 2 * SDF_PAD) * SDF_SCALE);
  const canvas = document.createElement('canvas');
  canvas.width = fw;
  canvas.height = fh;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Unable to rasterise the word');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, fw, fh);
  const face = (sizePx: number) => `900 ${sizePx}px "Arial Black", "Arial Bold", "Helvetica Neue", Arial, sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#fff';
  const trimmed = text.length ? text : ' ';
  /** Ink box of the whole word at a given size, in FIELD px. */
  const inkBox = (sizePx: number) => {
    ctx.font = face(sizePx);
    const m = ctx.measureText(trimmed);
    const w = (m.actualBoundingBoxLeft ?? 0) + (m.actualBoundingBoxRight ?? m.width);
    const h = (m.actualBoundingBoxAscent ?? sizePx * 0.72) + (m.actualBoundingBoxDescent ?? 0);
    return { w: w > 0 ? w : m.width, h: h > 0 ? h : sizePx * 0.72 };
  };
  let px = fontSize * SDF_SCALE;
  if (fontAuto) {
    // probe-g-global-warp.py render_word: the largest integer size whose INK
    // box fits inside 0.80 of the canvas width AND 0.32 of its height. Fitting
    // on width alone would let a short word like "3D" grow until it is taller
    // than the frame, and a tall word would then set a stroke so wide that the
    // lattice has no cells left to spell with.
    let lo = FIT_MIN_PX;
    let hi = FIT_MAX_PX;
    let best = FIT_MIN_PX;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const box = inkBox(mid * SDF_SCALE);
      if (box.w <= CANVAS_W * FIT_WIDTH * SDF_SCALE && box.h <= CANVAS_H * FIT_HEIGHT * SDF_SCALE) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    px = best * SDF_SCALE;
  }
  ctx.font = face(px);
  const metrics = ctx.measureText(trimmed);
  const left = metrics.actualBoundingBoxLeft ?? 0;
  const right = metrics.actualBoundingBoxRight ?? metrics.width;
  const ascent = metrics.actualBoundingBoxAscent ?? px * 0.72;
  const descent = metrics.actualBoundingBoxDescent ?? 0;
  let inkW = left + right;
  let inkH = ascent + descent;
  if (!(inkW > 0)) inkW = metrics.width;
  if (!(inkH > 0)) inkH = px * 0.72;
  // Shrink to fit if the word runs off the canvas, the same rule the probe used.
  const maxW = CANVAS_W * SDF_SCALE * 0.9;
  const squeeze = inkW > maxW ? maxW / inkW : 1;
  let letterCenters: Array<{ x: number; y: number }> = [];
  if (squeeze < 1) {
    ctx.font = face(px * squeeze);
    const m2 = ctx.measureText(trimmed);
    inkW = (m2.actualBoundingBoxLeft ?? 0) + (m2.actualBoundingBoxRight ?? m2.width);
    inkH = (m2.actualBoundingBoxAscent ?? px * 0.72) + (m2.actualBoundingBoxDescent ?? 0);
    const drawX = (fw - inkW) / 2 + (m2.actualBoundingBoxLeft ?? 0);
    const drawY = (fh - inkH) / 2 + (m2.actualBoundingBoxAscent ?? px * 0.72);
    ctx.fillText(trimmed, drawX, drawY);
    let cursor = drawX;
    for (const char of trimmed) {
      const width = ctx.measureText(char).width;
      letterCenters.push({ x: cursor + width * 0.5, y: drawY - (m2.actualBoundingBoxAscent ?? px * 0.72) * 0.5 });
      cursor += width;
    }
  } else {
    const drawX = (fw - inkW) / 2 + left;
    const drawY = (fh - inkH) / 2 + ascent;
    ctx.fillText(trimmed, drawX, drawY);
    let cursor = drawX;
    for (const char of trimmed) {
      const width = ctx.measureText(char).width;
      letterCenters.push({ x: cursor + width * 0.5, y: drawY - ascent * 0.5 });
      cursor += width;
    }
  }

  const pixels = ctx.getImageData(0, 0, fw, fh).data;
  const ink = new Uint8Array(fw * fh);
  let inkCount = 0;
  for (let p = 0; p < ink.length; p++) {
    const on = pixels[p * 4] > 127 ? 1 : 0;
    ink[p] = on;
    inkCount += on;
  }
  const outside = edt2d(ink, fw, fh, 0); // distance to the nearest non-ink
  const inside = edt2d(ink, fw, fh, 1); // distance to the nearest ink
  const step = 1 / SDF_SCALE; // canvas px per field sample
  const sdf = new Float32Array(fw * fh);
  for (let p = 0; p < sdf.length; p++) sdf[p] = (ink[p] ? -outside[p] : inside[p]) * step;
  blur3(sdf, fw, fh, 1); // the probe's 1.2 px Gaussian, at half resolution

  const gx = new Float32Array(fw * fh);
  const gy = new Float32Array(fw * fh);
  const coh = new Float32Array(fw * fh);
  for (let y = 0; y < fh; y++) {
    for (let x = 0; x < fw; x++) {
      const i = y * fw + x;
      const dx = (sdf[y * fw + Math.min(fw - 1, x + 1)] - sdf[y * fw + Math.max(0, x - 1)]) / (2 * step);
      const dy = (sdf[Math.min(fh - 1, y + 1) * fw + x] - sdf[Math.max(0, y - 1) * fw + x]) / (2 * step);
      const n = Math.hypot(dx, dy) + 1e-6;
      gx[i] = dx / n;
      gy[i] = dy / n;
      coh[i] = Math.min(1, n);
    }
  }
  blur3(coh, fw, fh, 2);
  // Curvature of the level set: |div(unit gradient)|. It blows up in the fan
  // outside a sharp corner, where a wedge of vertices all project to one point.
  const curv = new Float32Array(fw * fh);
  for (let y = 0; y < fh; y++) {
    for (let x = 0; x < fw; x++) {
      const i = y * fw + x;
      const kxx = (gx[y * fw + Math.min(fw - 1, x + 1)] - gx[y * fw + Math.max(0, x - 1)]) / (2 * step);
      const kyy = (gy[Math.min(fh - 1, y + 1) * fw + x] - gy[Math.max(0, y - 1) * fw + x]) / (2 * step);
      curv[i] = Math.abs(kxx + kyy);
    }
  }
  blur3(curv, fw, fh, 2);

  // Stroke width = 2 x the 97th percentile of the interior distance.
  let strokePx = 0;
  if (inkCount > 0) {
    const interior: number[] = [];
    for (let p = 0; p < ink.length; p += 1) if (ink[p]) interior.push(outside[p]);
    interior.sort((a, b) => a - b);
    strokePx = 2 * interior[Math.min(interior.length - 1, Math.floor(interior.length * 0.97))] * step;
  }
  return {
    w: fw,
    h: fh,
    sdf,
    gx,
    gy,
    coh,
    curv,
    strokePx,
    fontPx: (px * (squeeze < 1 ? squeeze : 1)) / SDF_SCALE,
    inkWidth: inkW / SDF_SCALE,
    inkHeight: inkH / SDF_SCALE,
    letterCenters: (letterCenters.length ? letterCenters : [{ x: fw * 0.5, y: fh * 0.5 }]).map((center) => ({
      x: center.x / SDF_SCALE - SDF_PAD,
      y: center.y / SDF_SCALE - SDF_PAD,
    })),
    empty: inkCount === 0,
  };
}

/** Bilinear sample of a padded field at canvas coordinates. */
function sampleField(field: Float32Array, w: number, h: number, x: number, y: number): number {
  let fx = (x + SDF_PAD) * SDF_SCALE;
  let fy = (y + SDF_PAD) * SDF_SCALE;
  if (fx < 0) fx = 0;
  if (fy < 0) fy = 0;
  if (fx > w - 1.001) fx = w - 1.001;
  if (fy > h - 1.001) fy = h - 1.001;
  const x0 = fx | 0;
  const y0 = fy | 0;
  const tx = fx - x0;
  const ty = fy - y0;
  const r0 = y0 * w + x0;
  const r1 = r0 + w;
  const a = field[r0];
  const b = field[r0 + 1];
  const c = field[r1];
  const d = field[r1 + 1];
  return a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty;
}

/* ---------------------------------------------------------------------- */
/* 2. lattice                                                              */
/* ---------------------------------------------------------------------- */

export type Lattice = {
  columns: number;
  rows: number;
  nx: number; // cells across, including overscan
  ny: number;
  h: number;
  hx: number;
  hy: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  nodes: number;
  restX: Float64Array;
  restY: Float64Array;
};

export function buildLattice(columns: number): Lattice {
  const h = CANVAS_W / columns;
  const ov = OVERSCAN_CELLS * h;
  const nx = Math.round((CANVAS_W + 2 * ov) / h);
  const ny = Math.round((CANVAS_H + 2 * ov) / h);
  const x0 = CANVAS_W / 2 - (nx * h) / 2;
  const y0 = CANVAS_H / 2 - (ny * h) / 2;
  const cols = nx + 1;
  const rows = ny + 1;
  const restX = new Float64Array(cols * rows);
  const restY = new Float64Array(cols * rows);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      restX[j * cols + i] = x0 + i * h;
      restY[j * cols + i] = y0 + j * h;
    }
  }
  return {
    columns,
    rows: Math.round(CANVAS_H / h),
    nx,
    ny,
    h,
    hx: h,
    hy: h,
    x0,
    y0,
    x1: x0 + nx * h,
    y1: y0 + ny * h,
    nodes: cols * rows,
    restX,
    restY,
  };
}

/* ---------------------------------------------------------------------- */
/* 3. the settle solver (probe F)                                          */
/* ---------------------------------------------------------------------- */

type SettleState = {
  lat: Lattice;
  glyph: Glyph;
  px: Float64Array;
  py: Float64Array;
  fx: Float64Array;
  fy: Float64Array;
  diag: Float64Array;
  kAtt: number;
  sigma: number;
  iterations: number;
  done: number;
};

function createSettle(lat: Lattice, glyph: Glyph, gain: number, band: number): SettleState {
  const n = lat.nodes;
  return {
    lat,
    glyph,
    px: Float64Array.from(lat.restX),
    py: Float64Array.from(lat.restY),
    fx: new Float64Array(n),
    fy: new Float64Array(n),
    diag: new Float64Array(n),
    kAtt: CFG.kAtt * gain,
    sigma: band,
    iterations: Math.round(600 + 9 * lat.columns),
    done: 0,
  };
}

function settleStep(s: SettleState, alpha: number, dt: number) {
  const { lat, glyph } = s;
  const cols = lat.nx + 1;
  const rows = lat.ny + 1;
  const { px, py, fx, fy, diag } = s;
  fx.fill(0);
  fy.fill(0);
  const base = 4 * CFG.kSpring + 4 * CFG.kShape + CFG.kHome + 1e-3;
  diag.fill(base);

  // axial springs
  for (let j = 0; j < rows; j++) {
    const row = j * cols;
    for (let i = 0; i < cols - 1; i++) {
      const a = row + i;
      const b = a + 1;
      const dx = px[b] - px[a];
      const dy = py[b] - py[a];
      const L = Math.hypot(dx, dy) + 1e-9;
      const f = (CFG.kSpring * (L - lat.hx)) / L;
      fx[a] += f * dx;
      fy[a] += f * dy;
      fx[b] -= f * dx;
      fy[b] -= f * dy;
    }
  }
  for (let j = 0; j < rows - 1; j++) {
    for (let i = 0; i < cols; i++) {
      const a = j * cols + i;
      const b = a + cols;
      const dx = px[b] - px[a];
      const dy = py[b] - py[a];
      const L = Math.hypot(dx, dy) + 1e-9;
      const f = (CFG.kSpring * (L - lat.hy)) / L;
      fx[a] += f * dx;
      fy[a] += f * dy;
      fx[b] -= f * dx;
      fy[b] -= f * dy;
    }
  }
  // diagonal (shape) springs: without shear stiffness the lattice shreds
  const Ld = Math.hypot(lat.hx, lat.hy);
  if (CFG.kShape > 0) {
    for (let j = 0; j < rows - 1; j++) {
      for (let i = 0; i < cols - 1; i++) {
        const a = j * cols + i;
        const b = a + 1;
        const c = a + cols + 1;
        const d = a + cols;
        let dx = px[c] - px[a];
        let dy = py[c] - py[a];
        let L = Math.hypot(dx, dy) + 1e-9;
        let f = (CFG.kShape * (L - Ld)) / L;
        fx[a] += f * dx;
        fy[a] += f * dy;
        fx[c] -= f * dx;
        fy[c] -= f * dy;
        dx = px[d] - px[b];
        dy = py[d] - py[b];
        L = Math.hypot(dx, dy) + 1e-9;
        f = (CFG.kShape * (L - Ld)) / L;
        fx[b] += f * dx;
        fy[b] += f * dy;
        fx[d] -= f * dx;
        fy[d] -= f * dy;
      }
    }
  }
  // attraction to the zero crossing, plus the home spring
  const sig = s.sigma * lat.h;
  for (let k = 0; k < s.px.length; k++) {
    const x = px[k];
    const y = py[k];
    const d = sampleField(glyph.sdf, glyph.w, glyph.h, x, y);
    let w = Math.exp(-((d / sig) * (d / sig)));
    if (w > 1e-6) {
      const c = Math.min(1, Math.max(0, sampleField(glyph.coh, glyph.w, glyph.h, x, y)));
      w *= c * c * c; // coh_pow = 3
      const kv = sampleField(glyph.curv, glyph.w, glyph.h, x, y) * lat.h;
      w /= 1 + CFG.kCurv * kv * kv;
      let ux = sampleField(glyph.gx, glyph.w, glyph.h, x, y);
      let uy = sampleField(glyph.gy, glyph.w, glyph.h, x, y);
      const n = Math.hypot(ux, uy) + 1e-6;
      ux /= n;
      uy /= n;
      const mag = alpha * s.kAtt * w * d;
      fx[k] -= mag * ux;
      fy[k] -= mag * uy;
      diag[k] += alpha * s.kAtt * w;
    }
    fx[k] += CFG.kHome * (lat.restX[k] - x);
    fy[k] += CFG.kHome * (lat.restY[k] - y);
  }
  // Jacobi-preconditioned descent with a loose step cap
  const cap = CFG.stepCap * lat.h;
  for (let k = 0; k < px.length; k++) {
    let dx = (dt * fx[k]) / diag[k];
    let dy = (dt * fy[k]) / diag[k];
    const m = Math.hypot(dx, dy);
    if (m > cap) {
      dx *= cap / (m + 1e-12);
      dy *= cap / (m + 1e-12);
    }
    px[k] += dx;
    py[k] += dy;
  }
  projectAreas(s);
  // the outer ring stays on the overscan rectangle and slides freely
  for (let i = 0; i < cols; i++) {
    py[i] = lat.y0;
    py[(rows - 1) * cols + i] = lat.y1;
  }
  for (let j = 0; j < rows; j++) {
    px[j * cols] = lat.x0;
    px[j * cols + cols - 1] = lat.x1;
  }
}

/** Gauss-Newton projection onto area >= projMin * rest area. A projection,
 *  not a force, so it cannot overshoot and cannot explode. */
function projectAreas(s: SettleState) {
  const { lat, px, py } = s;
  const cols = lat.nx + 1;
  const rows = lat.ny + 1;
  const A0 = lat.hx * lat.hy;
  const Amin = CFG.projMin * A0;
  const accX = new Float64Array(px.length);
  const accY = new Float64Array(px.length);
  const cnt = new Float64Array(px.length);
  for (let pass = 0; pass < CFG.projSub; pass++) {
    accX.fill(0);
    accY.fill(0);
    cnt.fill(0);
    let any = false;
    for (let j = 0; j < rows - 1; j++) {
      for (let i = 0; i < cols - 1; i++) {
        const a = j * cols + i;
        const b = a + 1;
        const c = a + cols + 1;
        const d = a + cols;
        const area =
          0.5 *
          (px[a] * py[b] - px[b] * py[a] + px[b] * py[c] - px[c] * py[b] + px[c] * py[d] - px[d] * py[c] + px[d] * py[a] - px[a] * py[d]);
        const C = area - Amin;
        if (C >= 0) continue;
        any = true;
        // dA/dp_k = 0.5 * rot(p_{k+1} - p_{k-1}), rot(u,v) = (v, -u)
        const gax = 0.5 * (py[b] - py[d]);
        const gay = -0.5 * (px[b] - px[d]);
        const gbx = 0.5 * (py[c] - py[a]);
        const gby = -0.5 * (px[c] - px[a]);
        const gcx = 0.5 * (py[d] - py[b]);
        const gcy = -0.5 * (px[d] - px[b]);
        const gdx = 0.5 * (py[a] - py[c]);
        const gdy = -0.5 * (px[a] - px[c]);
        const den = gax * gax + gay * gay + gbx * gbx + gby * gby + gcx * gcx + gcy * gcy + gdx * gdx + gdy * gdy + 1e-9;
        const lam = ((-C / den) * CFG.projRelax);
        accX[a] += lam * gax;
        accY[a] += lam * gay;
        accX[b] += lam * gbx;
        accY[b] += lam * gby;
        accX[c] += lam * gcx;
        accY[c] += lam * gcy;
        accX[d] += lam * gdx;
        accY[d] += lam * gdy;
        cnt[a]++;
        cnt[b]++;
        cnt[c]++;
        cnt[d]++;
      }
    }
    if (!any) break;
    for (let k = 0; k < px.length; k++) {
      if (cnt[k] > 0) {
        px[k] += accX[k] / cnt[k];
        py[k] += accY[k] / cnt[k];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */
/* 4. the biharmonic solver (probe G)                                      */
/* ---------------------------------------------------------------------- */

/** probe-g-global-warp.py WARP: build(alpha=3e-4, order=1). */
const WARP_ALPHA = 3e-4;
/** probe-g-global-warp.py WARP: source_mode="scale", scale_k, rot_deg. */
const LENS_SCALE = 0.82;
const LENS_DEG = 25;

type CgChannel = {
  u: Float64Array; // the solution, pre-seeded with g on the constrained nodes
  r: Float64Array;
  z: Float64Array;
  p: Float64Array;
  q: Float64Array;
  rz: number;
  rz0: number;
  started: boolean;
  settled: boolean;
};

type WarpState = {
  lat: Lattice;
  glyph: Glyph;
  ux: Float64Array;
  uy: Float64Array;
  fixed: Uint8Array;
  diag: Float64Array;
  tmp: Float64Array;
  cg: CgChannel[];
  iterations: number;
  done: number;
};

/**
 * G solves min ||L u||^2 + a||u||^2 subject to u = target - source on a chain
 * of nodes adjacent to a correspondence contour. `offset` is the first browser
 * approximation: nodes a stroke fraction outside the glyph land on the SDF
 * contour. `late-g-lens` is the probe's shipped correspondence: the source is
 * a similarity-scaled, rotated copy of each letter about its own centre
 * (`scale=.82`, `rotation=25°`), so the footage inside a letter is magnified
 * by 1/scale rather than creased along its edge.
 */
/** Surviving constraint count from the last warp build. A correspondence
 *  that spells nothing usually has an almost empty constraint set, which is
 *  invisible in the render but obvious in one number. */
let lastConstraints = 0;

function createWarp(lat: Lattice, glyph: Glyph, gain: number, correspondence: WarpCorrespondence): WarpState {
  const n = lat.nodes;
  const ux = new Float64Array(n);
  const uy = new Float64Array(n);
  const fixed = new Uint8Array(n);
  const cols = lat.nx + 1;
  const rows = lat.ny + 1;
  const off = Math.max(2, gain * glyph.strokePx);
  const lensAngle = (LENS_DEG * Math.PI) / 180;
  const inverseScale = 1 / LENS_SCALE;
  // The probe snaps every SOURCE sample to its NEAREST lattice node, so the
  // chain is one node wide. Inverting that test puts the acceptance band in
  // TARGET space, where half a cell of source becomes half a cell over the
  // scale factor. A wider band constrains two or three parallel rows of nodes
  // to conflicting targets and rigidifies the letter edge into a wall.
  const bandHalf = correspondence === 'late-g-lens' ? 0.5 * lat.h * inverseScale : 0.7 * lat.h;
  // R(-theta): the probe builds the SOURCE from the target as
  //   Q - cent = scale * R(+25 deg) * (P - cent)
  // so recovering P from a source node needs R transposed. The first port used
  // R(+25 deg) here, a 50 degree error that left the scale right and sheared
  // the lens the wrong way.
  const cos = Math.cos(lensAngle);
  const sin = Math.sin(lensAngle);

  const nearestLetterCenter = (x: number, y: number) => {
    let best = glyph.letterCenters[0];
    let bestD = Infinity;
    for (const center of glyph.letterCenters) {
      const dx = x - center.x;
      const dy = y - center.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        best = center;
        bestD = d;
      }
    }
    return best;
  };
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      const x = lat.restX[k];
      const y = lat.restY[k];
      let sourceX = x;
      let sourceY = y;
      let targetX = x;
      let targetY = y;
      let contourDistance = Infinity;
      let cohX = x;
      let cohY = y;
      if (correspondence === 'late-g-lens') {
        // The offline probe builds Q from each letter's target contour P. At
        // runtime we invert that similarity map for a source node and test
        // the resulting point against the browser SDF. This preserves the
        // correspondence's scale and rotation without shipping marching
        // squares/control-pair data; the solve itself is identical.
        const center = nearestLetterCenter(x, y);
        const dx = (x - center.x) * inverseScale;
        const dy = (y - center.y) * inverseScale;
        const px = center.x + cos * dx + sin * dy;
        const py = center.y - sin * dx + cos * dy;
        const pd = sampleField(glyph.sdf, glyph.w, glyph.h, px, py);
        contourDistance = Math.abs(pd);
        cohX = px;
        cohY = py;
        if (contourDistance > bandHalf) continue;
        sourceX = x;
        sourceY = y;
        targetX = px;
        targetY = py;
      } else {
        const d = sampleField(glyph.sdf, glyph.w, glyph.h, x, y);
        contourDistance = Math.abs(d - off);
        if (contourDistance > bandHalf) continue;
        let nx = sampleField(glyph.gx, glyph.w, glyph.h, x, y);
        let ny = sampleField(glyph.gy, glyph.w, glyph.h, x, y);
        const m = Math.hypot(nx, ny) + 1e-6;
        nx /= m;
        ny /= m;
        targetX = x - d * nx;
        targetY = y - d * ny;
        // Only the offset family projects along a normal, so only it can be
        // wrecked by an ambiguous one. The probe's lens correspondence needs
        // no normal at all, and dropping its high-curvature nodes deletes
        // exactly the corners that make a letterform read.
        const coh = Math.min(1, Math.max(0, sampleField(glyph.coh, glyph.w, glyph.h, cohX, cohY)));
        if (coh < 0.55) continue; // ambiguous normal, drop the constraint
        const curv = sampleField(glyph.curv, glyph.w, glyph.h, cohX, cohY) * lat.h;
        if (curv > 0.9) continue; // corner fan, a wedge of nodes on one point
      }
      fixed[k] = 1;
      ux[k] = (targetX - sourceX) * (correspondence === 'late-g-lens' ? gain / 0.8 : 1);
      uy[k] = (targetY - sourceY) * (correspondence === 'late-g-lens' ? gain / 0.8 : 1);
    }
  }
  let constraints = 0;
  for (let k = 0; k < n; k++) if (fixed[k]) constraints += 1;
  lastConstraints = constraints;

  // Jacobi diagonal of M = L^T L + alpha I, with L the probe's degree-aware
  // grid Laplacian: M[k][k] = deg^2 + deg + alpha.
  const diag = new Float64Array(n);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const deg = (i > 0 ? 1 : 0) + (i < cols - 1 ? 1 : 0) + (j > 0 ? 1 : 0) + (j < rows - 1 ? 1 : 0);
      diag[j * cols + i] = deg * deg + deg + WARP_ALPHA;
    }
  }
  const channel = (u: Float64Array): CgChannel => ({
    u,
    r: new Float64Array(n),
    z: new Float64Array(n),
    p: new Float64Array(n),
    q: new Float64Array(n),
    rz: 0,
    rz0: 0,
    started: false,
    settled: constraints === 0,
  });
  return {
    lat,
    glyph,
    ux,
    uy,
    fixed,
    diag,
    tmp: new Float64Array(n),
    cg: [channel(ux), channel(uy)],
    iterations: 3000,
    done: 0,
  };
}

/**
 * probe-g-global-warp.py grid_laplacian: 1 on each IN-GRID neighbour and minus
 * that degree on the diagonal, so the boundary carries the natural condition
 * the probe solved with rather than a Dirichlet ring that pins the field to
 * zero three cells outside the canvas.
 */
function laplacian(lat: Lattice, v: Float64Array, out: Float64Array) {
  const cols = lat.nx + 1;
  const rows = lat.ny + 1;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      let s = 0;
      let deg = 0;
      if (i > 0) {
        s += v[k - 1];
        deg++;
      }
      if (i < cols - 1) {
        s += v[k + 1];
        deg++;
      }
      if (j > 0) {
        s += v[k - cols];
        deg++;
      }
      if (j < rows - 1) {
        s += v[k + cols];
        deg++;
      }
      out[k] = s - deg * v[k];
    }
  }
}

/** M v = L^T L v + alpha v, then restricted to the free nodes. */
function applyM(s: WarpState, v: Float64Array, out: Float64Array) {
  laplacian(s.lat, v, s.tmp);
  laplacian(s.lat, s.tmp, out);
  for (let k = 0; k < out.length; k++) {
    out[k] = s.fixed[k] ? 0 : out[k] + WARP_ALPHA * v[k];
  }
}

/**
 * Preconditioned conjugate gradient on the normal equations of the probe's
 * objective, one channel per axis.
 *
 * The first port ran Jacobi on `(4 + alpha) u = sum of neighbours`, which is
 * LAPLACE, not the probe's solve. Minimising ||L u||^2 gives the fourth-order
 * L^T L u = 0: C1 across the constraint chain and a far broader reach. The
 * second-order field kinks at every constrained node and dies within a couple
 * of cells of the letter, which is why the word never reads as a lens.
 */
function cgIterate(s: WarpState, ch: CgChannel, iters: number) {
  if (ch.settled) return;
  const n = ch.u.length;
  if (!ch.started) {
    // r = -(M u) on the free nodes, with u carrying g on the constrained ones
    applyM(s, ch.u, ch.r);
    let rz = 0;
    for (let k = 0; k < n; k++) {
      ch.r[k] = -ch.r[k];
      ch.z[k] = ch.r[k] / s.diag[k];
      ch.p[k] = ch.z[k];
      rz += ch.r[k] * ch.z[k];
    }
    ch.rz = rz;
    ch.rz0 = rz;
    ch.started = true;
    if (!(rz > 0)) {
      ch.settled = true;
      return;
    }
  }
  for (let it = 0; it < iters; it++) {
    applyM(s, ch.p, ch.q);
    let pq = 0;
    for (let k = 0; k < n; k++) pq += ch.p[k] * ch.q[k];
    if (!(pq > 1e-300)) {
      ch.settled = true;
      return;
    }
    const a = ch.rz / pq;
    let rz2 = 0;
    for (let k = 0; k < n; k++) {
      ch.u[k] += a * ch.p[k];
      ch.r[k] -= a * ch.q[k];
      ch.z[k] = ch.r[k] / s.diag[k];
      rz2 += ch.r[k] * ch.z[k];
    }
    if (rz2 <= 1e-12 * ch.rz0) {
      ch.settled = true;
      return;
    }
    const b = rz2 / ch.rz;
    for (let k = 0; k < n; k++) ch.p[k] = ch.z[k] + b * ch.p[k];
    ch.rz = rz2;
  }
}

function warpStep(s: WarpState, sweeps: number) {
  cgIterate(s, s.cg[0], sweeps);
  cgIterate(s, s.cg[1], sweeps);
  if (s.cg[0].settled && s.cg[1].settled) s.done = s.iterations;
}

/* ---------------------------------------------------------------------- */
/* 5. the solved mesh                                                      */
/* ---------------------------------------------------------------------- */

export type Solved = {
  lat: Lattice;
  glyph: Glyph;
  ux: Float32Array; // displacement at full amplitude, per node, windowed
  uy: Float32Array;
  ux0: Float32Array; // the solved field before the spread window
  uy0: Float32Array;
  nodeDist: Float32Array; // glyph SDF at the undeformed node
  cellDist: Float32Array; // glyph SDF at the undeformed cell centre
  key: string;
  spread: number;
  maxDisp: number;
};

/**
 * How far the distortion fans out from the word. The solve produces whatever
 * reach the springs give it; this windows that field back toward the letters,
 * so the slider is live and needs no re-solve. `spread` is in stroke widths.
 */
function applySpread(solved: Solved, spread: number) {
  // 0 is OFF, which is what probe G does: it windows nothing and lets the
  // biharmonic reach as far as it reaches, because a one-stroke Gaussian
  // erases the surrounding magnification that makes the lens legible.
  //
  // That argument is about G ONLY. The F settle path needs the window: what
  // reads in a wireframe is the DIFFERENCE between neighbouring cells, so an
  // unwindowed F field drifts the whole lattice together and the word stops
  // deflecting even though every node moved further. The default stays 1, and
  // `faithful-g-single` is the preset that turns it off.
  const falloff = spread > 0 ? Math.max(4, spread * (solved.glyph.strokePx || 50)) : 0;
  let maxDisp = 0;
  for (let k = 0; k < solved.ux.length; k++) {
    const d = Math.max(0, solved.nodeDist[k]);
    const w = falloff <= 0 || d <= 0 ? 1 : Math.exp(-((d / falloff) * (d / falloff)));
    solved.ux[k] = solved.ux0[k] * w;
    solved.uy[k] = solved.uy0[k] * w;
    const m = Math.hypot(solved.ux[k], solved.uy[k]);
    if (m > maxDisp) maxDisp = m;
  }
  solved.maxDisp = Math.max(1, maxDisp);
  solved.spread = spread;
}

/* ---------------------------------------------------------------------- */
/* 6. wobble, cycle and strength schedule                                  */
/* ---------------------------------------------------------------------- */

/** probe-f-force-field.py wobble_home, verbatim. Loop periodic in T. */
function wobbleAt(x: number, y: number, h: number, phase: number, out: { x: number; y: number }) {
  const k = (2 * Math.PI) / (7.3 * h);
  const ph = phase;
  out.x =
    Math.sin(k * x * 0.9 + 1.7 * y * k * 0.45 + ph) * 0.55 +
    Math.sin(k * 1.9 * y - k * 0.7 * x + 2 * ph + 1.1) * 0.3 +
    Math.sin(k * 0.45 * x + k * 1.3 * y - 3 * ph + 2.4) * 0.15;
  out.y =
    Math.cos(k * 1.1 * y - k * 0.6 * x + ph + 0.6) * 0.55 +
    Math.cos(k * 0.8 * x + k * 1.6 * y - 2 * ph + 2.2) * 0.3 +
    Math.cos(k * 1.7 * x - k * 0.9 * y + 3 * ph + 0.3) * 0.15;
}

/**
 * probe-g-global-warp.py wobble_smooth: a few low-frequency plane waves.
 *
 * The per-node wobble above is `wobble_home`, which the probe wrote for the
 * WIREFRAME and explicitly rejected for the image warp: at a 7.3-cell
 * wavelength it is white noise in space and reads as chatter or grain once
 * footage is pushed through it. Five plane waves at one to three cycles per
 * canvas read as cloth. The draw comes from mulberry32 rather than numpy's
 * generator, so the waves are the same kind, not the same five.
 */
type SmoothWobble = { fx: number; fy: number; h: number; ph: number; cx: number; cy: number };

function buildSmoothWobble(seed: number, waves = 5): SmoothWobble[] {
  const rand = mulberry(Math.round(seed * 2654435761) | 0);
  const out: SmoothWobble[] = [];
  for (let i = 0; i < waves; i++) {
    const fx = 1 + Math.floor(rand() * 3);
    const fy = 1 + Math.floor(rand() * 3);
    const h = 1 + Math.floor(rand() * 2);
    const ph = rand() * 2 * Math.PI;
    const dir = rand() * 2 * Math.PI;
    out.push({ fx, fy, h, ph, cx: Math.cos(dir), cy: Math.sin(dir) });
  }
  return out;
}

function smoothWobbleAt(waves: SmoothWobble[], x: number, y: number, phase: number, out: { x: number; y: number }) {
  let ax = 0;
  let ay = 0;
  for (const w of waves) {
    const s = Math.sin(2 * Math.PI * ((w.fx * x) / CANVAS_W + (w.fy * y) / CANVAS_H) + w.h * phase + w.ph);
    ax += s * w.cx;
    ay += s * w.cy;
  }
  const norm = 1 / Math.sqrt(waves.length);
  out.x = ax * norm;
  out.y = ay * norm;
}

export type Cycle = { t0: number; t1: number; t2: number; t3: number; loop: number };

export function cycleOf(p: MeshParams): Cycle {
  const t0 = Math.max(0, p.idle);
  const t1 = t0 + Math.max(0.2, p.settle);
  const t2 = t1 + Math.max(0, p.hold);
  const t3 = t2 + Math.max(0.2, p.release);
  return { t0, t1, t2, t3, loop: t3 };
}

/** probe-g-global-warp.py atten2: raised cosine, C1 at every join, exact loop. */
export function amplitudeAt(time: number, c: Cycle, shape: ReleaseShape): number {
  const t = ((time % c.loop) + c.loop) % c.loop;
  if (t <= c.t0) return 0;
  if (t < c.t1) return 0.5 - 0.5 * Math.cos((Math.PI * (t - c.t0)) / (c.t1 - c.t0));
  if (t < c.t2) return 1;
  // `asymmetric` compresses the fall into the front of the release window, so
  // the word stays fully formed and then cuts, rather than rewinding.
  const span = (shape === 'asymmetric' ? 0.45 : 1) * (c.t3 - c.t2);
  if (t >= c.t2 + span) return 0;
  return 0.5 + 0.5 * Math.cos((Math.PI * (t - c.t2)) / span);
}

function smoothUnit(value: number): number {
  const u = Math.max(0, Math.min(1, value));
  return u * u * (3 - 2 * u);
}

/**
 * probe-g-global-warp.py strength_sched. The flare is a Gaussian in AMPLITUDE,
 * not in time, so it auto-centres on the forming moment at any settle length.
 */
export function strengthAt(time: number, c: Cycle, p: MeshParams): number {
  const t = ((time % c.loop) + c.loop) % c.loop;
  const a = amplitudeAt(t, c, p.releaseShape);
  const base = p.strength;
  const k = Math.max(0, p.flarePeak - base);
  const flare = (x: number) => base + k * Math.exp(-(((x - p.flareCentre) / p.flareWidth) ** 2));
  let s: number;
  if (t <= c.t1) {
    if (p.schedule === 'const') s = base;
    else if (p.schedule === 'inverse') s = 0.3 + 0.7 * a * a * a;
    else if (p.schedule === 'lead') s = flare(amplitudeAt(t + 0.55, c, p.releaseShape));
    else s = flare(a); // overshoot and lag share the formation flare
    if (p.clampIdleTail && a < 0.02) s = base;
  } else if (t < c.t2) {
    s = base;
  } else if (p.schedule === 'lag') {
    s = base + 0.75 * Math.max(0, amplitudeAt(t - 0.7, c, p.releaseShape) - a) * 2;
  } else if (p.releaseShape === 'disperse') {
    // strength collapses ahead of the geometry: the word loses its edge while
    // the mesh is still rippling, which reads as dispersal rather than rewind
    const f = (t - c.t2) / (c.t3 - c.t2);
    s = base * Math.max(0, 1 - 2.4 * f);
  } else {
    s = base;
  }
  return s;
}

/* ---------------------------------------------------------------------- */
/* 7. per-cell palette (probe F CellPalette)                               */
/* ---------------------------------------------------------------------- */

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type Palette = {
  mix: Int8Array; // the fixed jumble: one random clip per cell, never changed
  fam: Int8Array; // clip a cell settles to, only used by the commit variant
  idle: Int8Array; // clip a cell shows at amplitude 0
  core: Uint8Array; // the formed word and near contour, blended before the push
  tau: Float32Array; // amplitude at which the cell switches
  rFam: Float32Array; // the per-cell draws, kept so that moving the word does
  rIdle: Float32Array; // not reshuffle the collage
  rTau: Float32Array;
  offsetX: number;
  offsetY: number;
};

function buildPalette(solved: Solved, p: MeshParams, luma: number[]): Palette {
  const cells = solved.lat.nx * solved.lat.ny;
  const rand = mulberry(Math.round(p.seed * 2654435761) | 0);
  const pal: Palette = {
    mix: new Int8Array(cells),
    fam: new Int8Array(cells),
    idle: new Int8Array(cells),
    core: new Uint8Array(cells),
    tau: new Float32Array(cells),
    rFam: new Float32Array(cells),
    rIdle: new Float32Array(cells),
    rTau: new Float32Array(cells),
    offsetX: NaN,
    offsetY: NaN,
  };
  for (let c = 0; c < cells; c++) {
    pal.rFam[c] = rand();
    pal.rIdle[c] = rand();
    pal.rTau[c] = rand();
    // The palette is fixed for the whole cycle: idle and hold are identical
    // and nothing commits. Partitioning the clips by luminance made every cell
    // outside the word go black at the hold on this footage.
    pal.mix[c] = Math.min(3, Math.floor(pal.rIdle[c] * 4));
  }
  refreshPalette(pal, solved, p, luma, p.offsetX, p.offsetY);
  return pal;
}

/**
 * Family membership is decided from the glyph, once per position, never from
 * the live wobbling mesh: recomputing it per frame makes cells on the contour
 * flip family as the lattice breathes and the outline sparkles.
 */
function refreshPalette(pal: Palette, solved: Solved, p: MeshParams, luma: number[], offsetX: number, offsetY: number) {
  const { lat, glyph } = solved;
  const order = luma.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const home = Math.min(3, Math.max(0, Math.round(p.homeClip)));
  let outside: number[];
  let insideFam: number[];
  if (!p.familySplit) {
    outside = [0, 1, 2, 3];
    insideFam = [0, 1, 2, 3];
  } else if (p.familyMode !== 'luminance') {
    // The background commits to the clip the word is FOR, and every other clip
    // commits away out of it. The home clip has to be excluded from the inside
    // pool: an inside cell drawing the same clip as the ground behind it is
    // invisible, and those holes fall on the letters, which is the one place
    // the design cannot afford them.
    outside = [home];
    insideFam = [0, 1, 2, 3].filter((clip) => clip !== home);
  } else {
    outside = [order[0][1], order[1][1]];
    insideFam = [order[2][1], order[3][1]];
  }
  for (let j = 0; j < lat.ny; j++) {
    for (let i = 0; i < lat.nx; i++) {
      const c = j * lat.nx + i;
      const cx = lat.x0 + (i + 0.5) * lat.hx - offsetX;
      const cy = lat.y0 + (j + 0.5) * lat.hy - offsetY;
      const dv = sampleField(glyph.sdf, glyph.w, glyph.h, cx, cy);
      solved.cellDist[c] = dv;
      const pool = dv < 0 ? insideFam : outside;
      pal.fam[c] = pool[Math.min(pool.length - 1, Math.floor(pal.rFam[c] * pool.length))];
      // Letter cells reach their family before the field makes the word
      // readable, then stay fixed through push and hold. Surrounding cells
      // wait for the distance front. Release returns the original idle mix.
      const stableCore = p.commit && p.commitTransition === 'push' && p.familySplit && dv < 0;
      pal.core[c] = stableCore ? 1 : 0;
      pal.idle[c] = p.idleField === 'single' ? p.clip : Math.min(3, Math.floor(pal.rIdle[c] * 4));
      // cells near the contour commit first, so the letter resolves from its
      // own edge outward instead of speckling in at random
      const r = Math.min(1, Math.abs(dv) / (6 * lat.h));
      const u = Math.min(1, Math.max(0, p.stagger * r + (1 - p.stagger) * pal.rTau[c]));
      pal.tau[c] = 0.1 + 0.7 * u;
    }
  }
  pal.offsetX = offsetX;
  pal.offsetY = offsetY;
}

/* ---------------------------------------------------------------------- */
/* 8. WebGL mesh renderer                                                  */
/* ---------------------------------------------------------------------- */

const MESH_VS = `
attribute vec2 aPos;
attribute vec2 aUV;
attribute vec4 aData;
attribute vec4 aDebug;
attribute vec4 aTransition;
attribute vec2 aShadeUV;
uniform vec2 uCanvas;
varying vec2 vUV;
varying vec4 vData;
varying vec4 vDebug;
varying vec4 vTransition;
varying vec2 vShadeUV;
void main() {
  vUV = aUV;
  vData = aData;
  vDebug = aDebug;
  vTransition = aTransition;
  vShadeUV = aShadeUV;
  gl_Position = vec4(aPos.x / uCanvas.x * 2.0 - 1.0, 1.0 - aPos.y / uCanvas.y * 2.0, 0.0, 1.0);
}
`;

const MESH_FS = `
#extension GL_OES_standard_derivatives : enable
precision mediump float;
varying vec2 vUV;
varying vec4 vData;
varying vec4 vDebug;
varying vec4 vTransition;
varying vec2 vShadeUV;
uniform sampler2D clip0;
uniform sampler2D clip1;
uniform sampler2D clip2;
uniform sampler2D clip3;
uniform vec2 uFieldCanvas;
uniform vec4 uFit0;
uniform vec4 uFit1;
uniform vec4 uFit2;
uniform vec4 uFit3;
uniform int uView;
uniform float uStrength;
uniform float uShadeOn;
uniform float uLumaAdaptive;
uniform float uMedian;
uniform float uScreenField;
uniform vec2 uCellUvSize;
uniform float uCommitMotion;

vec2 fit(vec2 uv, vec4 f) { return (uv - 0.5) * f.xy + 0.5 + f.zw; }

vec3 clipColor(float id, vec2 uv) {
  if (id < 0.5) return texture2D(clip0, fit(uv, uFit0)).rgb;
  if (id < 1.5) return texture2D(clip1, fit(uv, uFit1)).rgb;
  if (id < 2.5) return texture2D(clip2, fit(uv, uFit2)).rgb;
  return texture2D(clip3, fit(uv, uFit3)).rgb;
}

vec3 heat(float t) {
  t = clamp(t, 0.0, 1.0);
  return clamp(vec3(1.6 * t - 0.2, 1.2 * t * t, 0.9 - 1.1 * t), 0.0, 1.0);
}

float screenDet() {
  vec2 dx = dFdx(vShadeUV) * vec2(uFieldCanvas.x, uFieldCanvas.y);
  vec2 dy = dFdy(vShadeUV) * vec2(uFieldCanvas.x, uFieldCanvas.y);
  return dx.x * dy.y - dx.y * dy.x;
}

vec3 signedHeat(float value) {
  float t = clamp((value - 0.40) / 2.10, 0.0, 1.0);
  if (value < 1.0) return mix(vec3(0.18, 0.35, 0.95), vec3(0.92), t / 0.6);
  return mix(vec3(0.92), vec3(0.88, 0.25, 0.12), clamp((value - 1.0) / 1.5, 0.0, 1.0));
}

void main() {
  // Each old clip can either contract or be carried out of its cell by the
  // local field direction. The word core fades between its two palettes only
  // at the opening and release; it stays on its family through push and hold.
  bool coreBlend = uCommitMotion > 1.5 && vTransition.w > 9.0;
  bool oldVisible = true;
  vec2 footageUV = vUV;
  if (coreBlend) {
    oldVisible = vTransition.x < 0.5;
  } else if (uCommitMotion > 1.5) {
    vec2 direction = vec2(cos(vTransition.w), sin(vTransition.w));
    vec2 travel = direction / max(abs(direction.x), abs(direction.y));
    vec2 moved = vec2(vTransition.y, vTransition.z) - travel * vTransition.x;
    oldVisible = vTransition.x <= 0.0 || (vTransition.x < 1.0 && all(greaterThanEqual(moved, vec2(0.0))) && all(lessThan(moved, vec2(1.0))));
    if (oldVisible) footageUV -= travel * vTransition.x * uCellUvSize;
  } else if (uCommitMotion > 0.5) {
    float width = 1.0 - vTransition.x;
    oldVisible = vTransition.x <= 0.0 || (width > 0.0 && abs(vTransition.y - 0.5) < 0.5 * width);
    if (oldVisible && vTransition.x > 0.0) {
      footageUV.x += (vTransition.y - 0.5) * (1.0 / max(width, 0.001) - 1.0) * uCellUvSize.x;
    }
  }
  float clipId = oldVisible ? vData.y : vDebug.w;
  float rawDet = uScreenField > 0.5 ? screenDet() : vDebug.x;
  float absDet = max(abs(rawDet), 0.0001);
  float clamped = clamp(absDet, 0.40, 2.50);
  // The CPU field supplies the Gaussian-like smoothing and median. Derivative
  // data supplies the per-pixel map within each triangle. Blending the two
  // keeps the browser field close to the probe without making triangle edges
  // sparkle when the mesh is coarse.
  float smoothRatio = max(vData.x, 0.05);
  float ratio = uScreenField > 0.5 ? mix(clamped / max(uMedian, 0.001), smoothRatio, 0.35) : smoothRatio;
  vec3 color;
  if (uView == 2) {
    color = heat(vData.z);
  } else if (uView == 3) {
    color = signedHeat(rawDet);
  } else if (uView == 4) {
    color = heat(clamped / 2.5);
  } else if (uView == 5) {
    color = vec3(clamp(pow(max(ratio, 0.05), uStrength) * 0.5, 0.0, 1.0));
  } else if (uView == 6) {
    color = vData.w > 0.5 ? vec3(0.93, 0.25, 0.28) : vec3(0.13, 0.14, 0.17);
  } else if (uView == 7) {
    float id = clipId;
    if (id < 0.5) color = vec3(0.94, 0.40, 0.20);
    else if (id < 1.5) color = vec3(0.15, 0.63, 0.85);
    else if (id < 2.5) color = vec3(0.73, 0.84, 0.31);
    else color = vec3(0.66, 0.43, 0.89);
  } else {
    vec3 src = coreBlend
      ? mix(clipColor(vData.y, vUV), clipColor(vDebug.w, vUV), vTransition.x)
      : clipColor(clipId, footageUV);
    float s = uStrength;
    if (uLumaAdaptive > 0.5) {
      float lum = dot(src, vec3(0.2126, 0.7152, 0.0722));
      s *= mix(1.0, 2.2, clamp(1.0 - lum / 0.35, 0.0, 1.0));
    }
    float shade = uShadeOn > 0.5 ? pow(ratio, s) : 1.0;
    color = clamp(src * shade, 0.0, 1.0);
  }
  gl_FragColor = vec4(color, 1.0);
}
`;

// Firefox on this Mac exposes WebGL without OES_standard_derivatives in
// headless mode. Keep a compileable fallback that uses the smoothed CPU field;
// interactive contexts with the extension use the probe-like per-pixel field.
const MESH_FS_FALLBACK = MESH_FS
  .replace('#extension GL_OES_standard_derivatives : enable\n', '')
  .replace(/float screenDet\(\) \{[\s\S]*?\n\}/, 'float screenDet() { return vDebug.x; }');

const LINE_VS = `
attribute vec2 aPos;
uniform vec2 uCanvas;
void main() {
  gl_Position = vec4(aPos.x / uCanvas.x * 2.0 - 1.0, 1.0 - aPos.y / uCanvas.y * 2.0, 0.0, 1.0);
}
`;

const LINE_FS = `
precision mediump float;
uniform vec3 uColor;
void main() { gl_FragColor = vec4(uColor, 1.0); }
`;

type MeshGL = {
  drawMesh: (data: Float32Array, vertices: number, view: number, strength: number, shadeOn: boolean, luma: boolean, median: number, screenField: boolean, cellUvSize: [number, number], motion: number) => void;
  drawLines: (data: Float32Array, vertices: number) => void;
  uploadVideos: () => void;
  clear: (r: number, g: number, b: number) => void;
  screenField: boolean;
};

function compile(gl: WebGLRenderingContext, type: number, source: string) {
  const sh = gl.createShader(type);
  if (!sh) throw new Error('Unable to create shader');
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? 'Shader compilation failed');
  return sh;
}

function link(gl: WebGLRenderingContext, vs: string, fs: string) {
  const program = gl.createProgram();
  if (!program) throw new Error('Unable to create program');
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'Shader linking failed');
  return program;
}

function createMeshGL(canvas: HTMLCanvasElement, videos: HTMLVideoElement[]): MeshGL | null {
  const gl = canvas.getContext('webgl', { alpha: false, antialias: true, preserveDrawingBuffer: true });
  if (!gl) return null;
  const derivativeExtension = Boolean(gl.getExtension('OES_standard_derivatives'));
  const mesh = link(gl, MESH_VS, derivativeExtension ? MESH_FS : MESH_FS_FALLBACK);
  const lines = link(gl, LINE_VS, LINE_FS);
  const buffer = gl.createBuffer();
  const lineBuffer = gl.createBuffer();
  // The mesh's UVs are the UNDEFORMED lattice in canvas coordinates, so v = 0
  // is the TOP of the canvas (see the aUV packing: restY / CANVAS_H, y down).
  // Flipping the upload would put the video's last row at v = 0 and draw every
  // clip mirrored vertically. The legacy selector-mask renderer does flip,
  // because its full-screen quad builds uv from NDC with y up.
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.disable(gl.BLEND);
  gl.disable(gl.DEPTH_TEST);

  const textures = videos.map((_, index) => {
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([12, 14, 18, 255]));
    return texture;
  });
  const lastTime = videos.map(() => -1);
  videos.forEach((video, index) => video.addEventListener('seeked', () => { lastTime[index] = -1; }));

  gl.useProgram(mesh);
  const uMeshCanvas = gl.getUniformLocation(mesh, 'uCanvas');
  const uFieldCanvas = gl.getUniformLocation(mesh, 'uFieldCanvas');
  const uView = gl.getUniformLocation(mesh, 'uView');
  const uStrength = gl.getUniformLocation(mesh, 'uStrength');
  const uShadeOn = gl.getUniformLocation(mesh, 'uShadeOn');
  const uLuma = gl.getUniformLocation(mesh, 'uLumaAdaptive');
  const uMedian = gl.getUniformLocation(mesh, 'uMedian');
  const uScreenField = gl.getUniformLocation(mesh, 'uScreenField');
  const uCellUvSize = gl.getUniformLocation(mesh, 'uCellUvSize');
  const uCommitMotion = gl.getUniformLocation(mesh, 'uCommitMotion');
  const fitLoc = [0, 1, 2, 3].map((i) => gl.getUniformLocation(mesh, `uFit${i}`));
  [0, 1, 2, 3].forEach((i) => gl.uniform1i(gl.getUniformLocation(mesh, `clip${i}`), i));
  const aPos = gl.getAttribLocation(mesh, 'aPos');
  const aUV = gl.getAttribLocation(mesh, 'aUV');
  const aData = gl.getAttribLocation(mesh, 'aData');
  const aDebug = gl.getAttribLocation(mesh, 'aDebug');
  const aTransition = gl.getAttribLocation(mesh, 'aTransition');
  const aShadeUV = gl.getAttribLocation(mesh, 'aShadeUV');
  gl.useProgram(lines);
  const uLineCanvas = gl.getUniformLocation(lines, 'uCanvas');
  const uLineColor = gl.getUniformLocation(lines, 'uColor');
  const aLinePos = gl.getAttribLocation(lines, 'aPos');

  const fit = (video: HTMLVideoElement) => {
    const vw = video.videoWidth || CANVAS_W;
    const vh = video.videoHeight || CANVAS_H;
    const target = CANVAS_W / CANVAS_H;
    const source = vw / vh;
    return source > target ? [target / source, 1] : [1, source / target];
  };

  return {
    screenField: derivativeExtension,
    clear(r, g, b) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(r, g, b, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    },
    uploadVideos() {
      videos.forEach((video, index) => {
        if (video.readyState < 2 || video.currentTime === lastTime[index]) return;
        gl.activeTexture(gl.TEXTURE0 + index);
        gl.bindTexture(gl.TEXTURE_2D, textures[index]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
        lastTime[index] = video.currentTime;
      });
    },
    drawMesh(data, vertices, view, strength, shadeOn, luma, median, screenField, cellUvSize, motion) {
      gl.useProgram(mesh);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      const stride = 72;
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(aUV);
      gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, stride, 8);
      gl.enableVertexAttribArray(aData);
      gl.vertexAttribPointer(aData, 4, gl.FLOAT, false, stride, 16);
      gl.enableVertexAttribArray(aDebug);
      gl.vertexAttribPointer(aDebug, 4, gl.FLOAT, false, stride, 32);
      gl.enableVertexAttribArray(aTransition);
      gl.vertexAttribPointer(aTransition, 4, gl.FLOAT, false, stride, 48);
      gl.enableVertexAttribArray(aShadeUV);
      gl.vertexAttribPointer(aShadeUV, 2, gl.FLOAT, false, stride, 64);
      gl.uniform2f(uMeshCanvas, CANVAS_W, CANVAS_H);
      gl.uniform2f(uFieldCanvas, CANVAS_W, CANVAS_H);
      gl.uniform1i(uView, view);
      gl.uniform1f(uStrength, strength);
      gl.uniform1f(uShadeOn, shadeOn ? 1 : 0);
      gl.uniform1f(uLuma, luma ? 1 : 0);
      gl.uniform1f(uMedian, median);
      gl.uniform1f(uScreenField, screenField && derivativeExtension ? 1 : 0);
      gl.uniform2f(uCellUvSize, cellUvSize[0], cellUvSize[1]);
      gl.uniform1f(uCommitMotion, motion);
      videos.forEach((video, index) => {
        const [sx, sy] = fit(video);
        gl.uniform4f(fitLoc[index], sx, sy, 0, 0);
      });
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.drawArrays(gl.TRIANGLES, 0, vertices);
      gl.disableVertexAttribArray(aUV);
      gl.disableVertexAttribArray(aData);
      gl.disableVertexAttribArray(aDebug);
      gl.disableVertexAttribArray(aTransition);
      gl.disableVertexAttribArray(aShadeUV);
    },
    drawLines(data, vertices) {
      gl.useProgram(lines);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(aLinePos);
      gl.vertexAttribPointer(aLinePos, 2, gl.FLOAT, false, 8, 0);
      gl.uniform2f(uLineCanvas, CANVAS_W, CANVAS_H);
      gl.uniform3f(uLineColor, 0.12, 0.12, 0.14);
      gl.lineWidth(2);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.drawArrays(gl.LINES, 0, vertices);
    },
  };
}

/* ---------------------------------------------------------------------- */
/* 9. the engine                                                           */
/* ---------------------------------------------------------------------- */

export type MeshStatus = {
  solving: boolean;
  progress: number;
  cellsPerStroke: number;
  cellSize: number;
  columns: number;
  rows: number;
  strokePx: number;
  fontPx: number;
  constraints: number;
  amplitude: number;
  strength: number;
  time: number;
  loop: number;
  foldRate: number;
  invalidRate: number;
  diagonalBdRate: number;
  medianRatio: number;
  smoothingPasses: number;
  screenField: boolean;
  frameMs: number;
  solveMs: number;
  vertices: number;
  note: string;
};

export type MeshShowcase = {
  setParams: (next: MeshParams) => void;
  frame: (clock: number) => void;
  status: () => MeshStatus;
  solveNow: () => void;
  setVisible: (visible: boolean) => void;
};

function debugHeat(t: number): string {
  const v = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 1.6 * v - 0.2));
  const g = Math.max(0, Math.min(1, 1.2 * v * v));
  const b = Math.max(0, Math.min(1, 0.9 - 1.1 * v));
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

function debugSignedHeat(value: number): string {
  const t = Math.max(0, Math.min(1, (value - 0.4) / 2.1));
  if (value < 1) {
    const u = Math.min(1, t / 0.6);
    const r = 0.18 + (0.92 - 0.18) * u;
    const g = 0.35 + (0.92 - 0.35) * u;
    const b = 0.95 + (0.92 - 0.95) * u;
    return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
  }
  const u = Math.max(0, Math.min(1, (value - 1) / 1.5));
  return `rgb(${Math.round((0.92 - 0.04 * u) * 255)}, ${Math.round((0.92 - 0.67 * u) * 255)}, ${Math.round((0.92 - 0.80 * u) * 255)})`;
}

/**
 * Firefox headless can execute the WebGL field but omit its bitmap from a
 * screenshot. This small canvas mirrors the diagnostic views in 2D so the
 * acceptance artifacts remain reviewable. It is hidden for footage and never
 * participates in the normal textured mesh render.
 */
function drawDebugOverlay(
  overlay: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  view: number,
  lineData: Float32Array,
  vertexData: Float32Array,
) {
  overlay.hidden = view < 1 || view > 7;
  if (overlay.hidden) return;
  const width = overlay.width;
  const height = overlay.height;
  context.clearRect(0, 0, width, height);
  if (view === 1) {
    context.fillStyle = '#fafaf7';
    context.fillRect(0, 0, width, height);
    context.strokeStyle = '#1b1d21';
    context.lineWidth = Math.max(1, width / 1280);
    context.beginPath();
    for (let i = 0; i + 3 < lineData.length; i += 4) {
      context.moveTo(lineData[i], lineData[i + 1]);
      context.lineTo(lineData[i + 2], lineData[i + 3]);
    }
    context.stroke();
    return;
  }
  context.fillStyle = '#0b0d10';
  context.fillRect(0, 0, width, height);
  const clipColors = ['rgb(240, 102, 51)', 'rgb(38, 161, 217)', 'rgb(186, 214, 79)', 'rgb(168, 110, 227)'];
  for (let cell = 0; cell + 107 < vertexData.length; cell += 108) {
    for (let tri = 0; tri < 2; tri++) {
      const a = cell + tri * 54;
      const b = a + 18;
      const c = a + 36;
      const raw = vertexData[a + 8];
      const clampRatio = Math.max(0, Math.min(2.5, Math.abs(vertexData[a + 9])));
      const finalMultiplier = Math.max(0, Math.min(2, vertexData[a + 10]));
      const displacement = Math.max(0, Math.min(1, vertexData[a + 6]));
      const invalid = vertexData[a + 7] > 0.5;
      const clip = Math.max(0, Math.min(3, Math.round(vertexData[a + (vertexData[a + 12] >= 0.5 ? 11 : 5)])));
      if (view === 2) context.fillStyle = debugHeat(displacement);
      else if (view === 3) context.fillStyle = debugSignedHeat(raw);
      else if (view === 4) context.fillStyle = debugHeat(clampRatio / 2.5);
      else if (view === 5) {
        const level = Math.max(0, Math.min(1, finalMultiplier * 0.5));
        const channel = Math.round(level * 255);
        context.fillStyle = `rgb(${channel}, ${channel}, ${channel})`;
      } else if (view === 6) context.fillStyle = invalid ? '#ed4047' : '#21242a';
      else context.fillStyle = clipColors[clip];
      context.beginPath();
      context.moveTo(vertexData[a], vertexData[a + 1]);
      context.lineTo(vertexData[b], vertexData[b + 1]);
      context.lineTo(vertexData[c], vertexData[c + 1]);
      context.closePath();
      context.fill();
    }
  }
}

export function createMeshShowcase(
  canvas: HTMLCanvasElement,
  videos: HTMLVideoElement[],
  debugCanvas?: HTMLCanvasElement,
): MeshShowcase | null {
  const gl = createMeshGL(canvas, videos);
  if (!gl) return null;
  const debugContext = debugCanvas?.getContext('2d', { alpha: false }) ?? null;
  if (debugCanvas && debugContext) {
    debugCanvas.width = CANVAS_W;
    debugCanvas.height = CANVAS_H;
    debugCanvas.hidden = true;
  }

  let params: MeshParams = { ...MESH_DEFAULTS };
  let solved: Solved | null = null;
  let palette: Palette | null = null;
  let vertexData = new Float32Array(0);
  let lineData = new Float32Array(0);
  let posX = new Float64Array(0);
  let posY = new Float64Array(0);
  let nodeRatio = new Float32Array(0);
  let nodeRatioB = new Float32Array(0);
  let nodeRawRatio = new Float32Array(0);
  let nodeClampRatio = new Float32Array(0);
  let nodeCount = new Float32Array(0);
  let nodeDisp = new Float32Array(0);
  let footageBreathX = new Float32Array(0);
  let footageBreathY = new Float32Array(0);
  let cellFold = new Uint8Array(0);
  let cellDiagonal = new Uint8Array(0);
  let visible = true;
  let frameMs = 0;
  let solveMs = 0;
  let foldRate = 0;
  let invalidRate = 0;
  let diagonalBdRate = 0;
  let medianRatio = 1;
  let smoothingPasses = 0;
  let lastAmplitude = 0;
  let lastStrength = 0;
  let lastTime = 0;
  let note = '';
  const luma = videos.map(() => 60);
  const lumaDone = videos.map(() => false);

  // one-off solve, run in slices so the previous mesh stays on screen
  type Job = {
    key: string;
    stage: 'glyph' | 'iterate' | 'finish';
    glyph: Glyph | null;
    lat: Lattice | null;
    settle: SettleState | null;
    warp: WarpState | null;
    algorithm: Algorithm;
    started: number;
  };
  let job: Job | null = null;

  const solveKey = (p: MeshParams) => [p.text, p.fontSize, p.fontAuto, p.columns, p.algorithm, p.warpCorrespondence, p.gain.toFixed(3), p.band.toFixed(3)].join('|');
  const paletteKey = (p: MeshParams) => [p.commit, p.commitTransition, p.idleField, p.stagger.toFixed(3), p.clip, p.familySplit, p.familyMode, p.homeClip, p.seed, luma.map((v) => Math.round(v)).join(',')].join('|');
  let lastPaletteKey = '';
  let smoothWobble: SmoothWobble[] = [];
  let smoothWobbleSeed = NaN;

  function measureLuma() {
    videos.forEach((video, index) => {
      if (lumaDone[index] || video.readyState < 2 || !video.videoWidth) return;
      const c = document.createElement('canvas');
      c.width = 32;
      c.height = 18;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;
      try {
        ctx.drawImage(video, 0, 0, 32, 18);
        const data = ctx.getImageData(0, 0, 32, 18).data;
        let sum = 0;
        for (let p = 0; p < data.length; p += 4) sum += 0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2];
        luma[index] = sum / (data.length / 4);
        lumaDone[index] = true; // the family split depends on this
      } catch {
        /* a tainted or undecoded frame: keep the previous estimate */
      }
    });
  }

  function beginSolve(p: MeshParams) {
    job = {
      key: solveKey(p),
      stage: 'glyph',
      glyph: null,
      lat: null,
      settle: null,
      warp: null,
      algorithm: p.algorithm,
      started: performance.now(),
    };
  }

  function advanceSolve(budgetMs: number) {
    if (!job) return;
    const deadline = performance.now() + budgetMs;
    if (job.stage === 'glyph') {
      const p = params;
      job.glyph = buildGlyph(p.text, p.fontSize, p.fontAuto);
      job.lat = buildLattice(p.columns);
      if (job.algorithm === 'settle') job.settle = createSettle(job.lat, job.glyph, p.gain, p.band);
      else job.warp = createWarp(job.lat, job.glyph, p.gain, p.warpCorrespondence);
      job.stage = 'iterate';
      // Draw immediately from an undeformed lattice rather than showing
      // nothing: the word arrives when the solve lands, and on a re-solve the
      // previous mesh keeps running instead.
      if (!solved || solved.lat.columns !== job.lat.columns) {
        solved = {
          lat: job.lat,
          glyph: job.glyph,
          ux: new Float32Array(job.lat.nodes),
          uy: new Float32Array(job.lat.nodes),
          ux0: new Float32Array(job.lat.nodes),
          uy0: new Float32Array(job.lat.nodes),
          nodeDist: new Float32Array(job.lat.nodes),
          cellDist: new Float32Array(job.lat.nx * job.lat.ny),
          key: `${job.key}~pending`,
          spread: params.spread,
          maxDisp: 1,
        };
        allocate(job.lat);
        lastPaletteKey = '';
      }
      return;
    }
    if (job.stage === 'iterate') {
      if (job.settle) {
        const s = job.settle;
        const rampN = Math.max(1, Math.round(s.iterations * CFG.rampFrac));
        while (s.done < s.iterations && performance.now() < deadline) {
          let a = Math.min(1, s.done / rampN);
          a = a * a * (3 - 2 * a);
          let cool = 1;
          const c0 = 1 - CFG.coolFrac;
          if (s.done > s.iterations * c0) cool = 1 - (0.85 * (s.done - s.iterations * c0)) / (s.iterations * CFG.coolFrac);
          settleStep(s, a, CFG.dt * cool);
          s.done++;
        }
        if (s.done >= s.iterations) job.stage = 'finish';
      } else if (job.warp) {
        const w = job.warp;
        while (w.done < w.iterations && performance.now() < deadline) {
          warpStep(w, 12);
          w.done += 12;
        }
        if (w.done >= w.iterations) job.stage = 'finish';
      } else {
        job.stage = 'finish';
      }
      return;
    }
    // finish
    const lat = job.lat!;
    const glyph = job.glyph!;
    const ux0 = new Float32Array(lat.nodes);
    const uy0 = new Float32Array(lat.nodes);
    const nodeDist = new Float32Array(lat.nodes);
    if (job.settle) {
      for (let k = 0; k < lat.nodes; k++) {
        ux0[k] = job.settle.px[k] - lat.restX[k];
        uy0[k] = job.settle.py[k] - lat.restY[k];
      }
    } else if (job.warp) {
      for (let k = 0; k < lat.nodes; k++) {
        ux0[k] = job.warp.ux[k];
        uy0[k] = job.warp.uy[k];
      }
    }
    for (let k = 0; k < lat.nodes; k++) {
      nodeDist[k] = sampleField(glyph.sdf, glyph.w, glyph.h, lat.restX[k], lat.restY[k]);
    }
    solved = {
      lat,
      glyph,
      ux: new Float32Array(lat.nodes),
      uy: new Float32Array(lat.nodes),
      ux0,
      uy0,
      nodeDist,
      cellDist: new Float32Array(lat.nx * lat.ny),
      key: job.key,
      spread: -1,
      maxDisp: 1,
    };
    applySpread(solved, params.spread);
    solveMs = performance.now() - job.started;
    allocate(lat);
    lastPaletteKey = '';
    job = null;
  }

  function allocate(lat: Lattice) {
    const cells = lat.nx * lat.ny;
    posX = new Float64Array(lat.nodes);
    posY = new Float64Array(lat.nodes);
    nodeRatio = new Float32Array(lat.nodes);
    nodeRatioB = new Float32Array(lat.nodes);
    nodeRawRatio = new Float32Array(lat.nodes);
    nodeClampRatio = new Float32Array(lat.nodes);
    nodeCount = new Float32Array(lat.nodes);
    nodeDisp = new Float32Array(lat.nodes);
    footageBreathX = new Float32Array(lat.nodes);
    footageBreathY = new Float32Array(lat.nodes);
    cellFold = new Uint8Array(cells);
    cellDiagonal = new Uint8Array(cells);
    vertexData = new Float32Array(cells * 6 * 18);
    const cols = lat.nx + 1;
    const rows = lat.ny + 1;
    lineData = new Float32Array((lat.nx * rows + lat.ny * cols) * 4);
  }

  /** Bilinear sample of the node displacement field, in lattice space. */
  function sampleDisp(lat: Lattice, field: Float32Array, x: number, y: number): number {
    const gx = (x - lat.x0) / lat.hx;
    const gy = (y - lat.y0) / lat.hy;
    if (gx <= 0 || gy <= 0 || gx >= lat.nx || gy >= lat.ny) return 0;
    const i = gx | 0;
    const j = gy | 0;
    const tx = gx - i;
    const ty = gy - j;
    const cols = lat.nx + 1;
    const k = j * cols + i;
    return (
      field[k] * (1 - tx) * (1 - ty) +
      field[k + 1] * tx * (1 - ty) +
      field[k + cols] * (1 - tx) * ty +
      field[k + cols + 1] * tx * ty
    );
  }

  function buildFrame(clock: number) {
    if (!solved) return;
    const p = params;
    note = `view ${p.view}`;
    const lat = solved.lat;
    const cols = lat.nx + 1;
    const rows = lat.ny + 1;
    const c = cycleOf(p);
    const t = ((clock % c.loop) + c.loop) % c.loop;
    const amplitude = p.amplitudeAuto ? amplitudeAt(t, c, p.releaseShape) : p.amplitude;
    let strength = p.amplitudeAuto ? strengthAt(t, c, p) : strengthWithAmplitude(p, amplitude);
    const stableHold = p.commit && p.commitTransition === 'push' && (p.amplitudeAuto || amplitude >= 0.999);
    // At full strength, hold the formed lattice at its settle pose while its
    // breathing continues as motion of the footage through that lattice.
    // Blend the geometry back to live breathing during release, with no jump.
    const holdWeight = stableHold && t >= c.t1
      ? t < c.t2 ? 1 : 1 - smoothUnit((t - c.t2) / (c.t3 - c.t2))
      : 0;
    if (stableHold && p.amplitudeAuto && t >= c.t1 && t < c.t2) {
      strength = strengthAt(c.t1, c, p);
    }
    lastAmplitude = amplitude;
    lastStrength = strength;
    lastTime = t;

    let offsetX = p.offsetX;
    let offsetY = p.offsetY;
    if (p.drift) {
      offsetX += Math.sin((2 * Math.PI * t) / c.loop) * p.driftAmount;
      offsetY += Math.sin((4 * Math.PI * t) / c.loop + 1.1) * p.driftAmount * 0.35;
      // the clip families have to travel with the word, or the contrast cue
      // detaches from the geometry it is meant to be carrying
      if (palette && p.commit && p.footage === 'jumble' && (Math.abs(palette.offsetX - offsetX) > 0.25 * lat.h || Math.abs(palette.offsetY - offsetY) > 0.25 * lat.h)) {
        refreshPalette(palette, solved, p, luma, offsetX, offsetY);
      }
    }

    // 1. wobble, then push the wobbled lattice through the static field
    const phase = (2 * Math.PI * clock * p.breathSpeed) / c.loop;
    const heldPhase = (2 * Math.PI * (clock - t + c.t1) * p.breathSpeed) / c.loop;
    if (!smoothWobble.length || smoothWobbleSeed !== p.seed) {
      smoothWobble = buildSmoothWobble(p.seed);
      smoothWobbleSeed = p.seed;
    }
    const w = { x: 0, y: 0 };
    const heldW = { x: 0, y: 0 };
    for (let k = 0; k < lat.nodes; k++) {
      const rx = lat.restX[k];
      const ry = lat.restY[k];
      let bx = rx;
      let by = ry;
      if (p.breath > 0) {
        // G's image warp breathes on plane waves; F's wireframe breathes on
        // the per-node harmonics. Using F's here is what makes the footage
        // crawl instead of drape.
        if (p.algorithm === 'warp') smoothWobbleAt(smoothWobble, rx, ry, phase, w);
        else wobbleAt(rx, ry, lat.h, phase, w);
        if (holdWeight > 0) {
          if (p.algorithm === 'warp') smoothWobbleAt(smoothWobble, rx, ry, heldPhase, heldW);
          else wobbleAt(rx, ry, lat.h, heldPhase, heldW);
          bx += p.breath * lat.h * (w.x + holdWeight * (heldW.x - w.x));
          by += p.breath * lat.h * (w.y + holdWeight * (heldW.y - w.y));
          footageBreathX[k] = p.breath * lat.h * holdWeight * (w.x - heldW.x);
          footageBreathY[k] = p.breath * lat.h * holdWeight * (w.y - heldW.y);
        } else {
          bx += p.breath * lat.h * w.x;
          by += p.breath * lat.h * w.y;
          footageBreathX[k] = 0;
          footageBreathY[k] = 0;
        }
      } else {
        footageBreathX[k] = 0;
        footageBreathY[k] = 0;
      }
      if (amplitude !== 0) {
        const dx = sampleDisp(lat, solved.ux, bx - offsetX, by - offsetY);
        const dy = sampleDisp(lat, solved.uy, bx - offsetX, by - offsetY);
        posX[k] = bx + amplitude * dx;
        posY[k] = by + amplitude * dy;
        // Extension seam for the direction-4/5 experiments: a future
        // optional relief or contour-lock pass can adjust this local sample
        // before packing. Both experiments stay zero in the faithful-G path.
        nodeDisp[k] = Math.min(1, (Math.hypot(dx, dy) * Math.abs(amplitude)) / solved.maxDisp);
      } else {
        posX[k] = bx;
        posY[k] = by;
        nodeDisp[k] = 0;
      }
    }
    // the boundary ring keeps the overscan rectangle so the field stays covered
    for (let i = 0; i < cols; i++) {
      posY[i] = lat.y0;
      posY[(rows - 1) * cols + i] = lat.y1;
    }
    for (let j = 0; j < rows; j++) {
      posX[j * cols] = lat.x0;
      posX[j * cols + cols - 1] = lat.x1;
    }

    // 2. per-cell area ratio (det of the inverse map) and the fold test.
    // Keep the signed raw determinant and its clamped magnitude separately so
    // the browser can compare the probe field instead of hiding the clamp.
    const A0 = lat.hx * lat.hy;
    nodeRatio.fill(0);
    nodeRawRatio.fill(0);
    nodeCount.fill(0);
    let folds = 0;
    let bdCells = 0;
    for (let j = 0; j < lat.ny; j++) {
      for (let i = 0; i < lat.nx; i++) {
        const a = j * cols + i;
        const b = a + 1;
        const cc = a + cols + 1;
        const d = a + cols;
        const area =
          0.5 *
          (posX[a] * posY[b] - posX[b] * posY[a] + posX[b] * posY[cc] - posX[cc] * posY[b] + posX[cc] * posY[d] - posX[d] * posY[cc] + posX[d] * posY[a] - posX[a] * posY[d]);
        const raw = A0 / (Math.abs(area) < 1e-6 ? (area < 0 ? -1e-6 : 1e-6) : area);
        let ratio = Math.abs(raw);
        if (ratio < DET_LO) ratio = DET_LO;
        if (ratio > DET_HI) ratio = DET_HI;
        nodeRawRatio[a] += raw;
        nodeRawRatio[b] += raw;
        nodeRawRatio[cc] += raw;
        nodeRawRatio[d] += raw;
        nodeRatio[a] += ratio;
        nodeRatio[b] += ratio;
        nodeRatio[cc] += ratio;
        nodeRatio[d] += ratio;
        nodeCount[a]++;
        nodeCount[b]++;
        nodeCount[cc]++;
        nodeCount[d]++;
        // a cell is sound if EITHER diagonal splits it into two positively
        // oriented triangles; requiring convexity flags every non-convex quad
        const cross = (p0: number, p1: number, p2: number) =>
          (posX[p1] - posX[p0]) * (posY[p2] - posY[p0]) - (posY[p1] - posY[p0]) * (posX[p2] - posX[p0]);
        const okAC = cross(a, b, cc) > 0 && cross(a, cc, d) > 0;
        const okBD = cross(a, b, d) > 0 && cross(b, cc, d) > 0;
        // If both diagonals are valid, choose the split with the healthier
        // pair of triangles. This keeps a non-convex but renderable quad from
        // being sent through the fixed AC split used by the old port.
        const triQuality = (p0: number, p1: number, p2: number, p3: number) => Math.min(Math.abs(cross(p0, p1, p2)), Math.abs(cross(p0, p2, p3)));
        const qualityAC = okAC ? triQuality(a, b, cc, d) : -1;
        const qualityBD = okBD ? Math.min(Math.abs(cross(a, b, d)), Math.abs(cross(b, cc, d))) : -1;
        const useBD = qualityBD > qualityAC;
        const bad = okAC || okBD ? 0 : 1;
        cellFold[j * lat.nx + i] = bad;
        cellDiagonal[j * lat.nx + i] = useBD ? 1 : 0;
        folds += bad;
        bdCells += useBD ? 1 : 0;
      }
    }
    foldRate = folds / (lat.nx * lat.ny);
    invalidRate = foldRate;
    diagonalBdRate = bdCells / (lat.nx * lat.ny);
    for (let k = 0; k < lat.nodes; k++) {
      nodeRatio[k] = nodeCount[k] > 0 ? nodeRatio[k] / nodeCount[k] : 1;
      nodeRawRatio[k] = nodeCount[k] > 0 ? nodeRawRatio[k] / nodeCount[k] : 1;
    }

    // 3. Laplacian smoothing is a lattice approximation to the probe's
    // screen-space Gaussian. The control is expressed in screen pixels; the
    // status/report exposes the resulting lattice pass count explicitly.
    const smoothingPx = Math.max(0, p.smoothingPx ?? p.smoothing * Math.max(1, lat.h));
    const passes = Math.max(0, Math.round(smoothingPx / Math.max(1, lat.h * 0.5)));
    smoothingPasses = passes;
    for (let pass = 0; pass < passes; pass++) {
      for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
          const k = j * cols + i;
          let sum = 0;
          let n = 0;
          if (i > 0) { sum += nodeRatio[k - 1]; n++; }
          if (i < cols - 1) { sum += nodeRatio[k + 1]; n++; }
          if (j > 0) { sum += nodeRatio[k - cols]; n++; }
          if (j < rows - 1) { sum += nodeRatio[k + cols]; n++; }
          nodeRatioB[k] = n > 0 ? 0.5 * nodeRatio[k] + 0.5 * (sum / n) : nodeRatio[k];
        }
      }
      nodeRatio.set(nodeRatioB);
    }

    nodeClampRatio.set(nodeRatio);

    // 4. normalise by the median, as the probe does
    const stride = Math.max(1, Math.floor(lat.nodes / 1024));
    const sample: number[] = [];
    for (let k = 0; k < lat.nodes; k += stride) sample.push(nodeRatio[k]);
    sample.sort((x, y) => x - y);
    const median = sample.length ? sample[sample.length >> 1] : 1;
    medianRatio = median;
    const inv = 1 / Math.max(1e-6, median);
    const jumble = p.footage === 'jumble';
    if (jumble && p.jumbleShading === 'off') strength = 0;
    const signed = signedExponent(p.polarity, strength);

    // 5. pack the vertex buffer. Vertices are NOT shared between cells, so the
    //    per-cell clip index cannot bleed; the shared node position is copied
    //    into each duplicate, so the mesh stays watertight.
    const pal = palette;
    const data = vertexData;
    let o = 0;
    const invW = 1 / CANVAS_W;
    const invH = 1 / CANVAS_H;
    // Release runs through the same distance thresholds as the incoming push.
    // The asymmetric amplitude fall keeps the full front inside its short
    // 1.35-second window on the default three-second release.
    const releaseProgress = t >= c.t2 ? 1 - amplitudeAt(t, c, p.releaseShape) : 0;
    for (let j = 0; j < lat.ny; j++) {
      for (let i = 0; i < lat.nx; i++) {
        const cell = j * lat.nx + i;
        const a = j * cols + i;
        const b = a + 1;
        const cc = a + cols + 1;
        const d = a + cols;
        let clip = p.clip;
        let nextClip = clip;
        let transition = 0;
        let coreBlend = false;
        if (jumble && pal) {
          clip = p.commit ? (amplitude >= pal.tau[cell] ? pal.fam[cell] : pal.idle[cell]) : pal.mix[cell];
          nextClip = clip;
          if (p.commit && p.commitTransition !== 'cut' && pal.idle[cell] !== pal.fam[cell]) {
            clip = pal.idle[cell];
            nextClip = pal.fam[cell];
            if (p.commitTransition === 'push') {
              coreBlend = pal.core[cell] === 1;
              const distance = solved.cellDist[cell];
              const far = Math.max(0, Math.min(1, distance / (CANVAS_H * 0.58)));
              const interior = Math.max(0, Math.min(1, -distance / (CANVAS_H * 0.18)));
              const jitter = (pal.rTau[cell] - 0.5) * 0.04;
              const start = Math.max(0.02, Math.min(0.95, 0.77 - 0.75 * far + 0.18 * interior + jitter));
              const width = Math.min(0.23, 1 - start);
              // The return front travels from distant background to the
              // contour, then through the core. Every cell is back at idle
              // by the time the field's release amplitude reaches zero.
              const returnFactor = 1 - smoothUnit((releaseProgress - start) / width);
              if (coreBlend) {
                // Prepare the core before the deformation becomes readable.
                // Its release is last, preserving the readable word while the
                // surrounding collage returns by distance.
                const prepare = Math.min(0.8, Math.max(0.2, (c.t1 - c.t0) * 0.18));
                transition = smoothUnit((t - c.t0) / prepare) * returnFactor;
              } else {
                // An inward-moving front follows the force field's amplitude.
                // Distant tiles leave first; the last cells beside the stable
                // core finish just before the field reaches full strength.
                const enter = smoothUnit((amplitude - start) / width);
                transition = (p.amplitudeAuto && t >= c.t2 ? 1 : enter) * returnFactor;
              }
            } else {
              transition = smoothUnit((amplitude - pal.tau[cell] + 0.14) / 0.28);
            }
          }
        }
        const fieldX = sampleDisp(lat, solved.ux, lat.x0 + (i + 0.5) * lat.hx - offsetX, lat.y0 + (j + 0.5) * lat.hy - offsetY);
        const fieldY = sampleDisp(lat, solved.uy, lat.x0 + (i + 0.5) * lat.hx - offsetX, lat.y0 + (j + 0.5) * lat.hy - offsetY);
        // Out in the still part of the field, continue the motion away from
        // the word so every old tile can leave its cell.
        const radialX = lat.x0 + (i + 0.5) * lat.hx - CANVAS_W * 0.5 - offsetX;
        const radialY = lat.y0 + (j + 0.5) * lat.hy - CANVAS_H * 0.5 - offsetY;
        const direction = coreBlend ? 10 : Math.hypot(fieldX, fieldY) > lat.h * 0.12
          ? Math.atan2(fieldY, fieldX)
          : Math.atan2(radialY, radialX);
        const fold = cellFold[cell];
        const quad = cellDiagonal[cell] === 1
          ? [a, b, d, b, cc, d]
          : [a, b, cc, a, cc, d];
        for (let v = 0; v < 6; v++) {
          const k = quad[v];
          data[o] = posX[k];
          data[o + 1] = posY[k];
          data[o + 2] = (lat.restX[k] + footageBreathX[k]) * invW;
          data[o + 3] = (lat.restY[k] + footageBreathY[k]) * invH;
          data[o + 4] = nodeRatio[k] * inv;
          data[o + 5] = clip;
          data[o + 6] = nodeDisp[k];
          data[o + 7] = fold;
          data[o + 8] = nodeRawRatio[k];
          data[o + 9] = nodeClampRatio[k];
          data[o + 10] = Math.pow(Math.max(0.05, nodeRatio[k] * inv), signed);
          data[o + 11] = nextClip;
          data[o + 12] = transition;
          data[o + 13] = k === a || k === d ? 0 : 1;
          data[o + 14] = k === a || k === b ? 0 : 1;
          data[o + 15] = direction;
          data[o + 16] = lat.restX[k] * invW;
          data[o + 17] = lat.restY[k] * invH;
          o += 18;
        }
      }
    }

    // 6. the wireframe buffer, every lattice edge, one colour
    if (p.view === 1) {
      let l = 0;
      for (let j = 0; j < rows; j++) {
        for (let i = 0; i < lat.nx; i++) {
          const k = j * cols + i;
          lineData[l] = posX[k];
          lineData[l + 1] = posY[k];
          lineData[l + 2] = posX[k + 1];
          lineData[l + 3] = posY[k + 1];
          l += 4;
        }
      }
      for (let j = 0; j < lat.ny; j++) {
        for (let i = 0; i < cols; i++) {
          const k = j * cols + i;
          lineData[l] = posX[k];
          lineData[l + 1] = posY[k];
          lineData[l + 2] = posX[k + cols];
          lineData[l + 3] = posY[k + cols];
          l += 4;
        }
      }
      let finiteLines = 0;
      let minLine = Infinity;
      let maxLine = -Infinity;
      for (let li = 0; li < lineData.length; li++) {
        if (Number.isFinite(lineData[li])) finiteLines++;
        minLine = Math.min(minLine, lineData[li]);
        maxLine = Math.max(maxLine, lineData[li]);
      }
      note = `view 1 · lines ${finiteLines}/${lineData.length} · ${minLine.toFixed(0)}…${maxLine.toFixed(0)}`;
    }

    if (debugCanvas && debugContext) drawDebugOverlay(debugCanvas, debugContext, p.view, lineData, data);

    lastStrength = strength;

    if (p.view === 1) {
      canvas.style.background = '#fafaf7';
      gl!.clear(0.98, 0.98, 0.972);
      gl!.drawLines(lineData, lineData.length / 2);
    } else {
      canvas.style.background = '#0b0d10';
      gl!.clear(0.043, 0.051, 0.063);
      if (p.view === 0) gl!.uploadVideos();
      gl!.drawMesh(data, lat.nx * lat.ny * 6, p.view, signed, strength > 0, p.lumaAdaptive, median, gl!.screenField, [lat.hx / CANVAS_W, lat.hy / CANVAS_H], p.commitTransition === 'push' ? 2 : p.commitTransition === 'squeeze' ? 1 : 0);
    }
  }

  /** Strength for a manually held amplitude: the schedule is a function of
   *  amplitude, so the manual override still gets the right flare. */
  function strengthWithAmplitude(p: MeshParams, a: number): number {
    const base = p.strength;
    const k = Math.max(0, p.flarePeak - base);
    if (p.schedule === 'const') return base;
    if (p.schedule === 'inverse') return 0.3 + 0.7 * a * a * a;
    return base + k * Math.exp(-(((a - p.flareCentre) / p.flareWidth) ** 2));
  }

  return {
    setParams(next) {
      params = { ...next };
      const key = solveKey(params);
      if (key !== (solved?.key ?? '')) {
        // The previous mesh stays on screen while the new one solves.
        if (!job || job.key !== key) beginSolve(params);
      } else if (job) {
        job = null;
      }
    },
    frame(clock) {
      const started = performance.now();
      measureLuma();
      if (job) advanceSolve(9);
      if (solved) {
        if (solved.spread !== params.spread) applySpread(solved, params.spread);
        if (!palette || lastPaletteKey !== paletteKey(params)) {
          palette = buildPalette(solved, params, luma);
          lastPaletteKey = paletteKey(params);
        } else if (Math.abs(palette.offsetX - params.offsetX) > 0.5 || Math.abs(palette.offsetY - params.offsetY) > 0.5) {
          refreshPalette(palette, solved, params, luma, params.offsetX, params.offsetY);
        }
      }
      if (solved && visible) buildFrame(clock);
      frameMs = performance.now() - started;
    },
    status() {
      const lat = solved?.lat;
      const stroke = solved?.glyph.strokePx ?? 0;
      const cellSize = lat ? lat.h : CANVAS_W / params.columns;
      const c = cycleOf(params);
      return {
        solving: job !== null,
        progress: job ? (job.stage === 'glyph' ? 0.02 : job.settle ? job.settle.done / job.settle.iterations : job.warp ? job.warp.done / job.warp.iterations : 0.99) : 1,
        cellsPerStroke: cellSize > 0 ? stroke / cellSize : 0,
        cellSize,
        columns: lat?.columns ?? params.columns,
        rows: lat?.rows ?? Math.round(CANVAS_H / cellSize),
        strokePx: stroke,
        fontPx: solved?.glyph.fontPx ?? 0,
        constraints: lastConstraints,
        amplitude: lastAmplitude,
        strength: lastStrength,
        time: lastTime,
        loop: c.loop,
        foldRate,
        invalidRate,
        diagonalBdRate,
        medianRatio,
        smoothingPasses,
        screenField: gl!.screenField,
        frameMs,
        solveMs,
        vertices: lat ? lat.nx * lat.ny * 6 : 0,
        note,
      };
    },
    solveNow() {
      let guard = 0;
      while (job && guard++ < 4000) advanceSolve(1000);
      if (solved) {
        palette = buildPalette(solved, params, luma);
        lastPaletteKey = paletteKey(params);
      }
    },
    setVisible(next) {
      visible = next;
      if (debugCanvas) debugCanvas.hidden = !next;
      if (next) note = '';
    },
  };
}
