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

/** scripts/bake-showcase-cloth.py, once per project: a Blender cloth on a
 *  floor with holes the shape of the word, pulled in by the letters falling
 *  through, seen from above. For each sample frame and each point of a screen
 *  grid, the offset in canvas px from the point to the material point of the
 *  cloth visible there, top layer first, and the cloth's height. */
type ClothBake = {
  /** The word exactly as the page sets it, and the size its glyph solved at. */
  word: string;
  fontPx: number;
  /** The fall it was baked with. `depth` is how far the letters go below the
   *  floor, in Blender units: the wipe's reference for a full-depth letter. */
  fall: { depth: number };
  frames: number[];
  grid: { x0: number; y0: number; step: number; width: number; height: number };
  offsetScale: number;
  heightScale: number;
  data: string;
};

const CLOTH_BAKES = new Map<string, ClothBake>();
for (const module of Object.values(import.meta.glob<{ default: ClothBake }>('../data/showcase-cloth/*.json', { eager: true }))) {
  CLOTH_BAKES.set(module.default.word, module.default);
}
/** The cloth wipe's own falls (bake-showcase-cloth.py --pour): the letters
 *  thrown through the floor and the sheet pouring down the holes after them.
 *  Keyed apart from the flow's, which still rides the bakes above. */
const POUR_PREFIX = 'pour:';
for (const module of Object.values(import.meta.glob<{ default: ClothBake }>('../data/showcase-cloth-pour/*.json', { eager: true }))) {
  CLOTH_BAKES.set(POUR_PREFIX + module.default.word, module.default);
}
const clothGrids = new Map<string, Int16Array>();

/** bake-showcase-cloth.py --mesh: the cloth itself, for the mesh wipe to draw
 *  in 3D rather than read off a raster. Each point's rest position, which is
 *  also where the clip is sampled on it; the triangles; and every exported
 *  frame's positions. With the floor's hole outlines, in canvas px. */
type ClothMeshBake = {
  word: string;
  points: number;
  triangles: number;
  frames: number[];
  offsetScale: number;
  heightScale: number;
  floorDepth: number;
  sheetZ: number;
  holes: number[][][];
  data: string;
  /** --lift bakes only: the frames the letters stop growing and start to
   *  slide away, which the page maps to its hold and its release. */
  lift?: { growEnd: number; slideStart: number } | null;
  /** compact-showcase-cloth.py's output: zlib-compressed, the frames stored
   *  as residuals from a linear prediction. */
  encoding?: string;
};
const CLOTH_MESHES = new Map<string, ClothMeshBake>();
for (const module of Object.values(import.meta.glob<{ default: ClothMeshBake }>('../data/showcase-cloth-mesh/*.json', { eager: true }))) {
  CLOTH_MESHES.set(module.default.word, module.default);
}
/** bake-showcase-cloth.py --lift: letters rising under the cloth and then
 *  sliding away with it. Keyed apart from the falls. */
const LIFT_PREFIX = 'lift:';
for (const module of Object.values(import.meta.glob<{ default: ClothMeshBake }>('../data/showcase-cloth-lift/*.json', { eager: true }))) {
  CLOTH_MESHES.set(LIFT_PREFIX + module.default.word, module.default);
}
/** The lifts as baked, before compact-showcase-cloth.py, for the test page to
 *  compare against (?full=1). */
const LIFT_FULL_PREFIX = 'liftfull:';
for (const module of Object.values(import.meta.glob<{ default: ClothMeshBake }>('../data/showcase-cloth-lift-full/*.json', { eager: true }))) {
  CLOTH_MESHES.set(LIFT_FULL_PREFIX + module.default.word, module.default);
}

/** A compact bake's bytes back into the plain layout: rest positions, the
 *  triangles, and every frame's positions point by point. */
async function decodeClothMesh(bake: ClothMeshBake, packed: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream('deflate'));
  const raw = await new Response(stream).arrayBuffer();
  const P = bake.points;
  const F = bake.frames.length;
  const head = P * 4 + bake.triangles * 6;
  const residuals = new Int16Array(raw, head, F * 3 * P);
  const out = new ArrayBuffer(head + F * P * 6);
  new Uint8Array(out, 0, head).set(new Uint8Array(raw, 0, head));
  const frames = new Int16Array(out, head, F * P * 3);
  for (let f = 0; f < F; f++) {
    for (let c = 0; c < 3; c++) {
      const r = (f * 3 + c) * P;
      for (let k = 0; k < P; k++) {
        const at = (f * P + k) * 3 + c;
        const back1 = f > 0 ? frames[at - P * 3] : 0;
        const back2 = f > 1 ? frames[at - 2 * P * 3] : 0;
        frames[at] = residuals[r + k] + (f > 1 ? 2 * back1 - back2 : back1);
      }
    }
  }
  return out;
}
type ClothMesh = { bake: ClothMeshBake; rest: Int16Array; tris: Uint16Array; frames: Int16Array };
const clothMeshes = new Map<string, ClothMesh>();
const clothMeshRequested = new Set<string>();

/** A word's cloth mesh, fetched on first use; null until it arrives. */
function clothMeshFor(word: string): ClothMesh | null {
  const bake = CLOTH_MESHES.get(word);
  if (!bake) return null;
  if (!clothMeshRequested.has(word) && typeof fetch === 'function') {
    clothMeshRequested.add(word);
    fetch(`/${bake.data}`)
      .then((response) => (response.ok ? response.arrayBuffer() : Promise.reject(new Error(`${response.status}`))))
      .then((packed) => (bake.encoding ? decodeClothMesh(bake, packed) : packed))
      .then((buffer) => {
        const restBytes = bake.points * 4;
        const triBytes = bake.triangles * 6;
        const frameValues = bake.frames.length * bake.points * 3;
        if (buffer.byteLength !== restBytes + triBytes + frameValues * 2) throw new Error('size mismatch');
        clothMeshes.set(word, {
          bake,
          rest: new Int16Array(buffer, 0, bake.points * 2),
          tris: new Uint16Array(buffer, restBytes, bake.triangles * 3),
          frames: new Int16Array(buffer, restBytes + triBytes, frameValues),
        });
      })
      .catch((error) => console.warn(`${word} cloth mesh unavailable:`, error));
  }
  return clothMeshes.get(word) ?? null;
}

/** The word's own glyph as the floor's holes, for the mesh wipe to draw: the
 *  same field as holeField, from the glyph's distance, so the letters keep
 *  their real corners and a crisp edge. The bake's holes are wider and
 *  rounded for the cloth's sake; the floor hides the cloth lining them. */
function glyphHoleField(glyph: Glyph): { bytes: Uint8Array; deepest: number } {
  const w = CANVAS_W;
  const h = CANVAS_H;
  const bytes = new Uint8Array(w * h);
  let deepest = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const f = -sampleField(glyph.sdf, glyph.w, glyph.h, x + 0.5, y + 0.5);
      deepest = Math.max(deepest, f);
      bytes[y * w + x] = Math.max(0, Math.min(255, Math.round((f + 1) * 2)));
    }
  }
  return { bytes, deepest };
}

/** The floor's holes as a field the mesh wipe can heal: in half px, 1 px
 *  outside a hole's edge at 0, its antialiased edge between, and inside it
 *  the distance to the edge. Raising a threshold through it closes every
 *  hole from its edges inward. Returns the bytes and the deepest point. */
function holeField(loops: number[][][]): { bytes: Uint8Array; deepest: number } {
  const w = CANVAS_W;
  const h = CANVAS_H;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext('2d')!;
  g.fillStyle = '#fff';
  g.beginPath();
  for (const loop of loops) {
    loop.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.closePath();
  }
  g.fill('nonzero');
  const pixels = g.getImageData(0, 0, w, h).data;
  // chamfer distance, inside the holes, to the nearest pixel not wholly in one
  const d = new Float32Array(w * h);
  for (let k = 0; k < w * h; k++) d[k] = pixels[4 * k] >= 255 ? 1e6 : 0;
  const pass = (x: number, y: number, dx: number, dy: number) => {
    const k = y * w + x;
    if (d[k] === 0) return;
    let best = d[k];
    const near = (xx: number, yy: number, cost: number) => {
      if (xx >= 0 && xx < w && yy >= 0 && yy < h) best = Math.min(best, d[yy * w + xx] + cost);
    };
    near(x - dx, y, 1);
    near(x, y - dy, 1);
    near(x - dx, y - dy, Math.SQRT2);
    near(x + dx, y - dy, Math.SQRT2);
    d[k] = best;
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pass(x, y, 1, 1);
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) pass(x, y, -1, -1);
  const bytes = new Uint8Array(w * h);
  let deepest = 0;
  for (let k = 0; k < w * h; k++) {
    const f = d[k] > 0 ? d[k] - 0.5 : pixels[4 * k] / 255 - 0.5;
    deepest = Math.max(deepest, f);
    bytes[k] = Math.max(0, Math.min(255, Math.round((f + 1) * 2)));
  }
  return { bytes, deepest };
}
const clothRequested = new Set<string>();

/** Whether a word has a cloth bake at all. */
export function hasClothBake(word: string): boolean {
  return CLOTH_BAKES.has(word);
}

/** A word's bake and its int16 grids, fetched on first use. It applies only
 *  at the size it was baked for, since it is fitted to that glyph, except to
 *  the cloth wipe, where the bake IS the word and there is no glyph to match.
 *  Until the grids arrive, and if they never do, the flood uses the plain
 *  frontier and the wipe draws its clip flat. */
function clothFor(word: string, fontPx?: number): { bake: ClothBake; grids: Int16Array } | null {
  const bake = CLOTH_BAKES.get(word);

  if (!bake || (fontPx !== undefined && Math.abs(bake.fontPx - fontPx) > 0.5)) return null;
  if (!clothRequested.has(word) && typeof fetch === 'function') {
    clothRequested.add(word);
    const { frames, grid } = bake;
    fetch(`/${bake.data}`)
      .then((response) => (response.ok ? response.arrayBuffer() : Promise.reject(new Error(`${response.status}`))))
      .then((buffer) => {
        // int16 little-endian, which is every browser's native order
        const grids = new Int16Array(buffer);
        if (grids.length !== frames.length * 3 * grid.width * grid.height) throw new Error('size mismatch');
        clothGrids.set(word, grids);
      })
      .catch((error) => console.warn(`${word} cloth bake unavailable:`, error));
  }
  const grids = clothGrids.get(word);
  return grids ? { bake, grids } : null;
}

/** The cloth at canvas point (x, y) and formation progress in [0, 1]:
 *  bilinear in space, linear between sample frames. `out.x`/`out.y` point
 *  from (x, y) to the material now visible there, in canvas px; `out.z` is
 *  its height in Blender units, negative where the letters pulled it down. */
function sampleCloth(cloth: { bake: ClothBake; grids: Int16Array }, x: number, y: number, progress: number, out: { x: number; y: number; z: number }) {
  const { frames, grid, offsetScale, heightScale } = cloth.bake;
  const data = cloth.grids;
  const size = grid.width * grid.height;
  const frame = Math.max(0, Math.min(frames.length - 1, progress * (frames.length - 1)));
  const f0 = Math.floor(frame);
  const f1 = Math.min(frames.length - 1, f0 + 1);
  const ft = frame - f0;
  const u = Math.max(0, Math.min(grid.width - 1, (x - grid.x0) / grid.step));
  const v = Math.max(0, Math.min(grid.height - 1, (y - grid.y0) / grid.step));
  const i0 = Math.min(grid.width - 2, Math.floor(u));
  const j0 = Math.min(grid.height - 2, Math.floor(v));
  const tx = u - i0;
  const ty = v - j0;
  const k = j0 * grid.width + i0;
  const at = (base: number) =>
    (data[base + k] * (1 - tx) + data[base + k + 1] * tx) * (1 - ty) +
    (data[base + k + grid.width] * (1 - tx) + data[base + k + grid.width + 1] * tx) * ty;
  const channel = (c: number) => at((f0 * 3 + c) * size) * (1 - ft) + at((f1 * 3 + c) * size) * ft;
  out.x = channel(0) * offsetScale;
  out.y = channel(1) * offsetScale;
  out.z = channel(2) * heightScale;
}

/** Where the sheet has left a grid point, the bake writes a marker offset:
 *  the point's own distance from the centre of the frame again, so the flood
 *  counts it as far from the word. */
function clothUncovered(wx: number, wy: number, x: number, y: number): boolean {
  return Math.abs(wx - (x - CANVAS_W / 2)) < 0.2 && Math.abs(wy - (y - CANVAS_H / 2)) < 0.2;
}

const clothWipes = new Map<string, Float32Array>();
/** Grids a frame in the wipe's field: drag x, drag y, height, coverage. */
const CLOTH_WIPE_GRIDS = 4;

/** The bake as one continuous field, in canvas px: the same grids, with every
 *  marker point filled in from its nearest covered neighbours, and a fourth
 *  grid a frame saying whether the sheet is there at all (1) or has left (0).
 *  The wipe paints the clip on the fabric, so it needs a material point
 *  everywhere, and the coverage says where to show the floor instead. Built
 *  once a word. */
function clothWipeFields(word: string, cloth: { bake: ClothBake; grids: Int16Array }): Float32Array {
  const held = clothWipes.get(word);
  if (held) return held;
  const { frames, grid, offsetScale, heightScale } = cloth.bake;
  const size = grid.width * grid.height;
  const field = new Float32Array(frames.length * CLOTH_WIPE_GRIDS * size);
  const known = new Uint8Array(size);
  const queue = new Int32Array(size);
  for (let f = 0; f < frames.length; f++) {
    const base = f * CLOTH_WIPE_GRIDS * size;
    const bakeBase = f * 3 * size;
    known.fill(1);
    let unknown = 0;
    for (let j = 0; j < grid.height; j++) {
      for (let i = 0; i < grid.width; i++) {
        const k = j * grid.width + i;
        const wx = cloth.grids[bakeBase + k] * offsetScale;
        const wy = cloth.grids[bakeBase + size + k] * offsetScale;
        field[base + k] = wx;
        field[base + size + k] = wy;
        field[base + 2 * size + k] = cloth.grids[bakeBase + 2 * size + k] * heightScale;
        field[base + 3 * size + k] = 1;
        if (clothUncovered(wx, wy, grid.x0 + i * grid.step, grid.y0 + j * grid.step)) {
          known[k] = 0;
          field[base + 3 * size + k] = 0;
          unknown++;
        }
      }
    }
    // Grow the covered field outward a ring at a time: each point still empty
    // with covered neighbours takes their mean, and joins them.
    for (let guard = 0; unknown > 0 && guard < grid.width + grid.height; guard++) {
      let filled = 0;
      for (let j = 0; j < grid.height; j++) {
        for (let i = 0; i < grid.width; i++) {
          const k = j * grid.width + i;
          if (known[k]) continue;
          let n = 0;
          let sx = 0;
          let sy = 0;
          let sz = 0;
          const add = (m: number) => {
            if (known[m] !== 1) return;
            sx += field[base + m];
            sy += field[base + size + m];
            sz += field[base + 2 * size + m];
            n++;
          };
          if (i > 0) add(k - 1);
          if (i < grid.width - 1) add(k + 1);
          if (j > 0) add(k - grid.width);
          if (j < grid.height - 1) add(k + grid.width);
          if (n === 0) continue;
          field[base + k] = sx / n;
          field[base + size + k] = sy / n;
          field[base + 2 * size + k] = sz / n;
          queue[filled++] = k;
        }
      }
      if (filled === 0) break;
      for (let q = 0; q < filled; q++) known[queue[q]] = 1;
      unknown -= filled;
    }
  }
  clothWipes.set(word, field);
  return field;
}

export type Algorithm = 'settle' | 'warp';
/** The two correspondence families are kept separate so the late-G reference
 * cannot silently replace the original offset experiment. */
export type WarpCorrespondence = 'offset' | 'late-g-lens';
export type ReleaseShape = 'disperse' | 'rewind' | 'asymmetric';
export type Schedule = 'const' | 'overshoot' | 'inverse' | 'lead' | 'lag';
/** Signed exponent meaning is explicit in the UI and in the shader. The old
 * names remain accepted for saved links made by the first port. */
export type Polarity = 'compression-bright' | 'compression-dark' | 'positive' | 'negative' | 'debossed' | 'embossed';
/** `territories` is prototype 3, revised: three broad footage regions whose
 *  borders breathe, with the featured project's region growing to dominate
 *  while only its own cells take the field.
 *
 *  `cloth` is the simplified opening: one clip, no collage and no force
 *  field at all. The project's clip is painted on the Blender cloth of its
 *  own bake, and the fall writes the word into it — the fabric drags the
 *  picture into the letter-shaped holes, and the sunken letters read as
 *  relief. The bake is the only geometry, so nothing is solved. */
export type FootageMode = 'single' | 'jumble' | 'territories' | 'cloth' | 'clothmesh' | 'clothlift';
export type IdleField = 'jumble' | 'single';
/** `flow` never switches a cell: the featured footage floods in from far to
 *  near and pushes the collage ahead of it, squeezing it into the letters,
 *  and release lets it flow back out. */
export type CommitTransition = 'cut' | 'squeeze' | 'push' | 'flow';
/** How the flow's frontier uses the project's cloth bake. `height` is the first
 *  mapping, kept for A/B: the sheet's height bends the distance frontier.
 *  `frontier` reads the distance at the material point of the sheet visible
 *  at each point, so the frontier rides the fabric. `collage` also carries
 *  the moving collage with the fabric. */
export type ClothFrontier = 'off' | 'height' | 'frontier' | 'collage';
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
  clothFrontier: ClothFrontier;
  /** The cloth wipe only: how dark a letter goes at the bottom of its hole.
   *  0 leaves the word to the fabric's own squeeze alone. */
  clothRelief: number;
  /** The cloth wipe only: the clip lying under the fabric, which the fall
   *  uncovers as it drags `homeClip` into the letters and pulls it through.
   *  In the sequence it is the project after this one. -1 uses `homeClip`
   *  itself, so the wipe only warps and lights the one clip. */
  clothUnder: number;
  /** The flow only: the formed word drops out of the middle of the frame at
   *  the hold, bounces, and rises back as it releases. */
  holdDrop: boolean;
  /** Restitution of the drop's bounces: the share of speed a bounce keeps. */
  dropBounce: number;
  idleField: IdleField;
  stagger: number;
  clip: number;
  familySplit: boolean;
  familyMode: FamilyMode;
  /** Resolved ground clip for the origin split. The panel sets it from the
   *  Clip select, or from the word when the mode is `origin-auto`. */
  homeClip: number;
  seed: number;
  // territories
  /** Clips in sequence order. The strip of regions repeats them, and the
   *  project shown `sequenceIndex`-th features ring[sequenceIndex % length]. */
  territoryRing: number[];
  /** Regions each project holds across the frame: 1 is prototype 3's layout.
   *  More interleave the projects in narrower bands, with the featured
   *  project's bands widening together. */
  territoryRegions: number;
  /** How far each border moves outward at full amplitude, as a fraction of
   *  the canvas width. 0.25 carries the featured region from 40% to 90%. */
  territoryGrowth: number;
  /** Scale on the borders' breathing; 1 is the prototype's swell of about
   *  40 px. */
  territoryBreath: number;
  /** Release hands the frame to the next project instead of shrinking back:
   *  the right region sweeps in to take the middle, pushing the featured one
   *  into the left slot, while the entering one arrives from the right edge.
   *  The cycle ends in the next project's opening layout. */
  territoryHandover: boolean;
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
  clothFrontier: 'frontier',
  clothRelief: 0.7,
  clothUnder: -1,
  holdDrop: false,
  dropBounce: 0.45,
  idleField: 'jumble',
  stagger: 0.72,
  clip: 0,
  familySplit: true,
  familyMode: 'luminance',
  homeClip: 0,
  seed: 11,
  territoryRing: [0, 1, 2],
  territoryRegions: 1,
  territoryGrowth: 0.25,
  territoryBreath: 1,
  territoryHandover: false,
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
/** A line break stacks the word. Prototype 3 set GlyphInterpreter as two
 *  170 px lines 186 px apart, which keeps its stroke above the density floor
 *  that one long line falls through. */
const LINE_GAP_EM = 186 / 170;

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
  const canvas = typeof document === 'undefined' ? new OffscreenCanvas(fw, fh) : document.createElement('canvas');
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
  const lines = (text.length ? text : ' ').split('\n').map((line) => (line.length ? line : ' '));
  const measureLine = (line: string, sizePx: number) => {
    const m = ctx.measureText(line);
    const left = m.actualBoundingBoxLeft ?? 0;
    const ascent = m.actualBoundingBoxAscent ?? sizePx * 0.72;
    const w = left + (m.actualBoundingBoxRight ?? m.width);
    const h = ascent + (m.actualBoundingBoxDescent ?? 0);
    return { text: line, left, ascent, w: w > 0 ? w : m.width, h: h > 0 ? h : sizePx * 0.72 };
  };
  /** Ink box of the whole word at a given size, in FIELD px. Each line is
   *  centred on its own ink box, one line gap below the last. */
  const inkBox = (sizePx: number) => {
    ctx.font = face(sizePx);
    const ink = lines.map((line) => measureLine(line, sizePx));
    const gap = LINE_GAP_EM * sizePx;
    const w = Math.max(...ink.map((line) => line.w));
    const h = (ink.length - 1) * gap + (ink[0].h + ink[ink.length - 1].h) / 2;
    return { ink, gap, w, h };
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
  // Shrink to fit if the word runs off the canvas, the same rule the probe used.
  const maxW = CANVAS_W * SDF_SCALE * 0.9;
  let block = inkBox(px);
  if (block.w > maxW) {
    px *= maxW / block.w;
    block = inkBox(px);
  }
  const letterCenters: Array<{ x: number; y: number }> = [];
  block.ink.forEach((line, index) => {
    const centreY = fh / 2 + (index - (block.ink.length - 1) / 2) * block.gap;
    const drawX = (fw - line.w) / 2 + line.left;
    const drawY = centreY - line.h / 2 + line.ascent;
    ctx.fillText(line.text, drawX, drawY);
    let cursor = drawX;
    for (const char of line.text) {
      const width = ctx.measureText(char).width;
      letterCenters.push({ x: cursor + width * 0.5, y: drawY - line.ascent * 0.5 });
      cursor += width;
    }
  });
  const inkW = block.w;
  const inkH = block.h;

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
    fontPx: px / SDF_SCALE,
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

export type MeshSolveGeometry = Pick<Solved, 'lat' | 'glyph' | 'ux0' | 'uy0' | 'nodeDist'>;

/** Run a project's geometry solve away from the animation frame. */
export function solveMeshGeometry(p: MeshParams, onProgress: (progress: number) => void): MeshSolveGeometry {
  const glyph = buildGlyph(p.text, p.fontSize, p.fontAuto);
  const lat = buildLattice(p.columns);
  const ux0 = new Float32Array(lat.nodes);
  const uy0 = new Float32Array(lat.nodes);
  if (p.algorithm === 'settle') {
    const s = createSettle(lat, glyph, p.gain, p.band);
    const rampN = Math.max(1, Math.round(s.iterations * CFG.rampFrac));
    while (s.done < s.iterations) {
      let a = Math.min(1, s.done / rampN);
      a = a * a * (3 - 2 * a);
      let cool = 1;
      const c0 = 1 - CFG.coolFrac;
      if (s.done > s.iterations * c0) cool = 1 - (0.85 * (s.done - s.iterations * c0)) / (s.iterations * CFG.coolFrac);
      settleStep(s, a, CFG.dt * cool);
      s.done++;
      if (s.done % 48 === 0) onProgress(s.done / s.iterations);
    }
    for (let k = 0; k < lat.nodes; k++) {
      ux0[k] = s.px[k] - lat.restX[k];
      uy0[k] = s.py[k] - lat.restY[k];
    }
  } else {
    const w = createWarp(lat, glyph, p.gain, p.warpCorrespondence);
    while (w.done < w.iterations) {
      warpStep(w, 12);
      w.done += 12;
      if (w.done % 96 === 0) onProgress(w.done / w.iterations);
    }
    for (let k = 0; k < lat.nodes; k++) {
      ux0[k] = w.ux[k];
      uy0[k] = w.uy[k];
    }
  }
  const nodeDist = new Float32Array(lat.nodes);
  for (let k = 0; k < lat.nodes; k++) {
    nodeDist[k] = sampleField(glyph.sdf, glyph.w, glyph.h, lat.restX[k], lat.restY[k]);
  }
  return { lat, glyph, ux0, uy0, nodeDist };
}

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
  const clipCount = luma.length;
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
    pal.mix[c] = Math.min(clipCount - 1, Math.floor(pal.rIdle[c] * clipCount));
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
  const clips = luma.map((_, index) => index);
  const home = Math.min(clips.length - 1, Math.max(0, Math.round(p.homeClip)));
  let outside: number[];
  let insideFam: number[];
  if (!p.familySplit) {
    outside = clips;
    insideFam = clips;
  } else if (p.familyMode !== 'luminance') {
    // The background commits to the clip the word is FOR, and every other clip
    // commits away out of it. The home clip has to be excluded from the inside
    // pool: an inside cell drawing the same clip as the ground behind it is
    // invisible, and those holes fall on the letters, which is the one place
    // the design cannot afford them.
    outside = [home];
    insideFam = clips.filter((clip) => clip !== home);
  } else {
    const split = Math.floor(order.length / 2);
    outside = order.slice(0, split).map((entry) => entry[1]);
    insideFam = order.slice(split).map((entry) => entry[1]);
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
      pal.idle[c] = p.idleField === 'single' ? p.clip : Math.min(clips.length - 1, Math.floor(pal.rIdle[c] * clips.length));
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
/* 7b. footage territories (prototype 3)                                   */
/* ---------------------------------------------------------------------- */

/** The field fades out across this many cells inside a border, so a node
 *  that a breathing border sweeps past does not pop between still and moved. */
const TERRITORY_MARGIN_CELLS = 2.5;
/** Successive pairs of borders bend out of step by the golden angle, so no
 *  two borders on screen together share a shape. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
/** Most regions a project can hold across the frame. */
export const TERRITORY_MAX_REGIONS = 4;
/** Borders a display can carry: every region of three projects, plus the one
 *  entering at a handover. MESH_FS sizes its border array from this. */
const MAX_BORDERS = 3 * TERRITORY_MAX_REGIONS + 1;
/** The furthest a border bends by its own shape at breath 1: the 17 px pulse
 *  plus the larger travelling pair, 16 + 8 px. */
const BORDER_REACH = 41;
/** The furthest the shared bend moves a border at breath 1. */
const SHARED_REACH = 22;

type TerritoryPhase = { six: number; nine: number };

/**
 * One frame of the territories. The borders run left to right, and the last
 * is the one that carries the next region in at a handover. `base` holds
 * each border before it breathes, as a fraction of the width. `shape` packs
 * its bend, a phase shift plus 8 for the prototype's right-hand form, fixed
 * by its place along the strip of projects, so a border keeps its shape from
 * one project's display to the next. `damp` is its share of the breathing.
 * There is one more region than borders.
 */
type TerritoryFrame = {
  count: number;
  base: Float32Array;
  shape: Float32Array;
  damp: Float32Array;
  clips: Float32Array;
  featured: Uint8Array;
  /** 1 when the borders also take the shared bend, which keeps thin bands
   *  wavy; one region per project goes without, as the prototype did. */
  shared: number;
  breath: number;
  phase: TerritoryPhase;
};

/** The borders breathe on 6 s and 9 s periods. Both phases are wrapped to
 *  [0, 1) before the shader sees them: a mediump sine of a clock that has run
 *  for an hour has lost its phase. */
function territoryPhase(seconds: number): TerritoryPhase {
  const wrap = (x: number) => x - Math.floor(x);
  return { six: wrap(seconds / 6), nine: wrap(seconds / 9) };
}

/**
 * The bend of one border at row fraction v, in canvas px at breath 1.
 * Borders alternate between prototype 3's left and right shapes, which is
 * also what makes neighbouring regions pulse in antiphase. MESH_FS evaluates
 * the same function per pixel, so the two have to change together.
 */
function borderWave(v: number, odd: number, shift: number, phase: TerritoryPhase): number {
  const tau = 2 * Math.PI;
  const pulse = 17 * Math.sin(tau * phase.six + 0.35);
  return odd
    ? pulse + 16 * Math.sin(tau * (1.05 * v - phase.nine) + 1.6 + shift) + 8 * Math.sin(tau * (2.6 * v + phase.six) + 0.1 + shift)
    : -pulse + 15 * Math.sin(tau * (1.15 * v + phase.nine) + shift) + 7 * Math.sin(tau * (2.8 * v - phase.six) + 0.6 + shift);
}

/**
 * A bend every border takes from where it sits across the frame, in canvas px
 * at breath 1. Neighbouring borders sit close together, so they take nearly
 * the same bend: a thin band wiggles as one thread instead of flattening.
 * Its slope across the frame stays under 0.075, so it can never carry one
 * border over another. MESH_FS evaluates the same function.
 */
function sharedWave(u: number, v: number, phase: TerritoryPhase): number {
  const tau = 2 * Math.PI;
  return 14 * Math.sin(tau * (1.1 * v + 0.8 * u + phase.nine) + 0.4) + 8 * Math.sin(tau * (2.7 * v - 0.5 * u - phase.six) + 1.9);
}

/** A non-negative remainder, for strip places left of the start. */
const wrapIndex = (value: number, period: number) => ((value % period) + period) % period;

/**
 * Lay the territories out for one frame of the project shown `index`-th.
 *
 * The regions are a window onto the strip of projects in sequence order,
 * centred on a featured region. It reaches far enough either side for the
 * featured project to hold `territoryRegions` regions' worth of the frame.
 * With an odd count the window ends on the other projects. With an even
 * count it ends on halves of the featured project, so a featured region
 * still sits in the middle, under the word. One region per project is
 * prototype 3's own layout.
 *
 * Formation widens every featured region together while the others thin,
 * from 40% of the width to 90% in all at the default growth. A handover then
 * moves each border one slot left during release, into its place in the
 * next project's opening layout: the featured regions end in the slots left
 * of the middle, the next project's take the middle, and a region arrives
 * from the right edge. Nothing ever changes clip, so nothing crossfades.
 */
function territoryFrame(p: MeshParams, t: number, c: Cycle, amplitude: number, seconds: number, index: number): TerritoryFrame {
  const ring = p.territoryRing.length ? p.territoryRing : [0];
  const period = ring.length;
  const perProject = Math.max(1, Math.min(TERRITORY_MAX_REGIONS, Math.round(p.territoryRegions)));
  const half = perProject % 2 ? (period * (perProject - 1)) / 2 + 1 : (period * perProject) / 2;
  const regions = Math.min(MAX_BORDERS, 2 * half + 1);
  const shared = perProject > 1 ? 1 : 0;
  const g = p.territoryGrowth;
  // far enough past the frame edge that breathing never brings a border back
  const off = (8 + (BORDER_REACH + shared * SHARED_REACH) * p.territoryBreath) / CANVAS_W;
  const featuredAt = (region: number) => wrapIndex(region - half, period) === 0;
  // region edges 0..1 for a given growth
  const layout = (grown: number) => {
    const weights: number[] = [];
    for (let r = 0; r < regions; r++) {
      const edge = r === 0 || r === regions - 1;
      weights.push(featuredAt(r) ? (0.4 + 2 * g * grown) * (edge ? 0.5 : 1) : Math.max(0, 0.3 - g * grown));
    }
    const total = weights.reduce((sum, w) => sum + w, 0) || 1;
    const edges = [0];
    for (const w of weights) edges.push(edges[edges.length - 1] + w / total);
    return edges;
  };

  const count = regions;
  const base = new Float32Array(count);
  if (p.territoryHandover && p.amplitudeAuto && t > c.t2) {
    const h = smoothUnit((t - c.t2) / (c.t3 - c.t2));
    const from = layout(1);
    const to = layout(0);
    for (let s = 1; s <= count; s++) {
      const start = s < count ? from[s] : 1 + off;
      const end = s === 1 ? -off : to[s - 1];
      base[s - 1] = (1 - h) * start + h * end;
    }
  } else {
    const edges = layout(Math.max(0, Math.min(1, amplitude)));
    for (let s = 1; s < count; s++) base[s - 1] = edges[s];
    base[count - 1] = 1 + off; // the entering border waits past the right edge
  }

  // Each border bends by its place along the strip of projects; the first
  // border of the index-th project is the (index - half + 1)-th.
  const shape = new Float32Array(count);
  const damp = new Float32Array(count);
  // Neighbouring borders never cross. Each takes at most 1 / 2.4 of the gap
  // to its nearer neighbour in its own bend, and the shared bend can close a
  // gap by less than a tenth of it, so thin bands wobble but hold.
  const reach = BORDER_REACH * p.territoryBreath;
  for (let s = 0; s < count; s++) {
    const border = index - half + 1 + s;
    const odd = wrapIndex(border, 2);
    shape[s] = wrapIndex(GOLDEN_ANGLE * Math.floor(border / 2), 2 * Math.PI) + 8 * odd;
    const gap = Math.min(s === 0 ? Infinity : base[s] - base[s - 1], s === count - 1 ? Infinity : base[s + 1] - base[s]);
    damp[s] = reach < 0.5 ? 1 : Math.max(0, Math.min(1, (gap * CANVAS_W) / (2.4 * reach)));
  }

  const clips = new Float32Array(count + 1);
  const featured = new Uint8Array(count + 1);
  for (let r = 0; r <= count; r++) {
    clips[r] = ring[wrapIndex(index - half + r, period)];
    featured[r] = featuredAt(r) ? 1 : 0;
  }
  return { count, base, shape, damp, clips, featured, shared, breath: p.territoryBreath, phase: territoryPhase(seconds * p.breathSpeed) };
}

/** Every border at canvas row y, in canvas px, left to right. */
function territoryBorders(frame: TerritoryFrame, y: number, out: Float32Array, offset: number) {
  const v = y / CANVAS_H;
  for (let s = 0; s < frame.count; s++) {
    const odd = frame.shape[s] >= 7 ? 1 : 0;
    let bend = frame.damp[s] * borderWave(v, odd, frame.shape[s] - 8 * odd, frame.phase);
    if (frame.shared) bend += sharedWave(frame.base[s], v, frame.phase);
    out[offset + s] = CANVAS_W * frame.base[s] + frame.breath * bend;
  }
}

/**
 * How much of the featured project's field reaches canvas x on a row whose
 * borders start at `offset`. Every other project's region suppresses it out
 * to the margin, in proportion to how open that region is, so a band that
 * pinches shut lets the word heal across it. With one region per project
 * this is the prototype's gate exactly.
 */
function territoryGate(frame: TerritoryFrame, borders: Float32Array, offset: number, x: number, margin: number): number {
  let gate = 1;
  for (let r = 0; r <= frame.count; r++) {
    if (frame.featured[r]) continue;
    const left = r === 0 ? -Infinity : borders[offset + r - 1];
    const right = r === frame.count ? Infinity : borders[offset + r];
    const outside = Math.max(0, left - x, x - right);
    if (outside >= margin) continue;
    gate *= 1 - smoothUnit(1 - outside / margin) * smoothUnit((right - left) / margin);
  }
  return gate;
}

/** The clips a display cannot draw without: every territory, the home and
 *  what lies under the cloth it is painted on, or the home alone. */
function requiredClips(p: MeshParams): number[] {
  if (p.footage === 'territories') return p.territoryRing;
  if ((p.footage === 'cloth' || p.footage === 'clothmesh' || p.footage === 'clothlift') && p.clothUnder >= 0 && p.clothUnder !== p.homeClip) return [p.homeClip, p.clothUnder];
  return [p.homeClip];
}

/* ---------------------------------------------------------------------- */
/* 7c. flow into the word                                                  */
/* ---------------------------------------------------------------------- */

/** How far the flood's leading edge pushes the collage ahead of itself, and
 *  the depth over which that push fades behind the edge, in canvas px. The
 *  collage right at the edge moves at 56% of the edge's speed, squeezed 2.25
 *  times, then slides under it. Nothing is ever replaced where it stands. */
const FLOW_PUSH = 50;
const FLOW_BAND = 40;
/** Small residual motion at the very start of the flood. */
const FLOW_WAVE = 5;
/** Height-only cloth: canvas px the frontier bends per Blender unit of sag. */
const FLOW_BAKED_CLOTH_PX = 46;
/** How far, in canvas px, the fabric may carry the frontier from the plain
 *  one, closing to nothing at full amplitude. The hold is then exactly the
 *  plain one whatever shape the fabric is in, and since the limit closes
 *  more slowly than the front advances, nothing flooded is uncovered. */
const FLOW_CLOTH_LEAD = 400;
/** The amplitude by which push and flow have let the lattice's breathing
 *  settle out, so the held word stands still. */
const BREATH_SETTLE = 0.75;

/** The hold drop. Once formed, the word waits DROP_PAUSE s, falls under
 *  gravity for DROP_TIME s, and bounces to rest DROP_MARGIN_CELLS above the
 *  frame's bottom edge, so the middle of the frame is clear while the word is
 *  held. It rises back as the release takes the amplitude to zero. */
const DROP_PAUSE = 0.35;
const DROP_TIME = 0.45;
const DROP_MARGIN_CELLS = 3;
/** A bounce this slow, in px/s, is the last. */
const DROP_REST_SPEED = 30;

/** How far the word drops, in canvas px: a whole number of lattice rows, so
 *  it lands exactly on the lattice it was solved on. */
function dropDistance(glyph: Glyph, lat: Lattice, offsetY: number): number {
  const inkBottom = CANVAS_H / 2 + offsetY + glyph.inkHeight / 2;
  const room = CANVAS_H - DROP_MARGIN_CELLS * lat.hy - inkBottom;
  return Math.max(0, Math.floor(room / lat.hy)) * lat.hy;
}

/** The drop `s` seconds after it starts: a fall under gravity, then bounces
 *  that each keep `restitution` of the speed, until one is slower than
 *  DROP_REST_SPEED. */
function dropFall(s: number, distance: number, restitution: number): number {
  if (s <= 0) return 0;
  const g = (2 * distance) / (DROP_TIME * DROP_TIME);
  if (s < DROP_TIME) return 0.5 * g * s * s;
  let time = s - DROP_TIME;
  let speed = g * DROP_TIME * Math.max(0, Math.min(0.95, restitution));
  while (speed > DROP_REST_SPEED) {
    const flight = (2 * speed) / g;
    if (time < flight) return distance - (speed * time - 0.5 * g * time * time);
    time -= flight;
    speed *= restitution;
  }
  return distance;
}

/** The word's drop at cycle time t: none while it forms, the fall and bounce
 *  through the hold, and it stays down through the release. */
function holdDropAt(t: number, c: Cycle, distance: number, restitution: number): number {
  if (distance <= 0 || t < c.t1) return 0;
  return dropFall(Math.min(t, c.t2) - c.t1 - DROP_PAUSE, distance, restitution);
}

/* ---------------------------------------------------------------------- */
/* 8. WebGL mesh renderer                                                  */
/* ---------------------------------------------------------------------- */

/** Floats per vertex in the mesh buffer: position, texture coordinates, four
 *  vec4 attributes, rest position, then the flow's collage coordinate, the
 *  distance to the word and the letter flag. */
const VERTEX_FLOATS = 22;

/** Canvas px to a Blender unit, as the bake's own heightUnit records. */
const CLOTH_PX = 100;

/** Floats per vertex of the cloth wipe's grid: screen position, the material
 *  point it reads the clip at, its sag and squeeze, its slope, and whether
 *  the sheet is there at all. The
 *  distances to the word are read per pixel from the glyph's own field. */
const CLOTH_FLOATS = 9;
/** The glyph's distance field goes to the cloth shader as 16 bits a sample:
 *  canvas px, offset so ±CLOTH_FIELD_RANGE fits, at 1/32 px. */
const CLOTH_FIELD_RANGE = 1024;
const CLOTH_FIELD_STEPS = 32;
/** The cloth wipe draws the bake's grid this many times finer, rebuilt from
 *  its samples by Catmull-Rom, so the squeeze and the slope are shaded on a
 *  smooth sheet instead of on 16 px triangles. 3 keeps the grid under the
 *  65536 vertices a Uint16 index can reach. */
const CLOTH_REFINE = 3;

function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number) {
  return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
}

/** A grid refined `k` times along both axes, through every sample, with its
 *  ends clamped. `tmp` holds the pass along x. */
function refineGrid(src: Float32Array, nx: number, ny: number, k: number, tmp: Float32Array, out: Float32Array) {
  const fx = (nx - 1) * k + 1;
  const fy = (ny - 1) * k + 1;
  const at = (i: number, n: number) => (i < 0 ? 0 : i >= n ? n - 1 : i);
  for (let j = 0; j < ny; j++) {
    const row = j * nx;
    for (let c = 0; c < fx; c++) {
      const i = Math.min(nx - 2, Math.floor(c / k));
      const t = c / k - i;
      tmp[j * fx + c] = catmullRom(src[row + at(i - 1, nx)], src[row + i], src[row + i + 1], src[row + at(i + 2, nx)], t);
    }
  }
  for (let r = 0; r < fy; r++) {
    const j = Math.min(ny - 2, Math.floor(r / k));
    const t = r / k - j;
    const r0 = at(j - 1, ny) * fx;
    const r1 = j * fx;
    const r2 = (j + 1) * fx;
    const r3 = at(j + 2, ny) * fx;
    for (let c = 0; c < fx; c++) out[r * fx + c] = catmullRom(tmp[r0 + c], tmp[r1 + c], tmp[r2 + c], tmp[r3 + c], t);
  }
}

const MESH_VS = `
attribute vec2 aPos;
attribute vec2 aUV;
attribute vec4 aData;
attribute vec4 aDebug;
attribute vec4 aTransition;
attribute vec2 aShadeUV;
attribute vec4 aFlow;
uniform vec2 uCanvas;
varying vec2 vUV;
varying vec4 vData;
varying vec4 vDebug;
varying vec4 vTransition;
varying vec2 vShadeUV;
varying vec4 vFlow;
void main() {
  vUV = aUV;
  vData = aData;
  vDebug = aDebug;
  vTransition = aTransition;
  vShadeUV = aShadeUV;
  vFlow = aFlow;
  gl_Position = vec4(aPos.x / uCanvas.x * 2.0 - 1.0, 1.0 - aPos.y / uCanvas.y * 2.0, 0.0, 1.0);
}
`;

const MESH_FS = `
#extension GL_OES_standard_derivatives : enable
precision mediump float;
#define MAX_BORDERS ${MAX_BORDERS}
varying vec2 vUV;
varying vec4 vData;
varying vec4 vDebug;
varying vec4 vTransition;
varying vec2 vShadeUV;
// The flow: how far the collage has been carried to reach this pixel, in cells, its
// distance to the word in canvas px, and 1 inside a letter cell.
varying vec4 vFlow;
uniform sampler2D clip0;
uniform sampler2D clip1;
uniform sampler2D clip2;
uniform sampler2D clip3;
uniform sampler2D clip4;
uniform vec2 uFieldCanvas;
uniform vec4 uFit0;
uniform vec4 uFit1;
uniform vec4 uFit2;
uniform vec4 uFit3;
uniform vec4 uFit4;
uniform int uView;
uniform float uStrength;
uniform float uShadeOn;
uniform float uLumaAdaptive;
uniform float uMedian;
uniform float uScreenField;
uniform vec2 uCellUvSize;
uniform float uCommitMotion;
uniform vec4 uTerritory; // on, breath, borders in use, clip of the leftmost region
uniform vec3 uTerritoryPhase; // the 6 s and 9 s breathing phases in [0, 1), and 1 for the shared bend
// Left to right: position before breathing (fraction of the width), shape,
// share of the breathing, clip of the region to the border's right.
uniform vec4 uBorder[MAX_BORDERS];
// The flow: on, the flood's edge as a distance to the word (canvas px), and
// the featured clip flooding in.
uniform vec4 uFlood;
uniform vec2 uJumbleGrid; // collage cells across and down
uniform sampler2D uJumble; // the collage: one clip per cell
uniform vec4 uRestCell; // rest position to collage cell: scale in xy, offset in zw
uniform float uCoverFront; // distance to the word that the collage covering the letters has reached, canvas px

const float TAU = 6.28318531;

vec2 fit(vec2 uv, vec4 f) { return (uv - 0.5) * f.xy + 0.5 + f.zw; }

float jumbleAt(vec2 cell) {
  return floor(texture2D(uJumble, (floor(cell) + 0.5) / uJumbleGrid).r * 255.0 + 0.5);
}

// The moving layer: every piece but the featured project's, carried by the
// flow. A featured cell of it is filled from the neighbour across its nearer
// edge, or failing that the other one, so the layer carries no hole.
float movingAt(vec2 p) {
  float id = jumbleAt(p);
  if (abs(id - uFlood.w) > 0.5) return id;
  vec2 f = fract(p) - 0.5;
  vec2 a = abs(f);
  vec2 near = a.x >= a.y ? vec2(sign(f.x), 0.0) : vec2(0.0, sign(f.y));
  vec2 far = a.x >= a.y ? vec2(0.0, sign(f.y)) : vec2(sign(f.x), 0.0);
  float first = jumbleAt(p + near);
  if (abs(first - uFlood.w) > 0.5) return first;
  float second = jumbleAt(p + far);
  return abs(second - uFlood.w) > 0.5 ? second : id;
}

// The collage at one sample. The featured project's pieces stand where they
// always were, above the moving layer, which the flow pushes in beneath them.
// Inside the letters, the collage flowing in covers them from the contour to
// the core as the word forms, so the letters hold only the other projects.
// At idle the two layers agree cell for cell, whichever project is featured.
float collageAt(vec2 moving, vec2 still, float distance, float letter) {
  float standing = jumbleAt(still);
  bool featured = abs(standing - uFlood.w) < 0.5;
  if (featured && (letter < 0.5 || distance <= uCoverFront)) return standing;
  return movingAt(moving);
}

// A quarter of a screen pixel along each screen axis, to sample the collage
// four times a pixel. That antialiases every seam in it, and the covering
// front inside the letters.
vec4 collageStep(vec2 cell) {
  return vec4(dFdx(cell), dFdy(cell)) * 0.25;
}

vec2 distanceStep(float distance) {
  return vec2(dFdx(distance), dFdy(distance)) * 0.25;
}

// Screen pixels across one px of distance, for antialiasing the flood's edge.
float floodFeather(float d) {
  return max(fwidth(d), 0.0001);
}

// borderWave() in the engine, in canvas px at breath 1. The shape packs the
// phase shift plus 8 for the prototype's right-hand form.
float borderWave(float v, float shape) {
  float odd = step(7.0, shape);
  float shift = shape - 8.0 * odd;
  float six = uTerritoryPhase.x;
  float nine = uTerritoryPhase.y;
  float pulse = 17.0 * sin(TAU * six + 0.35);
  float leftShape = -pulse + 15.0 * sin(TAU * (1.15 * v + nine) + shift) + 7.0 * sin(TAU * (2.8 * v - six) + 0.6 + shift);
  float rightShape = pulse + 16.0 * sin(TAU * (1.05 * v - nine) + 1.6 + shift) + 8.0 * sin(TAU * (2.6 * v + six) + 0.1 + shift);
  return mix(leftShape, rightShape, odd);
}

// sharedWave() in the engine: the bend every border takes from where it sits.
float sharedWave(float u, float v) {
  return 14.0 * sin(TAU * (1.1 * v + 0.8 * u + uTerritoryPhase.y) + 0.4) + 8.0 * sin(TAU * (2.7 * v - 0.5 * u - uTerritoryPhase.x) + 1.9);
}

// Rest-space width of one screen pixel, for the border's antialiasing.
float territoryFeather(float u) {
  return max(fwidth(u), 0.0001);
}

vec3 clipColor(float id, vec2 uv) {
  if (id < 0.5) return texture2D(clip0, fit(uv, uFit0)).rgb;
  if (id < 1.5) return texture2D(clip1, fit(uv, uFit1)).rgb;
  if (id < 2.5) return texture2D(clip2, fit(uv, uFit2)).rgb;
  if (id < 3.5) return texture2D(clip3, fit(uv, uFit3)).rgb;
  return texture2D(clip4, fit(uv, uFit4)).rgb;
}

vec3 familyColor(float id) {
  if (id < 0.5) return vec3(0.94, 0.40, 0.20);
  if (id < 1.5) return vec3(0.15, 0.63, 0.85);
  if (id < 2.5) return vec3(0.73, 0.84, 0.31);
  if (id < 3.5) return vec3(0.66, 0.43, 0.89);
  return vec3(0.92, 0.54, 0.58);
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
  // Territories replace the per-cell clip with the borders, evaluated per
  // pixel in the rest lattice. A moving border then slides through cells
  // instead of stepping between them, and it rides the mesh as footage does.
  float borderId = clipId;
  float borderMix = 0.0;
  if (uTerritory.x > 0.5) {
    float u = vShadeUV.x;
    float px = uTerritory.y / uFieldCanvas.x;
    float feather = territoryFeather(u);
    float here = uTerritory.w; // the region under this pixel
    float lastX = -1.0e4;
    float regionClip = uTerritory.w; // the region left of the border in hand
    float leftClip = uTerritory.w; // the nearest open region left of it
    float leftWidth = 1.0e4;
    // The nearest border feathers. A band that has pinched shut is looked
    // past, so it cannot leave a hairline of its clip behind.
    float nearest = 1.0e4;
    float nearLeft = here;
    float nearRight = here;
    float nearLeftWidth = 1.0e4;
    float nearRightWidth = 1.0e4;
    float open = -2.0; // the border whose right side is still being read
    for (int i = 0; i < MAX_BORDERS; i++) {
      if (float(i) >= uTerritory.z) break;
      vec4 border = uBorder[i];
      float x = border.x;
      if (abs(u - x) < (border.z * 41.0 + uTerritoryPhase.z * 22.0) * px + 2.0 * feather) {
        x += px * (border.z * borderWave(vShadeUV.y, border.y) + uTerritoryPhase.z * sharedWave(border.x, vShadeUV.y));
      }
      float width = x - lastX;
      bool shut = width <= 0.00001;
      if (!shut) {
        leftClip = regionClip;
        leftWidth = width;
      }
      if (open == float(i) - 1.0) {
        if (shut) {
          nearRight = border.w;
          open = float(i);
        } else {
          nearRightWidth = width;
          open = -2.0;
        }
      }
      float d = u - x;
      if (d >= 0.0) here = border.w;
      if (abs(d) < abs(nearest)) {
        nearest = d;
        nearLeft = leftClip;
        nearLeftWidth = leftWidth;
        nearRight = border.w;
        nearRightWidth = 1.0e4;
        open = float(i);
      }
      regionClip = border.w;
      lastX = x;
    }
    float side = clamp(nearest / feather + 0.5, 0.0, 1.0);
    float across = nearest >= 0.0 ? nearLeftWidth : nearRightWidth;
    clipId = here;
    borderId = nearest >= 0.0 ? nearLeft : nearRight;
    borderMix = (nearest >= 0.0 ? 1.0 - side : side) * clamp(across / feather, 0.0, 1.0);
  }
  // The flow reads the collage where the engine has carried it, four times a
  // pixel, then lays the flood over everything past the flood's edge except
  // the letters.
  float collage0 = 0.0;
  float collage1 = 0.0;
  float collage2 = 0.0;
  float collage3 = 0.0;
  float flood = 0.0;
  if (uFlood.x > 0.5) {
    vec2 still = vShadeUV * uRestCell.xy + uRestCell.zw;
    vec2 cell = still + vFlow.xy;
    vec4 q = collageStep(cell);
    vec4 r = collageStep(still);
    vec2 e = distanceStep(vFlow.z);
    collage0 = collageAt(cell - q.xy - q.zw, still - r.xy - r.zw, vFlow.z - e.x - e.y, vFlow.w);
    collage1 = collageAt(cell + q.xy - q.zw, still + r.xy - r.zw, vFlow.z + e.x - e.y, vFlow.w);
    collage2 = collageAt(cell - q.xy + q.zw, still - r.xy + r.zw, vFlow.z - e.x + e.y, vFlow.w);
    collage3 = collageAt(cell + q.xy + q.zw, still + r.xy + r.zw, vFlow.z + e.x + e.y, vFlow.w);
    flood = (1.0 - vFlow.w) * clamp((vFlow.z - uFlood.y) / floodFeather(vFlow.z) + 0.5, 0.0, 1.0);
  }
  bool collageSplit = collage1 != collage0 || collage2 != collage0 || collage3 != collage0;
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
    vec3 pieces = 0.25 * (familyColor(collage0) + familyColor(collage1) + familyColor(collage2) + familyColor(collage3));
    color = uFlood.x > 0.5
      ? mix(pieces, familyColor(uFlood.w), flood)
      : mix(familyColor(clipId), familyColor(borderId), borderMix);
  } else {
    vec3 src;
    if (uFlood.x > 0.5) {
      // Every clip at its own place on screen, the featured one included, so
      // a featured piece meets the flood without a seam.
      src = clipColor(collage0, vUV);
      if (collageSplit) src = 0.25 * (src + clipColor(collage1, vUV) + clipColor(collage2, vUV) + clipColor(collage3, vUV));
      if (flood > 0.0) src = mix(src, clipColor(uFlood.w, vUV), flood);
    } else {
      src = coreBlend
        ? mix(clipColor(vData.y, vUV), clipColor(vDebug.w, vUV), vTransition.x)
        : clipColor(clipId, footageUV);
      if (borderMix > 0.0) src = mix(src, clipColor(borderId, footageUV), borderMix);
    }
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
  .replace(/float screenDet\(\) \{[\s\S]*?\n\}/, 'float screenDet() { return vDebug.x; }')
  .replace(/float territoryFeather\(float u\) \{[\s\S]*?\n\}/, 'float territoryFeather(float u) { return 1.0 / uFieldCanvas.x; }')
  .replace(/vec4 collageStep\(vec2 cell\) \{[\s\S]*?\n\}/, 'vec4 collageStep(vec2 cell) { return vec4(0.0); }')
  .replace(/vec2 distanceStep\(float distance\) \{[\s\S]*?\n\}/, 'vec2 distanceStep(float distance) { return vec2(0.0); }')
  .replace(/float floodFeather\(float d\) \{[\s\S]*?\n\}/, 'float floodFeather(float d) { return 1.0; }');

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

/** The cloth wipe. The clips are stacked, each on the next; the fall drags
 *  the top one into the letter-shaped holes of its project's word, and the
 *  one beneath comes in as the fabric leaves. One quad grid, the bake's own,
 *  standing still on screen: the picture moves because each point reads the
 *  clip at the material point of the fabric visible there. Nothing here is
 *  solved, smoothed or carried; the vertices come straight off the bake. */
const CLOTH_VS = `
attribute vec2 aPos;
attribute vec2 aUV;
attribute vec2 aShade;
attribute vec2 aSlope;
attribute float aCover;
uniform vec2 uCanvas;
varying vec2 vUV;
varying vec2 vRest;
varying vec2 vShade;
varying vec2 vSlope;
varying float vEdge;
varying float vCover;
void main() {
  vCover = aCover;
  // The material point comes in canvas px and is divided here exactly as
  // the screen point is, so at rest the two are the same to the bit.
  vUV = aUV / uCanvas;
  // How far inside the painted sheet this material is, in canvas px. The
  // clip is painted on a frame-sized piece of the fabric, so its border is a
  // material line the fall drags in. 2 px of slack keeps the frame's own
  // edge covered at rest.
  vEdge = min(min(aUV.x, uCanvas.x - aUV.x), min(aUV.y, uCanvas.y - aUV.y)) + 2.0;
  vRest = aPos / uCanvas;
  vShade = aShade;
  vSlope = aSlope;
  gl_Position = vec4(aPos.x / uCanvas.x * 2.0 - 1.0, 1.0 - aPos.y / uCanvas.y * 2.0, 0.0, 1.0);
}
`;

const CLOTH_FS = `
#extension GL_OES_standard_derivatives : enable
// The field decodes 16-bit samples and the distances run to a thousand px,
// which half floats cannot hold.
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D clip0;
uniform sampler2D clip1;
uniform sampler2D clip2;
uniform sampler2D clip3;
uniform sampler2D clip4;
uniform vec4 uFit0;
uniform vec4 uFit1;
uniform vec4 uFit2;
uniform vec4 uFit3;
uniform vec4 uFit4;
// The clip on the fabric, and the clip under it that the fall uncovers.
uniform float uCover;
uniform float uUnder;
// The signed exponent on the fabric's squeeze, as the mesh uses, then how
// far the fabric's own shape is drawn at all.
uniform float uStrength;
uniform float uRelief;
uniform float uDepth;
// How far the frontier has come in, as a distance to the word, and how far
// into the letters the fabric is still kept while the last of it goes through.
uniform float uFront;
uniform float uKeep;
varying vec2 vUV;
varying vec2 vRest;
varying vec2 vShade; // sag below the resting sheet, and the squeeze ratio
varying vec2 vSlope; // the sheet's own slope, canvas px per canvas px
varying float vEdge;  // inside the painted sheet, canvas px
varying float vCover; // 1 where the bake has cloth over the floor, 0 where it has left
// The word's distance field, 16 bits a sample, and how canvas px map to it.
uniform sampler2D uField;
uniform vec2 uFieldSize;
uniform vec2 uFieldMap; // padding in canvas px, samples per canvas px
uniform vec2 uFrame;    // canvas px

// Over the shoulder, as the raw bake renders read best under. z points out of
// the screen and y down it, so the light comes from above and to the left.
const vec3 LIGHT = vec3(-0.372, -0.414, 0.830);

vec2 fit(vec2 uv, vec4 f) { return (uv - 0.5) * f.xy + 0.5 + f.zw; }

vec3 clipColor(float id, vec2 uv) {
  if (id < 0.5) return texture2D(clip0, fit(uv, uFit0)).rgb;
  if (id < 1.5) return texture2D(clip1, fit(uv, uFit1)).rgb;
  if (id < 2.5) return texture2D(clip2, fit(uv, uFit2)).rgb;
  if (id < 3.5) return texture2D(clip3, fit(uv, uFit3)).rgb;
  return texture2D(clip4, fit(uv, uFit4)).rgb;
}

float fieldSample(vec2 texel) {
  vec4 s = texture2D(uField, (texel + 0.5) / uFieldSize);
  return (s.r * 65280.0 + s.a * 255.0) / ${CLOTH_FIELD_STEPS.toFixed(1)} - ${CLOTH_FIELD_RANGE.toFixed(1)};
}

// The distance to the word at a canvas point, filtered as the CPU samples it.
// Read per pixel, so an edge in it follows the letters' own curves instead
// of the straight lines between the bake's grid points.
float fieldAt(vec2 p) {
  vec2 f = clamp((p + uFieldMap.x) * uFieldMap.y, vec2(0.0), uFieldSize - 1.001);
  vec2 i = floor(f);
  vec2 t = f - i;
  float a = fieldSample(i);
  float b = fieldSample(i + vec2(1.0, 0.0));
  float c = fieldSample(i + vec2(0.0, 1.0));
  float d = fieldSample(i + vec2(1.0, 1.0));
  return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

// Screen pixels across one px of distance, for antialiasing an edge in it.
float feather(float d) {
  return max(fwidth(d), 0.0001);
}

void main() {
  // The fabric's own shape, lit: the pleats and the walls of the letter holes
  // are the word. Flat cloth returns exactly 1, so idle is the bare clip.
  // Each term is written as 1 plus a change, so flat fabric comes to exactly
  // 1 in full precision and the plain clip is the clip to the bit.
  vec3 normal = normalize(vec3(-vSlope.x, -vSlope.y, 1.0));
  float lit = 1.0 + 0.65 * (max(dot(normal, LIGHT), 0.0) - LIGHT.z) / LIGHT.z;
  // and darker the deeper it has gone, as the bottom of a hole is
  float sunk = 1.0 - 0.55 * clamp(vShade.x / uDepth, 0.0, 1.0);
  float shade = pow(max(vShade.y, 0.05), uStrength) * (1.0 + uRelief * (lit * sunk - 1.0));
  vec3 fabric = clamp(clipColor(uCover, vUV) * shade, 0.0, 1.0);
  vec3 floorColor = clipColor(uUnder, vRest);
  // The letters are the one place the fabric is kept while the frontier goes
  // past. Then the keep line sinks through them as well, so the word is drawn
  // down its own holes, outline inwards, and nothing is faded out.
  float here = fieldAt(vRest * uFrame);
  float carried = fieldAt(vUV * uFrame);
  float letter = clamp((uKeep - here) / feather(here) + 0.5, 0.0, 1.0);
  float gone = clamp((carried - uFront) / feather(carried) + 0.5, 0.0, 1.0);
  // The sheet's own edge goes first: past it there is no picture on the
  // fabric, so what the fall pulls in from the frame's border is the clip's
  // edge itself, crumpled as the cloth is. That is what reads as cloth.
  gone = max(gone, clamp(-vEdge / feather(vEdge) + 0.5, 0.0, 1.0));
  // and where the bake itself has no cloth, poured down a hole or pulled
  // away, the floor shows: the sheet's own outline, straight off the bake.
  gone = max(gone, clamp((0.5 - vCover) / feather(vCover) + 0.5, 0.0, 1.0));
  gl_FragColor = vec4(mix(fabric, floorColor, (1.0 - letter) * gone), 1.0);
}
`;

// Firefox headless has no derivatives; one screen pixel of distance is the
// same feather to within a hair, since the field is a distance in canvas px.
const CLOTH_FS_FALLBACK = CLOTH_FS
  .replace('#extension GL_OES_standard_derivatives : enable\n', '')
  .replace(/float feather\(float d\) \{[\s\S]*?\n\}/, 'float feather(float d) { return 1.0; }');

/** The cloth wipe's frame: the clip on the fabric and the one under it, the
 *  signed exponent on the fabric's squeeze, how far its shape is drawn, the
 *  fall's depth in Blender units, the frontier as a distance to the word, and
 *  how much of the fabric caught in the letters is left. */

/** The mesh wipe: the bake's cloth drawn as it is, seen from straight above,
 *  over a floor that is the next clip with the word's holes in it. One
 *  program draws both: the floor as a quad at height 0 with its holes let
 *  through, then the cloth, its depth from its height, so cloth below the
 *  floor shows only through a hole and folds over folds sort themselves. */
const CLOTH_MESH_VS = `
attribute vec3 aPos;    // canvas px across and down, Blender units up
attribute vec2 aRest;   // where this point of the cloth started, canvas px
attribute vec3 aNormal;
attribute float aCavity; // px this point sits below its neighbours, smoothed
uniform vec2 uCanvas;
uniform float uSink;
varying vec2 vUV;
varying vec2 vScreen;
varying vec3 vNormal;
varying float vZ;
varying float vCavity;
void main() {
  vCavity = aCavity;
  vUV = aRest / uCanvas;
  vScreen = aPos.xy;
  vNormal = aNormal;
  vZ = aPos.z - uSink;
  // Higher is nearer: the floor at 0 sits at depth 0, cloth over it in front.
  gl_Position = vec4(aPos.x / uCanvas.x * 2.0 - 1.0, 1.0 - aPos.y / uCanvas.y * 2.0, clamp(-vZ / 40.0, -1.0, 1.0), 1.0);
}
`;

const CLOTH_MESH_FS = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D clip0;
uniform sampler2D clip1;
uniform sampler2D clip2;
uniform sampler2D clip3;
uniform sampler2D clip4;
uniform vec4 uFit0;
uniform vec4 uFit1;
uniform vec4 uFit2;
uniform vec4 uFit3;
uniform vec4 uFit4;
uniform float uFloor;   // 1 drawing the floor, 0 the cloth
uniform float uCover;   // the clip on the cloth
uniform float uUnder;   // the clip on the floor
uniform float uRelief;
uniform float uSheetZ;  // the cloth's height at rest
uniform float uFloorDepth;
uniform sampler2D uHoles;
uniform float uHeal;    // px the holes have closed by
uniform vec3 uVoid;     // down a hole, where no cloth is
uniform float uLift;    // the lift's height: cloth raised brightens, 0 off
uniform float uMargin;  // 1 to carry the clip's edge out over the margin
// A lift's letters, from the page's own glyph (in uHoles): how far they have
// slid down the frame, the height of their tops, and how much of the bevel
// to draw on the cloth lying over them.
uniform float uLetterDy;
uniform float uLetterTop;
uniform float uBevel;
// The lift's material, faded in with the rise: a satin sheen on the cloth's
// own folds, and the dark of its valleys.
uniform float uMaterial;
uniform float uSheen;
uniform float uCavity;
varying float vCavity;
uniform vec2 uFrame;    // canvas px
varying vec2 vScreen;
varying vec2 vUV;
varying vec3 vNormal;
varying float vZ;

const vec3 LIGHT = vec3(-0.372, -0.414, 0.830);

vec2 fit(vec2 uv, vec4 f) { return (uv - 0.5) * f.xy + 0.5 + f.zw; }

vec3 clipColor(float id, vec2 uv) {
  if (id < 0.5) return texture2D(clip0, fit(uv, uFit0)).rgb;
  if (id < 1.5) return texture2D(clip1, fit(uv, uFit1)).rgb;
  if (id < 2.5) return texture2D(clip2, fit(uv, uFit2)).rgb;
  if (id < 3.5) return texture2D(clip3, fit(uv, uFit3)).rgb;
  return texture2D(clip4, fit(uv, uFit4)).rgb;
}

void main() {
  if (uFloor > 0.5) {
    // How much of this pixel is hole: its antialiased edge, then inside it,
    // less what has healed. A whole hole lets the cloth under it show.
    float inside = texture2D(uHoles, vUV).r * 127.5 - 1.0;
    float hole = clamp(inside + 0.5 - uHeal, 0.0, 1.0);
    if (hole >= 1.0) discard;
    gl_FragColor = vec4(mix(clipColor(uUnder, vUV), uVoid, hole), 1.0);
    return;
  }
  // Past the frame at rest the sheet carries no picture: it is not there,
  // unless the clip's edge is carried out over it.
  vec2 uv = vUV;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    if (uMargin < 0.5) discard;
    // mirrored past the edge, so the margin carries picture, not a smear of
    // the edge's last row, and the seam at the frame's edge is continuous
    uv = 1.0 - abs(mod(uv, 2.0) - 1.0);
  }
  // Nor once it is out of the floor's underside: poured through, as the
  // raster export counts it.
  if (vZ < -uFloorDepth) discard;
  // Seen from above either face may be up. Each term is 1 plus a change, so
  // flat cloth at rest is the clip to the bit.
  vec3 n = normalize(vNormal);
  n = n.z < 0.0 ? -n : n;
  // A lift lights its folds twice as hard: the word is only creases in the
  // cloth, and the clip's own detail would bury them otherwise.
  float lit = 1.0 + (uLift > 0.0 ? 1.3 : 0.65) * (max(dot(n, LIGHT), 0.0) - LIGHT.z) / LIGHT.z;
  float sunk = 1.0 - 0.55 * clamp((uSheetZ - vZ) / uFloorDepth, 0.0, 1.0);
  float shade = 1.0 + uRelief * (lit * sunk - 1.0);
  float added = 0.0;
  float edge = 0.0;
  float sheen = 0.0;
  if (uMaterial > 0.0) {
    // The dark side of the relief: where the cloth sits below its
    // neighbours, in the dished strokes and the tight valleys between folds.
    shade *= 1.0 - uCavity * uMaterial * 0.5 * smoothstep(0.5, 6.0, vCavity);
    // A satin print: a broad highlight off the cloth's own normals, seen from
    // straight above. It is reflected light, so it shows over black.
    // Measured against flat cloth, so only faces tilted to the light catch
    // it and a flat stretch of footage is left exactly as it is.
    vec3 h = normalize(LIGHT + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, h), 0.0), 18.0) - pow(h.z, 18.0);
    sheen = uSheen * uMaterial * 0.6 * max(spec, 0.0);
  }
  if (uBevel > 0.0) {
    // Where the cloth lies on a letter, shade it as the letter's bevel: the
    // glyph's distance field, where the letters are now, gives the slope in
    // a band just inside the outline, lit like the cloth's own folds.
    vec2 at = (vScreen - vec2(0.0, uLetterDy)) / uFrame;
    vec2 px = 1.5 / uFrame;
    float inside = texture2D(uHoles, at).r * 127.5 - 1.0;
    float on = uBevel * smoothstep(uLetterTop - 0.6, uLetterTop - 0.3, vZ);
    if (on > 0.0 && inside > -1.0) {
      float dx = texture2D(uHoles, at + vec2(px.x, 0.0)).r - texture2D(uHoles, at - vec2(px.x, 0.0)).r;
      float dy = texture2D(uHoles, at + vec2(0.0, px.y)).r - texture2D(uHoles, at - vec2(0.0, px.y)).r;
      // up the field is inward; the bevel falls outward, steepest at the edge
      float band = 1.0 - smoothstep(0.0, 9.0, inside);
      vec3 slope = normalize(vec3(dx, dy, 0.0001)) * band;
      vec3 bn = normalize(vec3(slope.x, slope.y, 0.55));
      float bevelLit = 1.0 + 1.3 * (max(dot(bn, LIGHT), 0.0) - LIGHT.z) / LIGHT.z;
      // A contrast edge along the inside of the outline, stronger on the
      // faces turned to the light: lighter where the footage under it is
      // dark, darker where it is bright, so the outline always stands off
      // its own background. Signed here; resolved against the footage below.
      edge = on * (0.45 * band + 0.6 * max(bevelLit - 1.0, 0.0));
    }
  }
  vec3 lit3 = clipColor(uCover, uv) * shade;
  float luma = clamp(dot(lit3, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
  // bright footage takes the edge as shadow, dark footage as light
  float bright = smoothstep(0.3, 0.7, luma);
  // and twice as strong on bright footage, where a dark edge has more to
  // overcome: the tuned strength times 1 plus the brightness
  lit3 *= 1.0 - bright * min(edge * (1.0 + bright), 1.0) * 0.75;
  lit3 += (1.0 - bright) * edge;
  gl_FragColor = vec4(clamp(lit3 + added + sheen, 0.0, 1.0), 1.0);
}
`;

/** Test-page knobs for the lift, read off the URL: `flat` blends each
 *  triangle's own facet into the smooth normals (0 smooth, 1 faceted), and
 *  `bevel=0` turns the glyph's bevel off to judge the cloth on its own. */
const LIFT_KNOBS = (() => {
  const q = new URLSearchParams(typeof location === 'undefined' ? '' : location.search);
  const flat = Number(q.get('flat') ?? 0.35);
  const knob = (name: string, fallback: number) => {
    const v = Number(q.get(name) ?? fallback);
    return Number.isFinite(v) ? Math.max(0, v) : fallback;
  };
  return { flat: Number.isFinite(flat) ? Math.max(0, Math.min(1, flat)) : 0.35, bevel: knob('bevel', 0.4), sheen: knob('sheen', 1), cavity: knob('cavity', 1), full: q.get('full') === '1' };
})();

/** The test page's Bevel slider: how strong the lift's glyph glow is, 1 as
 *  tuned, 0 off. */
export function setLiftBevel(strength: number) {
  LIFT_KNOBS.bevel = Math.max(0, strength);
}

/** The lift's glyph glow strength, for the slider to start from. */
export function liftBevel(): number {
  return LIFT_KNOBS.bevel;
}

/** What the mesh wipe draws a frame with. */
type ClothMeshDraw = { cover: number; under: number; relief: number; sheetZ: number; floorDepth: number; heal: number; sink: number; lift?: number; margin?: boolean; letterDy?: number; letterTop?: number; bevel?: number; arrays?: number; material?: number; sheen?: number; cavity?: number };
/** Floats per cloth point: position (canvas px, px, units), rest (px), normal. */
const CLOTH_MESH_FLOATS = 9;

type ClothDraw = { cover: number; under: number; strength: number; relief: number; depth: number; front: number; keep: number };

/** The flood's edge as a distance to the word, the clip flooding in, the
 *  collage's size in cells, how a rest position maps onto a collage cell,
 *  and how far into the letters the incoming collage has covered them. */
type FlowDraw = { front: number; home: number; cells: [number, number]; rest: [number, number, number, number]; cover: number };

type MeshGL = {
  drawMesh: (data: Float32Array, vertices: number, view: number, strength: number, shadeOn: boolean, luma: boolean, median: number, screenField: boolean, cellUvSize: [number, number], motion: number, territory: TerritoryFrame | null, flow: FlowDraw | null) => void;
  drawLines: (data: Float32Array, vertices: number) => void;
  /** The cloth wipe's quad grid. The indices only change with the bake's
   *  grid, so they are uploaded on their own. */
  uploadClothIndex: (indices: Uint16Array) => void;
  /** The word's distance field, which the cloth shader reads per pixel. */
  uploadClothField: (sdf: Float32Array, w: number, h: number) => void;
  drawCloth: (data: Float32Array, indices: number, draw: ClothDraw) => void;
  /** The mesh wipe's triangles, once a word, and its holes' field. */
  uploadClothMeshIndex: (indices: Uint16Array) => void;
  uploadHoles: (bytes: Uint8Array) => void;
  /** The floor, then the cloth if there is one yet, with depth. */
  drawClothMesh: (data: Float32Array | null, indices: number, draw: ClothMeshDraw) => void;
  /** The collage the flow carries: one clip per lattice cell, row by row. */
  uploadCollage: (clips: Uint8Array, width: number, height: number) => void;
  /** Upload the current frames. True once every required clip has one. With
   *  `every` off, clips outside `required` are skipped. */
  uploadVideos: (required: number[], every: boolean) => boolean;
  contextLost: () => boolean;
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
  const cloth = link(gl, CLOTH_VS, derivativeExtension ? CLOTH_FS : CLOTH_FS_FALLBACK);
  const buffer = gl.createBuffer();
  const lineBuffer = gl.createBuffer();
  const clothBuffer = gl.createBuffer();
  const clothIndexBuffer = gl.createBuffer();
  const clothMesh = link(gl, CLOTH_MESH_VS, CLOTH_MESH_FS);
  const clothMeshBuffer = gl.createBuffer();
  const clothMeshIndexBuffer = gl.createBuffer();
  const floorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, floorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    0, 0, 0, 0, 0, 0, 0, 1, 0,
    CANVAS_W, 0, 0, CANVAS_W, 0, 0, 0, 1, 0,
    0, CANVAS_H, 0, 0, CANVAS_H, 0, 0, 1, 0,
    CANVAS_W, CANVAS_H, 0, CANVAS_W, CANVAS_H, 0, 0, 1, 0,
  ]), gl.STATIC_DRAW);
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
  // The collage takes the unit after the clips. Its cells are read exactly,
  // one byte each, so rows must not be padded to four bytes.
  const collageUnit = videos.length;
  const collageTexture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + collageUnit);
  gl.bindTexture(gl.TEXTURE_2D, collageTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, new Uint8Array([0]));

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
  const uTerritory = gl.getUniformLocation(mesh, 'uTerritory');
  const uTerritoryPhase = gl.getUniformLocation(mesh, 'uTerritoryPhase');
  const uBorder = gl.getUniformLocation(mesh, 'uBorder');
  const borderData = new Float32Array(MAX_BORDERS * 4);
  const uFlood = gl.getUniformLocation(mesh, 'uFlood');
  const uJumbleGrid = gl.getUniformLocation(mesh, 'uJumbleGrid');
  const uRestCell = gl.getUniformLocation(mesh, 'uRestCell');
  const uCoverFront = gl.getUniformLocation(mesh, 'uCoverFront');
  gl.uniform1i(gl.getUniformLocation(mesh, 'uJumble'), collageUnit);
  const fitLoc = videos.map((_, i) => gl.getUniformLocation(mesh, `uFit${i}`));
  videos.forEach((_, i) => gl.uniform1i(gl.getUniformLocation(mesh, `clip${i}`), i));
  const aPos = gl.getAttribLocation(mesh, 'aPos');
  const aUV = gl.getAttribLocation(mesh, 'aUV');
  const aData = gl.getAttribLocation(mesh, 'aData');
  const aDebug = gl.getAttribLocation(mesh, 'aDebug');
  const aTransition = gl.getAttribLocation(mesh, 'aTransition');
  const aShadeUV = gl.getAttribLocation(mesh, 'aShadeUV');
  const aFlow = gl.getAttribLocation(mesh, 'aFlow');
  gl.useProgram(lines);
  const uLineCanvas = gl.getUniformLocation(lines, 'uCanvas');
  const uLineColor = gl.getUniformLocation(lines, 'uColor');
  const aLinePos = gl.getAttribLocation(lines, 'aPos');
  gl.useProgram(cloth);
  const uClothCanvas = gl.getUniformLocation(cloth, 'uCanvas');
  const uClothCover = gl.getUniformLocation(cloth, 'uCover');
  const uClothUnder = gl.getUniformLocation(cloth, 'uUnder');
  const uClothStrength = gl.getUniformLocation(cloth, 'uStrength');
  const uClothRelief = gl.getUniformLocation(cloth, 'uRelief');
  const uClothDepth = gl.getUniformLocation(cloth, 'uDepth');
  const uClothFront = gl.getUniformLocation(cloth, 'uFront');
  const uClothKeep = gl.getUniformLocation(cloth, 'uKeep');
  const clothFitLoc = videos.map((_, i) => gl.getUniformLocation(cloth, `uFit${i}`));
  videos.forEach((_, i) => gl.uniform1i(gl.getUniformLocation(cloth, `clip${i}`), i));
  const aClothPos = gl.getAttribLocation(cloth, 'aPos');
  const aClothUV = gl.getAttribLocation(cloth, 'aUV');
  const aClothShade = gl.getAttribLocation(cloth, 'aShade');
  const aClothSlope = gl.getAttribLocation(cloth, 'aSlope');
  const aClothCover = gl.getAttribLocation(cloth, 'aCover');
  const uClothFieldSize = gl.getUniformLocation(cloth, 'uFieldSize');
  const uClothFieldMap = gl.getUniformLocation(cloth, 'uFieldMap');
  const uClothFrame = gl.getUniformLocation(cloth, 'uFrame');
  const clothFieldUnit = collageUnit + 1;
  const clothFieldTexture = gl.createTexture();
  let clothFieldSize: [number, number] = [1, 1];
  gl.useProgram(cloth);
  gl.uniform1i(gl.getUniformLocation(cloth, 'uField'), clothFieldUnit);

  gl.useProgram(clothMesh);
  const pourLoc = (name: string) => gl.getUniformLocation(clothMesh, name);
  const uPourCanvas = pourLoc('uCanvas');
  const uPourSink = pourLoc('uSink');
  const uPourFloor = pourLoc('uFloor');
  const uPourCover = pourLoc('uCover');
  const uPourUnder = pourLoc('uUnder');
  const uPourRelief = pourLoc('uRelief');
  const uPourSheetZ = pourLoc('uSheetZ');
  const uPourFloorDepth = pourLoc('uFloorDepth');
  const uPourHeal = pourLoc('uHeal');
  const uPourVoid = pourLoc('uVoid');
  const uPourLift = pourLoc('uLift');
  const uPourLetterDy = pourLoc('uLetterDy');
  const uPourLetterTop = pourLoc('uLetterTop');
  const uPourBevel = pourLoc('uBevel');
  const uPourMaterial = pourLoc('uMaterial');
  const uPourSheen = pourLoc('uSheen');
  const uPourCavity = pourLoc('uCavity');
  const uPourFrame = pourLoc('uFrame');
  const uPourMargin = pourLoc('uMargin');
  const pourFitLoc = videos.map((_, i) => pourLoc(`uFit${i}`));
  videos.forEach((_, i) => gl.uniform1i(pourLoc(`clip${i}`), i));
  const holesUnit = collageUnit + 2;
  const holesTexture = gl.createTexture();
  gl.uniform1i(pourLoc('uHoles'), holesUnit);
  const aPourPos = gl.getAttribLocation(clothMesh, 'aPos');
  const aPourRest = gl.getAttribLocation(clothMesh, 'aRest');
  const aPourNormal = gl.getAttribLocation(clothMesh, 'aNormal');
  const aPourCavity = gl.getAttribLocation(clothMesh, 'aCavity');

  const fit = (video: HTMLVideoElement) => {
    const vw = video.videoWidth || CANVAS_W;
    const vh = video.videoHeight || CANVAS_H;
    const target = CANVAS_W / CANVAS_H;
    const source = vw / vh;
    return source > target ? [target / source, 1] : [1, source / target];
  };

  return {
    screenField: derivativeExtension,
    contextLost: () => gl.isContextLost(),
    clear(r, g, b) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(r, g, b, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    },
    uploadVideos(required, every) {
      if (gl.isContextLost()) return false;
      let ready = required.every((clip) => clip >= 0 && clip < videos.length);
      videos.forEach((video, index) => {
        const needed = required.includes(index);
        if (!every && !needed) return;
        if (video.readyState < 2) {
          if (needed) ready = false;
          return;
        }
        if (video.currentTime === lastTime[index]) return;
        try {
          gl.activeTexture(gl.TEXTURE0 + index);
          gl.bindTexture(gl.TEXTURE_2D, textures[index]);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
          if (gl.getError() === gl.NO_ERROR) lastTime[index] = video.currentTime;
          else if (needed) ready = false;
        } catch {
          // A decoder can temporarily expose a frame WebGL cannot upload.
          // Keep the previous texture and retry on the next frame.
          if (needed) ready = false;
        }
      });
      return ready;
    },
    uploadCollage(clips, width, height) {
      if (gl.isContextLost()) return;
      gl.activeTexture(gl.TEXTURE0 + collageUnit);
      gl.bindTexture(gl.TEXTURE_2D, collageTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, width, height, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, clips);
    },
    drawMesh(data, vertices, view, strength, shadeOn, luma, median, screenField, cellUvSize, motion, territory, flow) {
      gl.useProgram(mesh);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      const stride = VERTEX_FLOATS * 4;
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
      gl.enableVertexAttribArray(aFlow);
      gl.vertexAttribPointer(aFlow, 4, gl.FLOAT, false, stride, 72);
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
      gl.uniform4f(uTerritory, territory ? 1 : 0, territory?.breath ?? 0, territory?.count ?? 0, territory?.clips[0] ?? 0);
      gl.uniform3f(uTerritoryPhase, territory?.phase.six ?? 0, territory?.phase.nine ?? 0, territory?.shared ?? 0);
      borderData.fill(0);
      for (let s = 0; territory && s < territory.count; s++) {
        borderData[4 * s] = territory.base[s];
        borderData[4 * s + 1] = territory.shape[s];
        borderData[4 * s + 2] = territory.damp[s];
        borderData[4 * s + 3] = territory.clips[s + 1];
      }
      gl.uniform4fv(uBorder, borderData);
      gl.uniform4f(uFlood, flow ? 1 : 0, flow?.front ?? 0, 0, flow?.home ?? 0);
      gl.uniform2f(uJumbleGrid, flow?.cells[0] ?? 1, flow?.cells[1] ?? 1);
      const [restX = 1, restY = 1, restU = 0, restV = 0] = flow?.rest ?? [];
      gl.uniform4f(uRestCell, restX, restY, restU, restV);
      gl.uniform1f(uCoverFront, flow?.cover ?? 0);
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
      gl.disableVertexAttribArray(aFlow);
    },
    uploadClothIndex(indices) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, clothIndexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    },
    uploadClothField(sdf, w, h) {
      if (gl.isContextLost()) return;
      const bytes = new Uint8Array(w * h * 2);
      for (let k = 0; k < w * h; k++) {
        const v = Math.max(0, Math.min(65535, Math.round((sdf[k] + CLOTH_FIELD_RANGE) * CLOTH_FIELD_STEPS)));
        bytes[2 * k] = v >> 8;
        bytes[2 * k + 1] = v & 255;
      }
      gl.activeTexture(gl.TEXTURE0 + clothFieldUnit);
      gl.bindTexture(gl.TEXTURE_2D, clothFieldTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE_ALPHA, w, h, 0, gl.LUMINANCE_ALPHA, gl.UNSIGNED_BYTE, bytes);
      clothFieldSize = [w, h];
    },
    drawCloth(data, indices, draw) {
      gl.useProgram(cloth);
      gl.bindBuffer(gl.ARRAY_BUFFER, clothBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      const stride = CLOTH_FLOATS * 4;
      gl.enableVertexAttribArray(aClothPos);
      gl.vertexAttribPointer(aClothPos, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(aClothUV);
      gl.vertexAttribPointer(aClothUV, 2, gl.FLOAT, false, stride, 8);
      gl.enableVertexAttribArray(aClothShade);
      gl.vertexAttribPointer(aClothShade, 2, gl.FLOAT, false, stride, 16);
      gl.enableVertexAttribArray(aClothSlope);
      gl.vertexAttribPointer(aClothSlope, 2, gl.FLOAT, false, stride, 24);
      gl.enableVertexAttribArray(aClothCover);
      gl.vertexAttribPointer(aClothCover, 1, gl.FLOAT, false, stride, 32);
      gl.uniform2f(uClothCanvas, CANVAS_W, CANVAS_H);
      gl.uniform2f(uClothFrame, CANVAS_W, CANVAS_H);
      gl.uniform2f(uClothFieldSize, clothFieldSize[0], clothFieldSize[1]);
      gl.uniform2f(uClothFieldMap, SDF_PAD, SDF_SCALE);
      gl.activeTexture(gl.TEXTURE0 + clothFieldUnit);
      gl.bindTexture(gl.TEXTURE_2D, clothFieldTexture);
      gl.uniform1f(uClothCover, draw.cover);
      gl.uniform1f(uClothUnder, draw.under);
      gl.uniform1f(uClothStrength, draw.strength);
      gl.uniform1f(uClothRelief, draw.relief);
      gl.uniform1f(uClothDepth, draw.depth);
      gl.uniform1f(uClothFront, draw.front);
      gl.uniform1f(uClothKeep, draw.keep);
      videos.forEach((video, index) => {
        const [sx, sy] = fit(video);
        gl.uniform4f(clothFitLoc[index], sx, sy, 0, 0);
      });
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, clothIndexBuffer);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.drawElements(gl.TRIANGLES, indices, gl.UNSIGNED_SHORT, 0);
      gl.disableVertexAttribArray(aClothUV);
      gl.disableVertexAttribArray(aClothShade);
      gl.disableVertexAttribArray(aClothSlope);
      gl.disableVertexAttribArray(aClothCover);
    },
    uploadClothMeshIndex(indices) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, clothMeshIndexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    },
    uploadHoles(bytes) {
      if (gl.isContextLost()) return;
      gl.activeTexture(gl.TEXTURE0 + holesUnit);
      gl.bindTexture(gl.TEXTURE_2D, holesTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, CANVAS_W, CANVAS_H, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, bytes);
    },
    drawClothMesh(data, indices, draw) {
      gl.useProgram(clothMesh);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.043, 0.051, 0.063, 1);
      gl.clearDepth(1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.uniform2f(uPourCanvas, CANVAS_W, CANVAS_H);
      gl.uniform1f(uPourCover, draw.cover);
      gl.uniform1f(uPourUnder, draw.under);
      gl.uniform1f(uPourRelief, draw.relief);
      gl.uniform1f(uPourSheetZ, draw.sheetZ);
      gl.uniform1f(uPourFloorDepth, draw.floorDepth);
      gl.uniform1f(uPourHeal, draw.heal);
      gl.uniform3f(uPourVoid, 0.043, 0.051, 0.063);
      gl.uniform1f(uPourLift, draw.lift ?? 0);
      gl.uniform1f(uPourMargin, draw.margin ? 1 : 0);
      gl.uniform1f(uPourLetterDy, draw.letterDy ?? 0);
      gl.uniform1f(uPourLetterTop, draw.letterTop ?? 0);
      gl.uniform1f(uPourBevel, draw.bevel ?? 0);
      gl.uniform1f(uPourMaterial, draw.material ?? 0);
      gl.uniform1f(uPourSheen, draw.sheen ?? 0);
      gl.uniform1f(uPourCavity, draw.cavity ?? 0);
      gl.uniform2f(uPourFrame, CANVAS_W, CANVAS_H);
      videos.forEach((video, index) => {
        const [sx, sy] = fit(video);
        gl.uniform4f(pourFitLoc[index], sx, sy, 0, 0);
      });
      gl.activeTexture(gl.TEXTURE0 + holesUnit);
      gl.bindTexture(gl.TEXTURE_2D, holesTexture);
      const stride = CLOTH_MESH_FLOATS * 4;
      const attributes = () => {
        gl.enableVertexAttribArray(aPourPos);
        gl.vertexAttribPointer(aPourPos, 3, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(aPourRest);
        gl.vertexAttribPointer(aPourRest, 2, gl.FLOAT, false, stride, 12);
        gl.enableVertexAttribArray(aPourNormal);
        gl.vertexAttribPointer(aPourNormal, 3, gl.FLOAT, false, stride, 20);
        gl.enableVertexAttribArray(aPourCavity);
        gl.vertexAttribPointer(aPourCavity, 1, gl.FLOAT, false, stride, 32);
      };
      // the floor, never sunk
      gl.bindBuffer(gl.ARRAY_BUFFER, floorBuffer);
      attributes();
      gl.uniform1f(uPourFloor, 1);
      gl.uniform1f(uPourSink, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      if (data && indices > 0) {
        gl.bindBuffer(gl.ARRAY_BUFFER, clothMeshBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        attributes();
        gl.uniform1f(uPourFloor, 0);
        gl.uniform1f(uPourSink, draw.sink);
        if (draw.arrays) {
          gl.drawArrays(gl.TRIANGLES, 0, draw.arrays);
        } else {
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, clothMeshIndexBuffer);
          gl.drawElements(gl.TRIANGLES, indices, gl.UNSIGNED_SHORT, 0);
        }
      }
      gl.disableVertexAttribArray(aPourRest);
      gl.disableVertexAttribArray(aPourNormal);
      gl.disableVertexAttribArray(aPourCavity);
      gl.disable(gl.DEPTH_TEST);
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
  ready: boolean;
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
  /** Mean share of the canvas width held by the featured territory, over
   *  the visible rows; 0 outside the territories footage mode. */
  territoryShare: number;
  note: string;
};

export type MeshShowcase = {
  setParams: (next: MeshParams) => void;
  /** Draw at `clock` seconds into this project's cycle. `motionClock` is a
   *  clock that never wraps, and `sequenceIndex` counts projects shown so
   *  far. Territories breathe on the one and key their borders to the other,
   *  so consecutive projects' displays meet exactly at a handover. */
  frame: (clock: number, solveBudgetMs?: number, motionClock?: number, sequenceIndex?: number) => void;
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
  const clipColors = ['rgb(240, 102, 51)', 'rgb(38, 161, 217)', 'rgb(186, 214, 79)', 'rgb(168, 110, 227)', 'rgb(235, 138, 148)'];
  for (let cell = 0; cell + 6 * VERTEX_FLOATS <= vertexData.length; cell += 6 * VERTEX_FLOATS) {
    for (let tri = 0; tri < 2; tri++) {
      const a = cell + tri * 3 * VERTEX_FLOATS;
      const b = a + VERTEX_FLOATS;
      const c = a + 2 * VERTEX_FLOATS;
      const raw = vertexData[a + 8];
      const clampRatio = Math.max(0, Math.min(2.5, Math.abs(vertexData[a + 9])));
      const finalMultiplier = Math.max(0, Math.min(2, vertexData[a + 10]));
      const displacement = Math.max(0, Math.min(1, vertexData[a + 6]));
      const invalid = vertexData[a + 7] > 0.5;
      const clip = Math.max(0, Math.min(clipColors.length - 1, Math.round(vertexData[a + (vertexData[a + 12] >= 0.5 ? 11 : 5)])));
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
  backgroundSolve = false,
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
  // The flow, per node: the distance to the word the frontier reads, a
  // smoothed plain distance for the push direction, and where the node reads
  // the collage, as a fraction of the lattice. Then the collage itself, per
  // cell.
  let flowDist = new Float32Array(0);
  let flowSmooth = new Float32Array(0);
  let flowScratch = new Float32Array(0);
  let flowU = new Float32Array(0);
  let flowV = new Float32Array(0);
  let collage = new Uint8Array(0);
  // The cloth wipe's grid, which is the bake's own: the material point each
  // grid point shows, its sag and its squeeze, then the packed vertices.
  let clothKey = '';
  // the mesh wipe's word, buffers and hole depth
  let meshWord = '';
  let meshData = new Float32Array(0);
  let meshPos = new Float32Array(0);
  let meshNormal = new Float32Array(0);
  let meshIndices = 0;
  // the lift's triangles unshared, so each can carry its own facet
  let meshFlat = new Float32Array(0);
  // each point's neighbours, compressed, and its cavity: how far below them
  let meshNbrStart = new Int32Array(0);
  let meshNbr = new Int32Array(0);
  let meshCavity = new Float32Array(0);
  let meshCavityTmp = new Float32Array(0);
  let holesDeepest = 0;
  let clothIndices = 0;
  let clothVertexData = new Float32Array(0);
  let clothMatX = new Float32Array(0);
  let clothMatY = new Float32Array(0);
  let clothSag = new Float32Array(0);
  let clothRatio = new Float32Array(0);
  let clothSlope = new Float32Array(0);
  // the bake's grid refined CLOTH_REFINE times, and the pass along x
  let fineMatX = new Float32Array(0);
  let fineMatY = new Float32Array(0);
  let fineSag = new Float32Array(0);
  let fineTmp = new Float32Array(0);
  let clothOffX = new Float32Array(0);
  let clothOffY = new Float32Array(0);
  let clothCover = new Float32Array(0);
  let fineCover = new Float32Array(0);
  let clothDistRest = new Float32Array(0);
  let clothDistMat = new Float32Array(0);
  let clothGlyph: Glyph | null = null;
  let clothGlyphKey = '';
  let clothFarthest = 0;
  let clothDeepest = 0;
  let cellFold = new Uint8Array(0);
  let cellDiagonal = new Uint8Array(0);
  // the territory borders at each lattice row, canvas px, MAX_BORDERS a row
  let rowBorders = new Float32Array(0);
  let motionClock = 0;
  let sequenceIndex = 0;
  let territoryShare = 0;
  let visible = true;
  let prepared = false;
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
  let worker: Worker | null = null;
  let workerDisabled = false;
  let workerProgress = 0;

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
    workerProgress = 0;
    worker?.terminate();
    worker = null;
    if (backgroundSolve && !workerDisabled && typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined') {
      try {
        const solvingKey = job.key;
        worker = new Worker(new URL('./showcase-solver-worker.ts', import.meta.url), { type: 'module' });
        worker.onmessage = (event: MessageEvent<{ type: string; progress?: number; geometry?: MeshSolveGeometry; error?: string }>) => {
          if (!job || job.key !== solvingKey) return;
          const message = event.data;
          if (message.type === 'progress') {
            workerProgress = message.progress ?? 0;
          } else if (message.type === 'done' && message.geometry) {
            const { lat, glyph, ux0, uy0, nodeDist } = message.geometry;
            solved = {
              lat, glyph, ux: new Float32Array(lat.nodes), uy: new Float32Array(lat.nodes),
              ux0, uy0, nodeDist, cellDist: new Float32Array(lat.nx * lat.ny),
              key: solvingKey, spread: -1, maxDisp: 1,
            };
            applySpread(solved, params.spread);
            solveMs = performance.now() - job.started;
            allocate(lat);
            lastPaletteKey = '';
            job = null;
            worker?.terminate();
            worker = null;
          } else if (message.type === 'error') {
            console.error(`Mesh solve failed: ${message.error}`);
            workerDisabled = true;
            beginSolve(params);
          }
        };
        worker.onerror = (event) => {
          console.error('Mesh worker failed', event);
          workerDisabled = true;
          beginSolve(params);
        };
        worker.postMessage(p);
        // The worker holds the word until it is done. Draw an undeformed
        // stand-in meanwhile, far from any letter, so the collage shows from
        // the first frame rather than an empty canvas.
        if (!solved || solved.lat.columns !== p.columns) {
          const lat = buildLattice(p.columns);
          const far = new Float32Array(4).fill(1e4);
          const none = new Float32Array(4);
          solved = {
            lat,
            glyph: {
              w: 2, h: 2, sdf: far, gx: none, gy: none, coh: none, curv: none, strokePx: 0, fontPx: 0,
              inkWidth: 0, inkHeight: 0, letterCenters: [{ x: CANVAS_W / 2, y: CANVAS_H / 2 }], empty: true,
            },
            ux: new Float32Array(lat.nodes), uy: new Float32Array(lat.nodes),
            ux0: new Float32Array(lat.nodes), uy0: new Float32Array(lat.nodes),
            nodeDist: new Float32Array(lat.nodes).fill(1e4), cellDist: new Float32Array(lat.nx * lat.ny),
            key: `${solvingKey}~pending`, spread: -1, maxDisp: 1,
          };
          allocate(lat);
          lastPaletteKey = '';
        }
      } catch (error) {
        console.error('Mesh worker unavailable', error);
        worker?.terminate();
        worker = null;
        workerDisabled = true;
      }
    }
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
    flowDist = new Float32Array(lat.nodes);
    flowSmooth = new Float32Array(lat.nodes);
    flowScratch = new Float32Array(lat.nodes);
    flowU = new Float32Array(lat.nodes);
    flowV = new Float32Array(lat.nodes);
    collage = new Uint8Array(cells);
    cellFold = new Uint8Array(cells);
    cellDiagonal = new Uint8Array(cells);
    vertexData = new Float32Array(cells * 6 * VERTEX_FLOATS);
    const cols = lat.nx + 1;
    const rows = lat.ny + 1;
    rowBorders = new Float32Array(rows * MAX_BORDERS);
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

  /** The cloth wipe. The bake's grid stands still on screen; the picture moves
   *  because every point reads the clip at the material point of the fabric
   *  visible there, and darkens as that material sinks into a letter. There is
   *  no glyph, no lattice and no force field: the fall writes the word. */
  /** The mesh wipe. The clip is plain until the fall; then the bake's cloth
   *  carries it, drawn as the mesh it is, and pours down the word's holes in
   *  the next clip. The word holds as those holes, which heal shut in the
   *  release, leaving the next clip alone: its own cover, so they meet. */
  function buildClothMeshFrame(clock: number) {
    const p = params;
    const c = cycleOf(p);
    const t = ((clock % c.loop) + c.loop) % c.loop;
    const amplitude = p.amplitudeAuto ? amplitudeAt(t, c, p.releaseShape) : p.amplitude;
    lastAmplitude = amplitude;
    lastStrength = 0;
    lastTime = t;
    const holding = p.amplitudeAuto && t >= c.t2;
    // the bake at its own pace, one frame per frame, through the fall
    const progress = !p.amplitudeAuto ? Math.max(0, Math.min(1, amplitude)) : t >= c.t1 ? 1 : Math.max(0, Math.min(1, (t - c.t0) / (c.t1 - c.t0)));
    // the release: done a little before the loop ends, so the last frames are
    // the next clip whole
    const exit = holding ? Math.max(0, Math.min(1, (t - c.t2) / ((c.t3 - c.t2) * 0.9))) : 0;
    const cover = p.homeClip;
    const under = p.clothUnder >= 0 ? p.clothUnder : p.homeClip;
    gl!.uploadVideos(requiredClips(p), false);
    const lifting = p.footage === 'clothlift';
    // ?full=1 compares the bakes before compact-showcase-cloth.py, when they
    // are in src/data/showcase-cloth-lift-full (they are kept outside the
    // site, in agent-work/showcase-cloth-archive/, and copied in to compare).
    const liftPrefix = LIFT_KNOBS.full && CLOTH_MESHES.has(LIFT_FULL_PREFIX + p.text) ? LIFT_FULL_PREFIX : LIFT_PREFIX;
    const mesh = clothMeshFor(lifting ? liftPrefix + p.text : p.text);
    if (!mesh) {
      // No mesh yet: the clip, whole, as the cloth at rest would show it.
      note = CLOTH_MESHES.has(lifting ? liftPrefix + p.text : p.text) ? 'cloth mesh · loading' : `cloth mesh · no bake for ${JSON.stringify(p.text)}`;
      gl!.drawClothMesh(null, 0, { cover, under: cover, relief: 0, sheetZ: 0, floorDepth: 1, heal: 1e4, sink: 0 });
      return;
    }
    const { bake } = mesh;
    const n = bake.points;
    const meshKey = (lifting ? liftPrefix : '') + p.text;
    if (meshWord !== meshKey) {
      meshData = new Float32Array(n * CLOTH_MESH_FLOATS);
      meshPos = new Float32Array(n * 3);
      meshNormal = new Float32Array(n * 3);
      gl!.uploadClothMeshIndex(mesh.tris);
      meshIndices = mesh.tris.length;
      // The glyph's field: the fall's holes, or the lift's letters to bevel.
      const field = glyphHoleField(buildGlyph(p.text, p.fontSize, p.fontAuto));
      gl!.uploadHoles(field.bytes);
      holesDeepest = field.deepest;
      meshWord = meshKey;
      // Neighbours from the triangles, for the cavity.
      const tri = mesh.tris;
      const count = new Int32Array(n);
      const pairs: number[] = [];
      const seen = new Set<number>();
      for (let i = 0; i < tri.length; i += 3) {
        for (const [a, b] of [[tri[i], tri[i + 1]], [tri[i + 1], tri[i + 2]], [tri[i + 2], tri[i]]]) {
          const key = a < b ? a * 65536 + b : b * 65536 + a;
          if (seen.has(key)) continue;
          seen.add(key);
          pairs.push(a, b);
          count[a]++;
          count[b]++;
        }
      }
      meshNbrStart = new Int32Array(n + 1);
      for (let k = 0; k < n; k++) meshNbrStart[k + 1] = meshNbrStart[k] + count[k];
      meshNbr = new Int32Array(meshNbrStart[n]);
      const fill = meshNbrStart.slice(0, n);
      for (let i = 0; i < pairs.length; i += 2) {
        meshNbr[fill[pairs[i]]++] = pairs[i + 1];
        meshNbr[fill[pairs[i + 1]]++] = pairs[i];
      }
      meshCavity = new Float32Array(n);
      meshCavityTmp = new Float32Array(n);
    }
    // After the fall the cloth straightens: every point goes back to where
    // it started, flat, just under the floor, so through the holes the word
    // is the old clip, whole and still, in the new one.
    const straightenFor = Math.max(0.05, Math.min(1, 0.5 * (c.t2 - c.t1)));
    const straighten = lifting || !p.amplitudeAuto || t < c.t1 ? 0 : t >= c.t2 ? 1 : smoothUnit(Math.min(1, (t - c.t1) / straightenFor));
    const flatZ = -0.3;
    // Between the two nearest exported frames. A lift plays its whole bake
    // across the segment: the rise over the page's fall, the hold over its
    // hold, the slide away over its release.
    let f = progress * (bake.frames.length - 1);
    let beyond = 0;
    let letterDy = 0;
    let letterTop = 0;
    let bevel = 0;
    if (lifting && bake.lift && p.amplitudeAuto) {
      // In the bake's own frame numbers, so frames the compactor dropped do
      // not change the pace; then the exported frames either side.
      const last = bake.frames.length - 1;
      const first = bake.frames[0];
      const end = bake.frames[last];
      const span = (a: number, b: number, from: number, to: number) => from + (to - from) * Math.max(0, Math.min(1, (t - a) / (b - a)));
      const target = t < c.t1 ? span(c.t0, c.t1, first, bake.lift.growEnd) : t < c.t2 ? span(c.t1, c.t2, bake.lift.growEnd, bake.lift.slideStart) : span(c.t2, c.t3, bake.lift.slideStart, end);
      let i = 0;
      while (i < last - 1 && bake.frames[i + 1] <= target) i++;
      f = last ? i + Math.max(0, Math.min(1, (target - bake.frames[i]) / (bake.frames[i + 1] - bake.frames[i]))) : 0;
      beyond = 0;
      // Where the bake's letters are at this frame, as it keyed them: rising
      // to their height by growEnd, then sliding away accelerating to the end.
      const lift = bake.lift as { growEnd: number; slideStart: number; height?: number; slide?: number };
      const g0 = Math.floor(f);
      const frame = bake.frames[g0] + (bake.frames[Math.min(last, g0 + 1)] - bake.frames[g0]) * (f - g0);
      const u = Math.min(1, (frame - 1) / Math.max(1, lift.growEnd - 1));
      const v = Math.max(0, (frame - lift.slideStart) / Math.max(1, end - lift.slideStart));
      const height = lift.height ?? 1.5;
      letterTop = -0.06 + (height + 0.06) * u * u * (3 - 2 * u);
      letterDy = (lift.slide ?? 12) * v * v * CLOTH_PX + beyond * (CANVAS_H + 200);
      // with the letters' height, squared: faint while they are low, full
      // once they stand, so the glow reads as the word coming up
      const risen = Math.max(0, letterTop) / height;
      bevel = risen * risen;
    }
    const f0 = Math.floor(f);
    const f1 = Math.min(bake.frames.length - 1, f0 + 1);
    const ft = f - f0;
    const a = f0 * n * 3;
    const b = f1 * n * 3;
    const xy = bake.offsetScale;
    const zs = bake.heightScale;
    for (let k = 0; k < n; k++) {
      const o = 3 * k;
      meshPos[o] = (mesh.frames[a + o] * (1 - ft) + mesh.frames[b + o] * ft) * xy;
      meshPos[o + 1] = (mesh.frames[a + o + 1] * (1 - ft) + mesh.frames[b + o + 1] * ft) * xy;
      meshPos[o + 2] = (mesh.frames[a + o + 2] * (1 - ft) + mesh.frames[b + o + 2] * ft) * zs;
      if (beyond > 0) meshPos[o + 1] += beyond * (CANVAS_H + 200);
      if (straighten > 0) {
        meshPos[o] += (mesh.rest[2 * k] * xy - meshPos[o]) * straighten;
        meshPos[o + 1] += (mesh.rest[2 * k + 1] * xy - meshPos[o + 1]) * straighten;
        meshPos[o + 2] += (flatZ - meshPos[o + 2]) * straighten;
      }
    }
    // Normals from the triangles, weighted by their area, with height in px
    // like the rest, so the light falls on the cloth's real slope.
    meshNormal.fill(0);
    const tris = mesh.tris;
    for (let i = 0; i < tris.length; i += 3) {
      const i0 = 3 * tris[i];
      const i1 = 3 * tris[i + 1];
      const i2 = 3 * tris[i + 2];
      const ux = meshPos[i1] - meshPos[i0];
      const uy = meshPos[i1 + 1] - meshPos[i0 + 1];
      const uz = (meshPos[i1 + 2] - meshPos[i0 + 2]) * CLOTH_PX;
      const vx = meshPos[i2] - meshPos[i0];
      const vy = meshPos[i2 + 1] - meshPos[i0 + 1];
      const vz = (meshPos[i2 + 2] - meshPos[i0 + 2]) * CLOTH_PX;
      // canvas y runs down, so flip it to make a right-handed normal
      const nx = -(uy * vz - uz * vy);
      const ny = -(uz * vx - ux * vz);
      const nz = -(ux * vy - uy * vx);
      for (const v of [i0, i1, i2]) {
        meshNormal[v] += nx;
        meshNormal[v + 1] += ny;
        meshNormal[v + 2] += nz;
      }
    }
    if (lifting && meshNbr.length) {
      // How far each point sits below its neighbours, in px, then smoothed
      // twice over them, so it stays broad enough to read at display size.
      for (let k = 0; k < n; k++) {
        let sum = 0;
        for (let e = meshNbrStart[k]; e < meshNbrStart[k + 1]; e++) sum += meshPos[3 * meshNbr[e] + 2];
        const m = meshNbrStart[k + 1] - meshNbrStart[k];
        meshCavity[k] = m ? (sum / m - meshPos[3 * k + 2]) * CLOTH_PX : 0;
      }
      for (let pass = 0; pass < 2; pass++) {
        for (let k = 0; k < n; k++) {
          let sum = meshCavity[k];
          for (let e = meshNbrStart[k]; e < meshNbrStart[k + 1]; e++) sum += meshCavity[meshNbr[e]];
          meshCavityTmp[k] = sum / (1 + meshNbrStart[k + 1] - meshNbrStart[k]);
        }
        meshCavity.set(meshCavityTmp);
      }
    }
    for (let k = 0; k < n; k++) {
      const o = 3 * k;
      const d = k * CLOTH_MESH_FLOATS;
      meshData[d] = meshPos[o];
      meshData[d + 1] = meshPos[o + 1];
      meshData[d + 2] = meshPos[o + 2];
      meshData[d + 3] = mesh.rest[2 * k] * xy;
      meshData[d + 4] = mesh.rest[2 * k + 1] * xy;
      meshData[d + 5] = meshNormal[o];
      meshData[d + 6] = meshNormal[o + 1];
      meshData[d + 7] = meshNormal[o + 2];
      meshData[d + 8] = lifting ? meshCavity[k] : 0;
    }
    // A lift draws its triangles unshared, each point's normal blended with
    // its triangle's own facet, so the creases over the letters stay crisp.
    let arrays = 0;
    const flat = lifting ? LIFT_KNOBS.flat : 0;
    if (flat > 0) {
      if (meshFlat.length !== tris.length * CLOTH_MESH_FLOATS) meshFlat = new Float32Array(tris.length * CLOTH_MESH_FLOATS);
      for (let i = 0; i < tris.length; i += 3) {
        const i0 = 3 * tris[i];
        const i1 = 3 * tris[i + 1];
        const i2 = 3 * tris[i + 2];
        const ux = meshPos[i1] - meshPos[i0];
        const uy = meshPos[i1 + 1] - meshPos[i0 + 1];
        const uz = (meshPos[i1 + 2] - meshPos[i0 + 2]) * CLOTH_PX;
        const vx = meshPos[i2] - meshPos[i0];
        const vy = meshPos[i2 + 1] - meshPos[i0 + 1];
        const vz = (meshPos[i2 + 2] - meshPos[i0 + 2]) * CLOTH_PX;
        let fx = -(uy * vz - uz * vy);
        let fy = -(uz * vx - ux * vz);
        let fz = -(ux * vy - uy * vx);
        const fl = Math.hypot(fx, fy, fz) || 1;
        fx /= fl;
        fy /= fl;
        fz /= fl;
        for (let c3 = 0; c3 < 3; c3++) {
          const k = tris[i + c3];
          const src = k * CLOTH_MESH_FLOATS;
          const dst = (i + c3) * CLOTH_MESH_FLOATS;
          const vl = Math.hypot(meshData[src + 5], meshData[src + 6], meshData[src + 7]) || 1;
          for (let q = 0; q < 5; q++) meshFlat[dst + q] = meshData[src + q];
          meshFlat[dst + 8] = meshData[src + 8];
          meshFlat[dst + 5] = (meshData[src + 5] / vl) * (1 - flat) + fx * flat;
          meshFlat[dst + 6] = (meshData[src + 6] / vl) * (1 - flat) + fy * flat;
          meshFlat[dst + 7] = (meshData[src + 7] / vl) * (1 - flat) + fz * flat;
        }
      }
      arrays = tris.length;
    }
    note = `cloth mesh · frame ${f.toFixed(1)}/${bake.frames.length - 1}${straighten > 0 && straighten < 1 ? ` · straightening ${(straighten * 100).toFixed(0)}%` : ''}${exit > 0 ? ` · healing ${(exit * 100).toFixed(0)}%` : ''}`;
    gl!.drawClothMesh(arrays ? meshFlat : meshData, meshIndices, {
      cover,
      under,
      relief: p.clothRelief,
      sheetZ: bake.sheetZ,
      floorDepth: bake.floorDepth,
      heal: lifting ? 1e4 : exit * (holesDeepest + 1),
      lift: lifting ? 1.5 : 0,
      letterDy,
      letterTop,
      bevel: bevel * LIFT_KNOBS.bevel,
      material: lifting ? bevel : 0,
      sheen: LIFT_KNOBS.sheen,
      cavity: LIFT_KNOBS.cavity,
      arrays,
      ...(lifting ? { relief: 1 } : {}),
      margin: lifting,
      sink: 0,
    });
  }

  function buildClothFrame(clock: number) {
    const p = params;
    const c = cycleOf(p);
    const t = ((clock % c.loop) + c.loop) % c.loop;
    const amplitude = p.amplitudeAuto ? amplitudeAt(t, c, p.releaseShape) : p.amplitude;
    const strength = p.amplitudeAuto ? strengthAt(t, c, p) : strengthWithAmplitude(p, amplitude);
    // Nothing is rewound here: this clip does not come back. The fall runs
    // once and stays down, and the release pulls the last of the fabric — the
    // letters themselves — down through the holes, leaving the clip underneath
    // alone on the frame. That is the next project's own clip, so the two meet
    // exactly.
    const holding = p.amplitudeAuto && t >= c.t2;
    const exit = holding ? 1 - Math.max(0, Math.min(1, amplitude)) : 0;
    lastAmplitude = amplitude;
    lastStrength = strength;
    lastTime = t;
    foldRate = 0;
    invalidRate = 0;
    diagonalBdRate = 0;
    smoothingPasses = 0;
    territoryShare = 0;
    // No bake, or none yet: the same grid with the fabric at rest, which is
    // the clip alone. The wipe starts and ends there, so nothing pops when it
    // lands.
    // The wipe's own pour when the word has one, else the flow's bake.
    const clothKeyWord = hasClothBake(POUR_PREFIX + p.text) ? POUR_PREFIX + p.text : p.text;
    const cloth = clothFor(clothKeyWord);
    const grid = cloth?.bake.grid;
    const nx = grid ? grid.width : 2;
    const ny = grid ? grid.height : 2;
    const xAt = (i: number) => (grid ? grid.x0 + i * grid.step : i * CANVAS_W);
    const yAt = (j: number) => (grid ? grid.y0 + j * grid.step : j * CANVAS_H);
    const size = nx * ny;
    const fnx = (nx - 1) * CLOTH_REFINE + 1;
    const fny = (ny - 1) * CLOTH_REFINE + 1;
    const fsize = fnx * fny;
    const fxAt = (c: number) => xAt(0) + (c / CLOTH_REFINE) * (xAt(1) - xAt(0));
    const fyAt = (r: number) => yAt(0) + (r / CLOTH_REFINE) * (yAt(1) - yAt(0));
    if (clothKey !== `${nx}x${ny}`) {
      clothVertexData = new Float32Array(fsize * CLOTH_FLOATS);
      clothMatX = new Float32Array(size);
      clothMatY = new Float32Array(size);
      clothSag = new Float32Array(size);
      fineMatX = new Float32Array(fsize);
      fineMatY = new Float32Array(fsize);
      fineSag = new Float32Array(fsize);
      fineTmp = new Float32Array(fnx * ny);
      clothOffX = new Float32Array(size);
      clothOffY = new Float32Array(size);
      clothCover = new Float32Array(size);
      fineCover = new Float32Array(fsize);
      clothRatio = new Float32Array(fsize);
      clothSlope = new Float32Array(fsize * 2);
      clothDistRest = new Float32Array(size);
      clothDistMat = new Float32Array(size);
      clothGlyphKey = '';
      const indices = new Uint16Array((fnx - 1) * (fny - 1) * 6);
      let n = 0;
      for (let j = 0; j < fny - 1; j++) {
        for (let i = 0; i < fnx - 1; i++) {
          const a = j * fnx + i;
          indices[n] = a;
          indices[n + 1] = a + 1;
          indices[n + 2] = a + fnx + 1;
          indices[n + 3] = a;
          indices[n + 4] = a + fnx + 1;
          indices[n + 5] = a + fnx;
          n += 6;
        }
      }
      gl!.uploadClothIndex(indices);
      clothIndices = indices.length;
      clothKey = `${nx}x${ny}`;
    }
    // The word itself, for the letters the fabric is kept in and for the
    // frontier it carries. It is the page's own glyph, at the size the bake
    // was traced from; nothing is solved on it.
    const glyphKey = `${p.text}|${p.fontSize}|${p.fontAuto}`;
    if (clothGlyphKey !== glyphKey) {
      clothGlyph = buildGlyph(p.text, p.fontSize, p.fontAuto);
      clothGlyphKey = glyphKey;
      gl!.uploadClothField(clothGlyph.sdf, clothGlyph.w, clothGlyph.h);
      clothFarthest = -Infinity;
      clothDeepest = Infinity;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const d = sampleField(clothGlyph.sdf, clothGlyph.w, clothGlyph.h, xAt(i), yAt(j));
          clothDistRest[j * nx + i] = d;
          clothFarthest = Math.max(clothFarthest, d);
          clothDeepest = Math.min(clothDeepest, d);
        }
      }
    }
    const glyph = clothGlyph!;
    // The bake plays at its own pace, one frame per frame, not on the eased
    // amplitude: the letters are falling under gravity, and easing them into
    // the hold would stop a falling cloth in mid-air.
    const progress = holding
      ? 1
      : p.amplitudeAuto
        ? Math.max(0, Math.min(1, (t - c.t0) / (c.t1 - c.t0)))
        : Math.max(0, Math.min(1, amplitude));
    let bakeFrame = 0;
    if (cloth) {
      const field = clothWipeFields(clothKeyWord, cloth);
      const count = cloth.bake.frames.length;
      bakeFrame = progress * (count - 1);
      const f0 = Math.floor(bakeFrame);
      const f1 = Math.min(count - 1, f0 + 1);
      const ft = bakeFrame - f0;
      const b0 = f0 * CLOTH_WIPE_GRIDS * size;
      const b1 = f1 * CLOTH_WIPE_GRIDS * size;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          clothMatX[k] = xAt(i) + field[b0 + k] * (1 - ft) + field[b1 + k] * ft;
          clothMatY[k] = yAt(j) + field[b0 + size + k] * (1 - ft) + field[b1 + size + k] * ft;
          const z = field[b0 + 2 * size + k] * (1 - ft) + field[b1 + 2 * size + k] * ft;
          // Sag below the sheet at rest, so an untouched fabric is unshaded.
          clothSag[k] = Math.max(0, field[2 * size + k] - z);
          clothCover[k] = field[b0 + 3 * size + k] * (1 - ft) + field[b1 + 3 * size + k] * ft;
        }
      }
    } else {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          clothMatX[k] = xAt(i);
          clothMatY[k] = yAt(j);
          clothSag[k] = 0;
          clothCover[k] = 1;
        }
      }
    }
    // Where the material at each grid point started from, as a distance to
    // the word, for the deepest of it: the release's end. The shader reads
    // the same distance per pixel.
    let matDeepest = Infinity;
    for (let k = 0; k < size; k++) {
      clothDistMat[k] = sampleField(glyph.sdf, glyph.w, glyph.h, clothMatX[k], clothMatY[k]);
      matDeepest = Math.min(matDeepest, clothDistMat[k]);
    }
    // The sheet, refined through the bake's own samples. What is refined is
    // the drag, not the material point: at rest the drag and the sag are all
    // 0, so the refined sheet is the grid itself to the bit and the plain
    // clip stays exact.
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        clothOffX[k] = clothMatX[k] - xAt(i);
        clothOffY[k] = clothMatY[k] - yAt(j);
      }
    }
    // fineMatX/Y hold the refined drag; the material point is the screen
    // point plus it, added only when packed.
    refineGrid(clothOffX, nx, ny, CLOTH_REFINE, fineTmp, fineMatX);
    refineGrid(clothOffY, nx, ny, CLOTH_REFINE, fineTmp, fineMatY);
    refineGrid(clothSag, nx, ny, CLOTH_REFINE, fineTmp, fineSag);
    refineGrid(clothCover, nx, ny, CLOTH_REFINE, fineTmp, fineCover);
    for (let k = 0; k < fsize; k++) if (fineSag[k] < 0) fineSag[k] = 0;
    // How much material each point of the screen now shows: the area the
    // material map takes, by central differences on the refined sheet. Above
    // 1 the fabric has been squeezed, which is where the letters are.
    for (let j = 0; j < fny; j++) {
      const jU = j > 0 ? j - 1 : j;
      const jD = j < fny - 1 ? j + 1 : j;
      const hy = fyAt(jD) - fyAt(jU);
      for (let i = 0; i < fnx; i++) {
        const iL = i > 0 ? i - 1 : i;
        const iR = i < fnx - 1 ? i + 1 : i;
        const hx = fxAt(iR) - fxAt(iL);
        const row = j * fnx;
        // the identity plus the drag's own derivative, so flat fabric is 1
        // exactly rather than to float precision
        const dxu = 1 + (fineMatX[row + iR] - fineMatX[row + iL]) / hx;
        const dyu = (fineMatY[row + iR] - fineMatY[row + iL]) / hx;
        const dxv = (fineMatX[jD * fnx + i] - fineMatX[jU * fnx + i]) / hy;
        const dyv = 1 + (fineMatY[jD * fnx + i] - fineMatY[jU * fnx + i]) / hy;
        const det = Math.abs(dxu * dyv - dxv * dyu);
        clothRatio[row + i] = Math.max(DET_LO, Math.min(DET_HI, det));
        // The sheet's slope in canvas px per canvas px, up out of the screen,
        // which is what lights the pleats and the walls of the holes.
        clothSlope[2 * (row + i)] = (-CLOTH_PX * (fineSag[row + iR] - fineSag[row + iL])) / hx;
        clothSlope[2 * (row + i) + 1] = (-CLOTH_PX * (fineSag[jD * fnx + i] - fineSag[jU * fnx + i])) / hy;
      }
    }
    // Normalise by the median, as the mesh does, so the fabric's stretch
    // cannot brighten or dim the frame as a whole.
    const stride = Math.max(1, Math.floor(fsize / 1024));
    const sample: number[] = [];
    for (let k = 0; k < fsize; k += stride) sample.push(clothRatio[k]);
    sample.sort((x, y) => x - y);
    const median = sample.length ? sample[sample.length >> 1] : 1;
    medianRatio = median;
    const inv = 1 / Math.max(1e-6, median);
    for (let j = 0; j < fny; j++) {
      for (let i = 0; i < fnx; i++) {
        const k = j * fnx + i;
        const o = k * CLOTH_FLOATS;
        clothVertexData[o] = fxAt(i);
        clothVertexData[o + 1] = fyAt(j);
        clothVertexData[o + 2] = fxAt(i) + fineMatX[k];
        clothVertexData[o + 3] = fyAt(j) + fineMatY[k];
        clothVertexData[o + 4] = fineSag[k];
        clothVertexData[o + 5] = clothRatio[k] * inv;
        clothVertexData[o + 6] = clothSlope[2 * k];
        clothVertexData[o + 7] = clothSlope[2 * k + 1];
        clothVertexData[o + 8] = fineCover[k];
      }
    }
    // The frontier runs on the distance the fabric carries, as the flow's
    // does, from past the farthest ground at rest to past the deepest material
    // at the hold, so by then the only fabric left is what lies in the letters
    // themselves and the word's outline is the glyph's own. It comes in on the
    // square of the fall, so early on the sheet's own edge, dragged in by the
    // bake, is what uncovers the clip underneath. The release takes the keep
    // line down through the letters: the word goes outline first, pulled
    // through its own holes, and the frame is left on the clip underneath.
    // The shader reads the distances per pixel, between the grid points these
    // extremes were taken at, so each end keeps slack: a distance cannot
    // change by more than a px per px, and a stretched triangle of material
    // can span more than a cell.
    const cell = grid ? grid.step : 16;
    const start = clothFarthest + cell;
    const end = Math.min(-0.75 * cell, matDeepest) - 2 * cell;
    const front = start + (end - start) * progress * progress;
    const keep = (Math.min(0, clothDeepest) - cell) * exit;
    note = !cloth
      ? `cloth wipe · no bake for ${JSON.stringify(p.text)}`
      : Math.abs(cloth.bake.fontPx - glyph.fontPx) > 0.5
        ? `cloth wipe · bake is ${cloth.bake.fontPx.toFixed(0)} px, the word is ${glyph.fontPx.toFixed(0)} px`
        : `cloth wipe · bake frame ${bakeFrame.toFixed(1)}/${cloth.bake.frames.length - 1}`;
    gl!.uploadVideos(requiredClips(p), false);
    canvas.style.background = '#0b0d10';
    gl!.clear(0.043, 0.051, 0.063);
    gl!.drawCloth(clothVertexData, clothIndices, {
      cover: p.homeClip,
      under: p.clothUnder >= 0 ? p.clothUnder : p.homeClip,
      strength: signedExponent(p.polarity, strength),
      relief: p.clothRelief,
      depth: cloth?.bake.fall.depth ?? 1,
      front,
      keep,
    });
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
    const territories = p.footage === 'territories';
    const flowing = p.footage === 'jumble' && p.commit && p.commitTransition === 'flow';
    // Both hand one project's display to the next at zero amplitude with
    // nothing to hide the swap, so they breathe on the unbroken clock.
    const unbroken = territories || flowing;
    const settles = !territories && p.commit && (p.commitTransition === 'push' || p.commitTransition === 'flow');
    const stableHold = settles && (p.amplitudeAuto || amplitude >= 0.999);
    // The breathing settles out as the word forms, so the held word stands on
    // still footage. Held at a breathing pose, the lattice left a grid of
    // shaded dimples across the whole ground, and footage breathing through
    // that pose slid the video about under them. The settle follows the
    // amplitude alone, so both sides of a project swap breathe fully.
    const breathScale = settles ? 1 - smoothUnit(Math.min(1, Math.abs(amplitude)) / BREATH_SETTLE) : 1;
    if (stableHold && p.amplitudeAuto && t >= c.t1 && t < c.t2) {
      strength = strengthAt(c.t1, c, p);
    }
    // With nothing formed there is nothing to shade. These modes pass from one
    // project's display to the next at zero amplitude, where the release has
    // already taken the shading to zero, so idle has to meet it there.
    if (unbroken) strength *= smoothUnit(Math.abs(amplitude) / 0.12);
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
    // The hold drop moves the word by whole lattice rows, where the solved
    // field, the letter cells and the collage in them are exact copies moved
    // down, and draws the lattice shifted by the fraction of a row left over.
    // The footage, the standing collage and the breathing stay where they are
    // on screen. The drop lands on a whole row, so at rest nothing is shifted.
    // The word stays down through the release, but the collage it carried
    // down goes back up as the amplitude falls: at idle it must match the
    // standing collage, or the next project's first frame would jump.
    let dropRows = 0;
    let dropShift = 0;
    let dropCarry = 0;
    if (flowing && p.holdDrop) {
      const fall = holdDropAt(t, c, dropDistance(solved.glyph, lat, offsetY), p.dropBounce);
      dropRows = Math.floor(fall / lat.hy + 1e-6);
      dropShift = fall - dropRows * lat.hy;
      dropCarry = t >= c.t2 ? fall * smoothUnit(Math.max(0, Math.min(1, amplitude))) : fall;
      offsetY += dropRows * lat.hy;
    }
    // Whether a cell is a letter cell, with the word dropped by dropRows.
    const cellDist = solved.cellDist;
    const letterCell = (cell: number) => {
      const from = cell - dropRows * lat.nx;
      return from >= 0 && cellDist[from] < 0;
    };

    // 1. wobble, then push the wobbled lattice through the static field.
    // Territories and the flow breathe on the unbroken clock, so the next
    // project's display takes over mid-breath.
    const breathClock = unbroken ? motionClock : clock;
    const phase = (2 * Math.PI * breathClock * p.breathSpeed) / c.loop;
    const breath = p.breath * breathScale;
    if (!smoothWobble.length || smoothWobbleSeed !== p.seed) {
      smoothWobble = buildSmoothWobble(p.seed);
      smoothWobbleSeed = p.seed;
    }
    // Territories: the featured regions grow with the field and hand over on
    // release while every border keeps breathing. Only the featured
    // project's nodes take the displacement, through a gate that is smooth
    // across the margin, as in the prototype.
    const territory = territories ? territoryFrame(p, t, c, amplitude, motionClock, sequenceIndex) : null;
    const margin = TERRITORY_MARGIN_CELLS * lat.hx;
    if (territory) {
      let share = 0;
      let shareRows = 0;
      for (let j = 0; j < rows; j++) {
        const y = lat.restY[j * cols];
        const at = j * MAX_BORDERS;
        territoryBorders(territory, y, rowBorders, at);
        if (y < 0 || y > CANVAS_H) continue;
        for (let r = 0; r <= territory.count; r++) {
          if (!territory.featured[r]) continue;
          const left = r === 0 ? 0 : rowBorders[at + r - 1];
          const right = r === territory.count ? CANVAS_W : rowBorders[at + r];
          share += Math.max(0, Math.min(CANVAS_W, right) - Math.max(0, left)) / CANVAS_W;
        }
        shareRows += 1;
      }
      territoryShare = shareRows > 0 ? share / shareRows : 0;
    } else {
      territoryShare = 0;
    }
    const w = { x: 0, y: 0 };
    for (let k = 0; k < lat.nodes; k++) {
      const rx = lat.restX[k];
      const ry = lat.restY[k];
      const gate = territory ? territoryGate(territory, rowBorders, ((k / cols) | 0) * MAX_BORDERS, rx, margin) : 1;
      let bx = rx;
      let by = ry;
      if (breath > 0) {
        // G's image warp breathes on plane waves; F's wireframe breathes on
        // the per-node harmonics. Using F's here is what makes the footage
        // crawl instead of drape.
        if (p.algorithm === 'warp') smoothWobbleAt(smoothWobble, rx, ry + dropShift, phase, w);
        else wobbleAt(rx, ry + dropShift, lat.h, phase, w);
        bx += breath * lat.h * w.x;
        by += breath * lat.h * w.y;
      }
      if (amplitude !== 0 && gate > 0) {
        const dx = sampleDisp(lat, solved.ux, bx - offsetX, by - offsetY);
        const dy = sampleDisp(lat, solved.uy, bx - offsetX, by - offsetY);
        const reach = amplitude * gate;
        posX[k] = bx + reach * dx;
        posY[k] = by + reach * dy;
        // Extension seam for the direction-4/5 experiments: a future
        // optional relief or contour-lock pass can adjust this local sample
        // before packing. Both experiments stay zero in the faithful-G path.
        nodeDisp[k] = Math.min(1, (Math.hypot(dx, dy) * Math.abs(reach)) / solved.maxDisp);
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
    const pal = palette;

    // 4b. The flow. The collage is a pattern each pixel reads through a map,
    // so it can be carried instead of switched. The featured footage floods in
    // along the distance to the word, far first, and reaches the letters as
    // the word finishes forming. Just ahead of its edge it pushes the collage
    // inward, squeezed, until the collage slides under it, and what reaches
    // the letters is packed into them. Release runs the same map backwards.
    let flowFront = 0;
    let flowCover = 0;
    if (flowing && pal) {
      const glyph = solved.glyph;
      // Every node counts, overscan too: a pixel at the frame's edge reads a
      // distance partly from outside it, and at idle not one may be flooded.
      let farthest = -Infinity;
      for (let k = 0; k < lat.nodes; k++) {
        flowDist[k] = sampleField(glyph.sdf, glyph.w, glyph.h, lat.restX[k] - offsetX, lat.restY[k] - offsetY);
        farthest = Math.max(farthest, flowDist[k]);
      }
      // The push follows the distance field smoothed over about a cell. Its
      // gradient fades to nothing on the lines between and inside letters, so
      // the collage stretches there instead of tearing.
      flowSmooth.set(flowDist);
      for (let pass = 0; pass < 2; pass++) {
        for (let j = 0; j < rows; j++) {
          for (let i = 0; i < cols; i++) {
            const k = j * cols + i;
            flowScratch[k] = 0.25 * (flowSmooth[i > 0 ? k - 1 : k] + 2 * flowSmooth[k] + flowSmooth[i < cols - 1 ? k + 1 : k]);
          }
        }
        for (let j = 0; j < rows; j++) {
          for (let i = 0; i < cols; i++) {
            const k = j * cols + i;
            flowSmooth[k] = 0.25 * (flowScratch[j > 0 ? k - cols : k] + 2 * flowScratch[k] + flowScratch[j < rows - 1 ? k + cols : k]);
          }
        }
      }
      const progress = Math.max(0, Math.min(1, amplitude));
      const start = farthest + FLOW_WAVE + 2;
      const end = -0.75 * lat.h; // just inside the letters, past every ground pixel
      flowFront = start + (end - start) * progress;
      // Inside the letters the collage flowing in covers the featured pieces
      // standing there, from the contour to the core, before the hold. It
      // starts past every pixel of a letter cell, breathing included, so at
      // idle it covers nothing.
      const covered = smoothUnit((progress - 0.05) / 0.55);
      flowCover = (1 - covered) * (0.75 * lat.h + FLOW_WAVE + 1) - covered * 0.6 * solved.glyph.strokePx;
      const ramp = smoothUnit(progress / 0.05);
      const fade = Math.max(0, Math.min(1, flowFront / (3 * FLOW_BAND)));
      const flowPhase = territoryPhase(motionClock * p.breathSpeed);
      // Each project's word has a Blender cloth bake: a cloth on a floor,
      // which the letters pull into letter-shaped holes as they fall through
      // them. Seen from above, each point shows some material point of the
      // cloth. The frontier reads the word's distance there, so it rides the
      // fabric, at the fabric's own pace, and the collage can ride it too.
      // The bake never moves lattice vertices or changes which clips there
      // are. At idle it is off, so the swap stays exact; at the hold the lead
      // limit returns the flood to exactly the ground. A word without a bake,
      // or at another size, uses the plain frontier. The bake is asked for
      // from idle on, so it has landed before the first formation.
      const clothSource = p.clothFrontier !== 'off' ? clothFor(p.text, glyph.fontPx) : null;
      const clothData = progress > 0 ? clothSource : null;
      const clothHeight = p.clothFrontier === 'height' ? FLOW_BAKED_CLOTH_PX * 4 * progress * (1 - progress) : 0;
      const clothLead = FLOW_CLOTH_LEAD * (1 - progress);
      // The collage leaves the fabric before the hold, so the letters settle
      // into the same collage as without the cloth.
      const collageCarry = p.clothFrontier === 'collage' ? smoothUnit(progress / 0.08) * (1 - smoothUnit((progress - 0.8) / 0.2)) : 0;
      const riding = p.clothFrontier === 'frontier' || p.clothFrontier === 'collage';
      const sheet = { x: 0, y: 0, z: 0 };
      for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
          const k = j * cols + i;
          const rx = lat.restX[k];
          const ry = lat.restY[k];
          const plain = flowDist[k];
          let d = plain;
          sheet.x = 0;
          sheet.y = 0;
          if (clothData) {
            sampleCloth(clothData, rx - offsetX, ry - offsetY, progress, sheet);
            if (riding) {
              // The fabric runs at its own pace, often ahead of the plain
              // frontier, but never further from it than the lead, which
              // closes at the hold.
              const carried = sampleField(glyph.sdf, glyph.w, glyph.h, rx - offsetX + sheet.x, ry - offsetY + sheet.y);
              d = Math.max(plain - clothLead, Math.min(plain + clothLead, carried));
            } else {
              // Where the sheet sags, the outgoing collage holds nearer the letters.
              d += sheet.z * clothHeight;
            }
          }
          d -= (FLOW_WAVE / SHARED_REACH) * fade * sharedWave(rx / CANVAS_W, ry / CANVAS_H, flowPhase);
          flowDist[k] = d;
          const left = i > 0 ? k - 1 : k;
          const right = i < cols - 1 ? k + 1 : k;
          const up = j > 0 ? k - cols : k;
          const down = j < rows - 1 ? k + cols : k;
          const gx = (flowSmooth[right] - flowSmooth[left]) / ((right - left) * lat.hx);
          const gy = (flowSmooth[down] - flowSmooth[up]) / (((down - up) / cols) * lat.hy);
          // read the collage from further out, so it appears to move inward
          const push = ramp * FLOW_PUSH * Math.exp(Math.min(0, d - flowFront) / FLOW_BAND);
          // As an offset in cells from the node's own cell, exactly zero at
          // idle, so the moving collage then matches the standing one bit for bit.
          flowU[k] = (gx * push + collageCarry * sheet.x) / lat.hx;
          // The dropped word carries its collage down with it.
          flowV[k] = (gy * push + collageCarry * sheet.y - dropCarry) / lat.hy;
        }
      }
      for (let cell = 0; cell < collage.length; cell++) collage[cell] = pal.idle[cell];
      gl!.uploadCollage(collage, lat.nx, lat.ny);
    }

    // 5. pack the vertex buffer. Vertices are NOT shared between cells, so the
    //    per-cell clip index cannot bleed; the shared node position is copied
    //    into each duplicate, so the mesh stays watertight.
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
        if (territory) {
          // The shader picks the territory per pixel. This per-cell copy only
          // feeds the 2D inspection overlay.
          const cx = lat.x0 + (i + 0.5) * lat.hx;
          let region = 0;
          while (region < territory.count && cx >= 0.5 * (rowBorders[j * MAX_BORDERS + region] + rowBorders[(j + 1) * MAX_BORDERS + region])) region++;
          clip = territory.clips[region];
          nextClip = clip;
        } else if (flowing && pal) {
          // The shader reads the collage per pixel. This per-cell copy only
          // feeds the 2D inspection overlay.
          const inside = letterCell(cell);
          const flooded = !inside && 0.25 * (flowDist[a] + flowDist[b] + flowDist[cc] + flowDist[d]) >= flowFront;
          clip = flooded ? p.homeClip : pal.idle[cell];
          nextClip = clip;
        } else if (jumble && pal) {
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
        // letter cells keep the collage to the end; the flood never enters them
        const letter = flowing && letterCell(cell) ? 1 : 0;
        for (let v = 0; v < 6; v++) {
          const k = quad[v];
          data[o] = posX[k];
          data[o + 1] = posY[k] + dropShift;
          data[o + 2] = lat.restX[k] * invW;
          data[o + 3] = (lat.restY[k] + dropShift) * invH;
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
          data[o + 18] = flowing ? flowU[k] : 0;
          data[o + 19] = flowing ? flowV[k] : 0;
          data[o + 20] = flowing ? flowDist[k] : 0;
          data[o + 21] = letter;
          o += VERTEX_FLOATS;
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
          lineData[l + 1] = posY[k] + dropShift;
          lineData[l + 2] = posX[k + 1];
          lineData[l + 3] = posY[k + 1] + dropShift;
          l += 4;
        }
      }
      for (let j = 0; j < lat.ny; j++) {
        for (let i = 0; i < cols; i++) {
          const k = j * cols + i;
          lineData[l] = posX[k];
          lineData[l + 1] = posY[k] + dropShift;
          lineData[l + 2] = posX[k + cols];
          lineData[l + 3] = posY[k + cols] + dropShift;
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
      if (p.view === 0) gl!.uploadVideos(requiredClips(p), !territories);
      const motion = territories ? 0 : p.commitTransition === 'push' ? 2 : p.commitTransition === 'squeeze' ? 1 : 0;
      const flow: FlowDraw | null = flowing && pal
        ? {
          front: flowFront,
          home: p.homeClip,
          cells: [lat.nx, lat.ny],
          // the standing collage stays put on screen through the drop's shift
          rest: [CANVAS_W / lat.hx, CANVAS_H / lat.hy, -lat.x0 / lat.hx, (dropShift - lat.y0) / lat.hy],
          cover: flowCover,
        }
        : null;
      gl!.drawMesh(data, lat.nx * lat.ny * 6, p.view, signed, strength > 0, p.lumaAdaptive, median, gl!.screenField, [lat.hx / CANVAS_W, lat.hy / CANVAS_H], motion, territory, flow);
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
      if (solveKey(next) !== solveKey(params) || next.view !== params.view || requiredClips(next).join() !== requiredClips(params).join()) prepared = false;
      params = { ...next };
      if (params.footage === 'cloth' || params.footage === 'clothmesh' || params.footage === 'clothlift') {
        // The bake is the geometry: there is nothing to solve, and a solve
        // left running would only spend a worker on a word nobody draws.
        worker?.terminate();
        worker = null;
        job = null;
        return;
      }
      const key = solveKey(params);
      if (key !== (solved?.key ?? '')) {
        // The previous mesh stays on screen while the new one solves.
        if (!job || job.key !== key) beginSolve(params);
      } else if (job) {
        job = null;
      }
    },
    frame(clock, solveBudgetMs = 9, motion = clock, index = 0) {
      const started = performance.now();
      motionClock = motion;
      sequenceIndex = index;
      if (params.footage === 'cloth' || params.footage === 'clothmesh' || params.footage === 'clothlift') {
        const build = params.footage === 'cloth' ? buildClothFrame : buildClothMeshFrame;
        if (!visible && !prepared) {
          if (gl!.uploadVideos(requiredClips(params), false)) {
            build(0);
            if (debugCanvas) debugCanvas.hidden = true;
            prepared = !gl!.contextLost();
          }
        }
        if (visible) build(clock);
        frameMs = performance.now() - started;
        return;
      }
      measureLuma();
      if (job && !worker) advanceSolve(solveBudgetMs);
      if (solved) {
        if (solved.spread !== params.spread) applySpread(solved, params.spread);
        if (!palette || lastPaletteKey !== paletteKey(params)) {
          palette = buildPalette(solved, params, luma);
          lastPaletteKey = paletteKey(params);
        } else if (Math.abs(palette.offsetX - params.offsetX) > 0.5 || Math.abs(palette.offsetY - params.offsetY) > 0.5) {
          refreshPalette(palette, solved, params, luma, params.offsetX, params.offsetY);
        }
      }
      // Draw the incoming display once while hidden. Its clips and WebGL
      // buffer must exist before the opacity blend can reveal this canvas.
      if (solved && !job && !visible && !prepared) {
        if (params.view !== 0 || gl!.uploadVideos(requiredClips(params), params.footage !== 'territories')) {
          buildFrame(0);
          if (debugCanvas) debugCanvas.hidden = true;
          prepared = !gl!.contextLost();
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
        ready: prepared && !gl!.contextLost(),
        progress: job ? (worker ? workerProgress : job.stage === 'glyph' ? 0.02 : job.settle ? job.settle.done / job.settle.iterations : job.warp ? job.warp.done / job.warp.iterations : 0.99) : 1,
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
        vertices: params.footage === 'cloth' ? clothIndices : params.footage === 'clothmesh' || params.footage === 'clothlift' ? Math.max(1, meshIndices) : lat ? lat.nx * lat.ny * 6 : 0,
        territoryShare,
        note,
      };
    },
    solveNow() {
      if (worker) {
        workerDisabled = true;
        beginSolve(params);
      }
      let guard = 0;
      while (job && guard++ < 4000) advanceSolve(1000);
      if (solved) {
        palette = buildPalette(solved, params, luma);
        lastPaletteKey = paletteKey(params);
      }
    },
    setVisible(next) {
      visible = next;
      if (debugCanvas) debugCanvas.hidden = !next || params.view < 1 || params.view > 7;
      if (next) note = '';
    },
  };
}
