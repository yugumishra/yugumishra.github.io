import {
  createMeshShowcase,
  cycleOf,
  amplitudeAt,
  strengthAt,
  MESH_DEFAULTS,
  type MeshParams,
  type Algorithm,
  type WarpCorrespondence,
  type ReleaseShape,
  type Schedule,
  type Polarity,
  type JumbleShading,
  type FootageMode,
  type IdleField,
  type CommitTransition,
  type ClothFrontier,
  hasClothBake,
  liftBevel,
  setLiftBevel,
  type FamilyMode,
  type MeshShowcase,
} from './showcase-mesh';

type Settings = { view: number; copies: number };
type MaskSource = HTMLVideoElement | HTMLCanvasElement;
type Renderer = { draw: (settings: Settings, maskSource: MaskSource, maskChanged?: boolean) => void };
type PointTopology = 'contour-faces' | 'edge-splits' | 'lattice-tessellation' | 'pure-lattice';

type MeshPresetName = 'custom' | 'origin-flow' | 'cloth-wipe' | 'cloth-mesh' | 'cloth-lift' | 'territories' | 'faithful-g-single' | 'fixed-jumble' | 'origin-commit' | 'origin-squeeze' | 'origin-push' | 'family-hybrid';

/** The page opens on this preset, and Reset all returns to it. */
const DEFAULT_PRESET: Exclude<MeshPresetName, 'custom'> = 'origin-flow';

const MESH_PRESETS: Record<Exclude<MeshPresetName, 'custom'>, Partial<MeshParams>> = {
  // Origin push, with the collage carried instead of switched. The featured
  // footage floods in from far to near and pushes the collage ahead of it
  // until it is packed into the letters; release lets it flow back out. The
  // ends match origin push: the collage at idle, and at the hold the letters
  // full of collage over the featured project's footage.
  'origin-flow': {
    algorithm: 'settle', warpCorrespondence: 'offset', gain: 1, footage: 'jumble', clip: 0,
    commit: true, commitTransition: 'flow', clothFrontier: 'frontier', holdDrop: true, dropBounce: 0.45, idleField: 'jumble', familySplit: true, familyMode: 'origin', jumbleShading: 'on',
    columns: 88, strength: 0.55, polarity: 'compression-dark', smoothingPx: 3, stagger: 0.72, spread: 1, releaseShape: 'asymmetric',
    breath: 0.3, idle: 2, settle: 4.5, hold: 3, release: 3,
  },
  // The opening simplified: the clips play in order, one a project, plain,
  // and the only thing that happens is the transition out of each one. The
  // project's own clip is the fabric: its cloth bake drags it into the
  // letter-shaped holes of its name and pulls it through them, leaving the
  // next project's clip. No collage, no force field, no glyph; the squeeze
  // and the sag are the only shading.
  'cloth-wipe': {
    algorithm: 'settle', footage: 'cloth', clip: 0, commit: false, commitTransition: 'cut',
    familySplit: false, familyMode: 'origin', idleField: 'single', jumbleShading: 'on',
    clothRelief: 0.7, holdDrop: false, breath: 0, strength: 0.55, polarity: 'compression-dark',
    columns: 88, spread: 1, releaseShape: 'asymmetric',
    // Mostly the clip, plain: the fall is the transition out of it, not a
    // display. The word holds long enough to read, then is pulled through.
    idle: 6, settle: 2.5, hold: 2, release: 1.5,
  },
  // The cloth wipe drawn from the bake's own mesh (bake-showcase-cloth.py
  // --mesh): the clip on the real cloth, textured through its UVs and lit by
  // its normals, pouring down the word's holes in the next clip's floor. The
  // word holds as those holes, which heal shut in the release.
  'cloth-mesh': {
    algorithm: 'settle', footage: 'clothmesh', clip: 0, commit: false, commitTransition: 'cut',
    familySplit: false, familyMode: 'origin', idleField: 'single', jumbleShading: 'on',
    clothRelief: 0.7, holdDrop: false, breath: 0, strength: 0.55, polarity: 'compression-dark',
    columns: 88, spread: 1, releaseShape: 'asymmetric',
    // the hold: a second for the cloth to straighten under the holes, then
    // the word, still, long enough to read
    idle: 6, settle: 2.5, hold: 2.8, release: 1.5,
  },
  // The inverse (bake-showcase-cloth.py --lift): dished letters rise under
  // the resting cloth, the word holds draped over them, and they slide away
  // down the frame taking the cloth with them, the next clip left behind.
  'cloth-lift': {
    algorithm: 'settle', footage: 'clothlift', clip: 0, commit: false, commitTransition: 'cut',
    familySplit: false, familyMode: 'origin', idleField: 'single', jumbleShading: 'on',
    clothRelief: 0.7, holdDrop: false, breath: 0, strength: 0.55, polarity: 'compression-dark',
    columns: 88, spread: 1, releaseShape: 'asymmetric',
    idle: 6, settle: 2.5, hold: 2.8, release: 4,
  },
  // Prototype 3, revised (agent-work/prototype-3-broad-regions): three
  // breathing footage territories. The featured project's territory grows to
  // dominate the frame while only its own cells take F's pull toward the
  // letter contour, using the raw settle field the prototype baked. Each
  // project runs 12 s: 1.7 s idle, 3.8 s formation, 3.2 s hold, then a
  // 3.3 s release in which the next project's region sweeps in and takes the
  // middle. That release replaces the prototype's 2 s release and 1.3 s
  // crossfade seam with one continuous movement.
  territories: {
    algorithm: 'settle', warpCorrespondence: 'offset', gain: 1, footage: 'territories', clip: 0,
    commit: false, commitTransition: 'cut', idleField: 'jumble', familySplit: false, familyMode: 'luminance', jumbleShading: 'on',
    columns: 80, breath: 0.1, strength: 0.55, polarity: 'compression-dark', smoothingPx: 3, spread: 0, releaseShape: 'disperse',
    idle: 1.7, settle: 3.8, hold: 3.2, release: 3.3, territoryGrowth: 0.25, territoryBreath: 1, territoryRegions: 1,
  },
  'faithful-g-single': {
    algorithm: 'warp', warpCorrespondence: 'late-g-lens', gain: 0.8, footage: 'single', clip: 0,
    commit: false, commitTransition: 'cut', idleField: 'single', familySplit: false, familyMode: 'luminance', jumbleShading: 'on',
    columns: 112, breath: 0.35, strength: 0.55, flarePeak: 1.7, flareCentre: 0.45, spread: 0,
    // probe-g-global-warp.py RAMP_SHIP = (0.8, 5.3, 8.3, 11.3)
    polarity: 'compression-bright', smoothingPx: 3, idle: 0.8, settle: 4.5, hold: 3, release: 3, releaseShape: 'disperse',
  },
  // The F presets below were judged on the page's own breathing and cycle,
  // which the territories preset changes, so each one names them.
  'fixed-jumble': {
    algorithm: 'settle', warpCorrespondence: 'offset', gain: 1, footage: 'jumble', clip: 0,
    commit: false, commitTransition: 'cut', idleField: 'jumble', familySplit: false, familyMode: 'luminance', jumbleShading: 'on',
    columns: 88, strength: 0.55, polarity: 'compression-dark', smoothingPx: 3, spread: 1, releaseShape: 'disperse',
    breath: 0.3, idle: 2, settle: 4.5, hold: 3, release: 3,
  },
  // The fixed jumble, with commitment switched on and pooled by origin: the
  // ground resolves into the clip the word is for, the letters keep the other
  // sources. Nothing else moves, so it is a true A/B against the jumble.
  'origin-commit': {
    algorithm: 'settle', warpCorrespondence: 'offset', gain: 1, footage: 'jumble', clip: 0,
    commit: true, commitTransition: 'cut', idleField: 'jumble', familySplit: true, familyMode: 'origin', jumbleShading: 'on',
    columns: 88, strength: 0.55, polarity: 'compression-dark', smoothingPx: 3, stagger: 0.72, spread: 1, releaseShape: 'disperse',
    breath: 0.3, idle: 2, settle: 4.5, hold: 3, release: 3,
  },
  'origin-squeeze': {
    algorithm: 'settle', warpCorrespondence: 'offset', gain: 1, footage: 'jumble', clip: 0,
    commit: true, commitTransition: 'squeeze', idleField: 'jumble', familySplit: true, familyMode: 'origin', jumbleShading: 'on',
    columns: 88, strength: 0.55, polarity: 'compression-dark', smoothingPx: 3, stagger: 0.72, spread: 1, releaseShape: 'disperse',
    breath: 0.3, idle: 2, settle: 4.5, hold: 3, release: 3,
  },
  'origin-push': {
    algorithm: 'settle', warpCorrespondence: 'offset', gain: 1, footage: 'jumble', clip: 0,
    commit: true, commitTransition: 'push', idleField: 'jumble', familySplit: true, familyMode: 'origin', jumbleShading: 'on',
    columns: 88, strength: 0.55, polarity: 'compression-dark', smoothingPx: 3, stagger: 0.72, spread: 1, releaseShape: 'asymmetric',
    breath: 0.3, idle: 2, settle: 4.5, hold: 3, release: 3,
  },
  'family-hybrid': {
    algorithm: 'settle', warpCorrespondence: 'offset', gain: 1, footage: 'jumble', clip: 0,
    commit: true, commitTransition: 'cut', idleField: 'jumble', familySplit: true, familyMode: 'luminance', jumbleShading: 'on',
    columns: 64, strength: 0.55, polarity: 'compression-dark', smoothingPx: 3, spread: 1, releaseShape: 'disperse',
    breath: 0.3, idle: 2, settle: 4.5, hold: 3, release: 3,
  },
};

const COLORS = [[0.94, 0.40, 0.20], [0.15, 0.63, 0.85], [0.73, 0.84, 0.31], [0.66, 0.43, 0.89]];
const MASK_WIDTH = 1280;
const MASK_HEIGHT = 720;
const MASK_DURATION = 12;
const MAX_COPIES = 12;
const ATLAS_COLUMNS = 6;
const ATLAS_ROWS = Math.ceil(MAX_COPIES / ATLAS_COLUMNS);
const ATLAS_WIDTH = 640;
const ATLAS_HEIGHT = 360;

const COPY_TRANSFORMS = [
  { zoom: 1.00, x: 0.00, y: 0.00 },
  { zoom: 1.12, x: -0.10, y: 0.08 },
  { zoom: 1.24, x: 0.12, y: -0.08 },
  { zoom: 1.36, x: -0.15, y: -0.12 },
  { zoom: 1.48, x: 0.08, y: 0.12 },
  { zoom: 1.60, x: -0.04, y: 0.02 },
  { zoom: 1.18, x: 0.16, y: 0.14 },
  { zoom: 1.30, x: -0.18, y: 0.04 },
  { zoom: 1.44, x: 0.05, y: -0.16 },
  { zoom: 1.56, x: 0.14, y: -0.02 },
  { zoom: 1.70, x: -0.12, y: 0.12 },
  { zoom: 1.84, x: 0.02, y: -0.12 },
];

function drawFootageAtlas(
  video: HTMLVideoElement,
  context: CanvasRenderingContext2D,
  sourceIndex: number,
  copies: number,
) {
  context.imageSmoothingEnabled = true;
  for (let copy = 0; copy < copies; copy++) {
    const transform = COPY_TRANSFORMS[(copy + sourceIndex) % COPY_TRANSFORMS.length];
    const xOffset = (copy % ATLAS_COLUMNS) * ATLAS_WIDTH;
    const yOffset = Math.floor(copy / ATLAS_COLUMNS) * ATLAS_HEIGHT;
    const scale = Math.max(ATLAS_WIDTH / video.videoWidth, ATLAS_HEIGHT / video.videoHeight) * transform.zoom;
    const width = video.videoWidth * scale;
    const height = video.videoHeight * scale;
    context.fillStyle = '#0b0d10';
    context.fillRect(xOffset, yOffset, ATLAS_WIDTH, ATLAS_HEIGHT);
    context.drawImage(
      video,
      xOffset + (ATLAS_WIDTH - width) / 2 + transform.x * ATLAS_WIDTH,
      yOffset + (ATLAS_HEIGHT - height) / 2 + transform.y * ATLAS_HEIGHT,
      width,
      height,
    );
  }
}

type PieceVertex = {
  x: number;
  y: number;
  phaseX: number;
  phaseY: number;
  speedX: number;
  speedY: number;
};

type WordMask = { width: number; height: number; pixels: Uint8Array };
type Point2D = { x: number; y: number };
type LatticeEdge = {
  start: Point2D;
  end: Point2D;
  textPoints: Point2D[];
  adjacentSlots: number[];
};
type PointAttachment = { point: Point2D; foot: Point2D; edge: LatticeEdge };
type TessellationFace = {
  polygon: Point2D[];
  slots: number[];
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

type GeneratedMask = {
  canvas: HTMLCanvasElement;
  setDensity: (value: number) => void;
  setFluidity: (value: number) => void;
  setCopies: (value: number) => void;
  setPointText: (enabled: boolean) => void;
  setPointTopology: (value: PointTopology) => void;
  renderAt: (timeSeconds: number) => void;
  update: (timeSeconds: number) => void;
  snapshot: () => Uint8ClampedArray;
  readonly pieceCount: number;
};

function seeded(seed: number) {
  const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function gridForDensity(target: number) {
  const aspect = MASK_WIDTH / MASK_HEIGHT;
  let best = { columns: 4, rows: 3, score: Number.POSITIVE_INFINITY };
  for (let columns = 2; columns <= 20; columns++) {
    for (let rows = 2; rows <= 14; rows++) {
      const pieceError = Math.abs(columns * rows - target);
      const aspectError = Math.abs(columns / rows - aspect);
      const score = pieceError * 6 + aspectError;
      if (score < best.score) best = { columns, rows, score };
    }
  }
  return best;
}


const ORIGINAL_VERTICES = [
  [[0, 0], [290, 0], [581, 0], [986, 0], [1280, 0]],
  [[0, 184], [326, 223], [565, 241], [881, 231], [1280, 184]],
  [[0, 427], [307, 522], [576, 444], [982, 538], [1280, 490]],
  [[0, 720], [302, 720], [721, 720], [883, 720], [1280, 720]],
] as const;

function wordMaskFromReference(reference: HTMLImageElement): WordMask {
  const source = document.createElement('canvas');
  source.width = reference.naturalWidth || MASK_WIDTH;
  source.height = reference.naturalHeight || MASK_HEIGHT;
  const sourceContext = source.getContext('2d', { willReadFrequently: true })!;
  sourceContext.drawImage(reference, 0, 0, source.width, source.height);
  const pixels = sourceContext.getImageData(0, 0, source.width, source.height).data;
  // These are the original Arial Black word-image bounds from the asset
  // generator: the visible pixels have a small amount of horizontal padding.
  const minX = 155;
  const minY = 254;
  const width = 970;
  const height = 213;
  const word = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      word[y * width + x] = pixels[((minY + y) * source.width + minX + x) * 4] >= 128 ? 1 : 0;
    }
  }
  return { width, height, pixels: word };
}

// Trace the word's outer and inner contours as straight-segment polylines.
// Point mode uses these samples as vertices in a subdivided lattice mesh.
// Marching squares keeps holes such as the o, p, and e counters instead of
// flattening them into one scan-order point cloud.
function textContoursFromMask(word: WordMask): Point2D[][] {
  type Segment = { a: Point2D; b: Point2D; aKey: string; bKey: string };
  type Node = { point: Point2D; segments: number[] };
  const cases: { [code: number]: Array<[string, string]> } = {
    1: [['top', 'left']],
    2: [['top', 'right']],
    3: [['left', 'right']],
    4: [['right', 'bottom']],
    5: [['top', 'left'], ['right', 'bottom']],
    6: [['top', 'bottom']],
    7: [['left', 'bottom']],
    8: [['left', 'bottom']],
    9: [['top', 'bottom']],
    10: [['top', 'right'], ['left', 'bottom']],
    11: [['right', 'bottom']],
    12: [['left', 'right']],
    13: [['top', 'right']],
    14: [['top', 'left']],
  };
  const edgePoint = (name: string, x: number, y: number): Point2D => {
    if (name === 'top') return { x: x + 0.5, y };
    if (name === 'right') return { x: x + 1, y: y + 0.5 };
    if (name === 'bottom') return { x: x + 0.5, y: y + 1 };
    return { x, y: y + 0.5 };
  };
  const key = (point: Point2D) => `${point.x}:${point.y}`;
  const filled = (x: number, y: number) => x >= 0 && x < word.width && y >= 0 && y < word.height
    && word.pixels[y * word.width + x] === 1;
  const segments: Segment[] = [];
  const nodes = new Map<string, Node>();
  const addSegment = (a: Point2D, b: Point2D) => {
    const segmentIndex = segments.length;
    const segment = { a, b, aKey: key(a), bKey: key(b) };
    segments.push(segment);
    for (const [point, pointKey] of [[a, segment.aKey], [b, segment.bKey]] as const) {
      const node = nodes.get(pointKey);
      if (node) node.segments.push(segmentIndex);
      else nodes.set(pointKey, { point, segments: [segmentIndex] });
    }
  };

  // Pad the mask with empty pixels so contours at the word bounds close too.
  for (let y = -1; y < word.height; y++) {
    for (let x = -1; x < word.width; x++) {
      const code = (filled(x, y) ? 1 : 0)
        | (filled(x + 1, y) ? 2 : 0)
        | (filled(x + 1, y + 1) ? 4 : 0)
        | (filled(x, y + 1) ? 8 : 0);
      for (const [first, second] of cases[code] ?? []) {
        addSegment(edgePoint(first, x, y), edgePoint(second, x, y));
      }
    }
  }

  const unused = new Set(segments.map((_, index) => index));
  const contours: Point2D[][] = [];
  while (unused.size > 0) {
    const firstIndex = unused.values().next().value as number;
    const startKey = segments[firstIndex].aKey;
    let currentKey = startKey;
    let previousSegment = -1;
    const contour: Point2D[] = [];
    while (true) {
      let candidates = (nodes.get(currentKey)?.segments ?? []).filter((index) => unused.has(index));
      if (previousSegment >= 0 && candidates.length > 1) {
        candidates = candidates.filter((index) => index !== previousSegment);
      }
      if (candidates.length === 0) break;
      const segmentIndex = candidates[0];
      unused.delete(segmentIndex);
      const segment = segments[segmentIndex];
      contour.push(nodes.get(currentKey)!.point);
      currentKey = segment.aKey === currentKey ? segment.bKey : segment.aKey;
      previousSegment = segmentIndex;
      if (currentKey === startKey) break;
    }
    if (currentKey !== startKey || contour.length < 3) continue;

    let totalLength = 0;
    for (let index = 0; index < contour.length; index++) {
      const next = contour[(index + 1) % contour.length];
      totalLength += Math.hypot(next.x - contour[index].x, next.y - contour[index].y);
    }
    const sampleCount = Math.max(3, Math.ceil(totalLength / 8));
    const sampled: Point2D[] = [];
    for (let sample = 0; sample < sampleCount; sample++) {
      let target = sample * totalLength / sampleCount;
      for (let index = 0; index < contour.length; index++) {
        const start = contour[index];
        const end = contour[(index + 1) % contour.length];
        const length = Math.hypot(end.x - start.x, end.y - start.y);
        if (target <= length || index === contour.length - 1) {
          const amount = length === 0 ? 0 : target / length;
          sampled.push({
            x: start.x + (end.x - start.x) * amount,
            y: start.y + (end.y - start.y) * amount,
          });
          break;
        }
        target -= length;
      }
    }
    contours.push(sampled);
  }
  return contours;
}

function segmentDistanceSquared(point: Point2D, start: Point2D, end: Point2D) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const rawProjection = lengthSquared === 0
    ? 0
    : ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared;
  const projection = Math.max(0, Math.min(1, rawProjection));
  const x = start.x + projection * dx;
  const y = start.y + projection * dy;
  return { distanceSquared: (point.x - x) ** 2 + (point.y - y) ** 2, projection: rawProjection };
}

// Generate the selector mask itself. The original mask movie is the default
// 4x3 static field; sliders change this same field before the moving lettering
// is applied. Shared four-corner vertices keep neighboring straight edges
// attached while fluidity moves them.
function createGeneratedMask(
  canvas: HTMLCanvasElement,
  word: WordMask,
  initialCopies = 1,
  initialPointText = false,
  initialPointTopology: PointTopology = 'contour-faces',
): GeneratedMask {
  const context = canvas.getContext('2d', { alpha: false })!;
  const baseCanvas = document.createElement('canvas');
  baseCanvas.width = MASK_WIDTH;
  baseCanvas.height = MASK_HEIGHT;
  const baseContext = baseCanvas.getContext('2d', { alpha: false, willReadFrequently: true })!;
  const output = context.createImageData(MASK_WIDTH, MASK_HEIGHT);
  context.imageSmoothingEnabled = false;
  baseContext.imageSmoothingEnabled = false;
  let density = 12;
  let fluidity = 0;
  let copies = Math.max(1, Math.min(MAX_COPIES, Math.round(initialCopies)));
  let columns = 0;
  let rows = 0;
  let pieceColumns = 0;
  let pieceRows = 0;
  let vertices: PieceVertex[] = [];
  let pieceCount = 0;
  let currentTime = 0;
  let pointText = initialPointText;
  let pointTopology = initialPointTopology;
  const textContours = textContoursFromMask(word);
  const textPoints: Point2D[] = [];
  for (const contour of textContours) textPoints.push(...contour);
  const textPath = new Path2D();
  for (const contour of textContours) {
    textPath.moveTo(contour[0].x, contour[0].y);
    for (let index = 1; index < contour.length; index++) textPath.lineTo(contour[index].x, contour[index].y);
    textPath.closePath();
  }

  function rebuild() {
    const grid = gridForDensity(density);
    pieceColumns = grid.columns;
    pieceRows = grid.rows;
    pieceCount = pieceColumns * pieceRows;
    // Point mode needs enough lattice faces for the inserted vertices to
    // remain visible at the original 12-piece setting. The coarse field still
    // supplies the positions, so the default four-point geometry is retained.
    columns = pointText ? Math.max(pieceColumns, 24) : pieceColumns;
    rows = pointText ? Math.max(pieceRows, 16) : pieceRows;
    const coarsePositions: Point2D[] = [];
    for (let coarseRow = 0; coarseRow <= pieceRows; coarseRow++) {
      for (let coarseColumn = 0; coarseColumn <= pieceColumns; coarseColumn++) {
        const original = pieceColumns === 4 && pieceRows === 3
          ? ORIGINAL_VERTICES[coarseRow][coarseColumn]
          : null;
        coarsePositions.push({
          x: original?.[0] ?? coarseColumn * MASK_WIDTH / pieceColumns,
          y: original?.[1] ?? coarseRow * MASK_HEIGHT / pieceRows,
        });
      }
    }
    const coarsePosition = (column: number, row: number) => (
      coarsePositions[row * (pieceColumns + 1) + column]
    );
    vertices = [];
    for (let row = 0; row <= rows; row++) {
      for (let column = 0; column <= columns; column++) {
        const seed = row * 97 + column * 53 + columns * 11 + rows * 7;
        const coarseX = column / columns * pieceColumns;
        const coarseY = row / rows * pieceRows;
        const left = Math.min(pieceColumns - 1, Math.floor(coarseX));
        const top = Math.min(pieceRows - 1, Math.floor(coarseY));
        const xAmount = coarseX - left;
        const yAmount = coarseY - top;
        const topLeft = coarsePosition(left, top);
        const topRight = coarsePosition(left + 1, top);
        const bottomLeft = coarsePosition(left, top + 1);
        const bottomRight = coarsePosition(left + 1, top + 1);
        vertices.push({
          x: topLeft.x * (1 - xAmount) * (1 - yAmount)
            + topRight.x * xAmount * (1 - yAmount)
            + bottomLeft.x * (1 - xAmount) * yAmount
            + bottomRight.x * xAmount * yAmount,
          y: topLeft.y * (1 - xAmount) * (1 - yAmount)
            + topRight.y * xAmount * (1 - yAmount)
            + bottomLeft.y * (1 - xAmount) * yAmount
            + bottomRight.y * xAmount * yAmount,
          phaseX: seeded(seed + 1) * Math.PI * 2,
          phaseY: seeded(seed + 2) * Math.PI * 2,
          speedX: 0.72 + seeded(seed + 3) * 0.56,
          speedY: 0.72 + seeded(seed + 4) * 0.56,
        });
      }
    }
    draw(currentTime);
  }

  function draw(timeSeconds: number) {
    currentTime = timeSeconds;
    const amount = fluidity / 100;
    const cellWidth = MASK_WIDTH / columns;
    const cellHeight = MASK_HEIGHT / rows;
    const amplitude = Math.min(cellWidth, cellHeight) * 0.24 * amount;
    const motion = 0.55 + amount * 0.85;
    const points = vertices.map((vertex, index) => {
      const column = index % (columns + 1);
      const row = Math.floor(index / (columns + 1));
      const boundary = column === 0 || column === columns || row === 0 || row === rows;
      if (boundary || amplitude === 0) return { x: vertex.x, y: vertex.y };
      const phase = timeSeconds * motion;
      const wobbleX = Math.sin(phase * vertex.speedX + vertex.phaseX) * 0.68
        + Math.sin(phase * 0.61 + vertex.phaseY) * 0.32;
      const wobbleY = Math.sin(phase * vertex.speedY + vertex.phaseY) * 0.68
        + Math.cos(phase * 0.57 + vertex.phaseX) * 0.32;
      return { x: vertex.x + wobbleX * amplitude, y: vertex.y + wobbleY * amplitude };
    });
    const phase = timeSeconds / MASK_DURATION * Math.PI * 2;
    const wordX = Math.round((MASK_WIDTH - word.width) / 2 + 125 * Math.sin(phase));
    const wordY = Math.round((MASK_HEIGHT - word.height) / 2 + 120 * Math.sin(phase * 2));
    // v2 demonstration: populate the whole travelling glyph field at once,
    // breathe between 80% and 100%, then dissolve through the same globally
    // ordered pixels.
    const fadeIn = Math.max(0, Math.min(1, timeSeconds / 2.5));
    const fadeOut = Math.max(0, Math.min(1, (MASK_DURATION - timeSeconds) / 2.5));
    const pulse = 0.9 + 0.1 * Math.cos((timeSeconds - 2.5) * Math.PI * 2 / 2.2);
    const textCoverage = Math.min(fadeIn, fadeOut, pulse);

    const horizontalEdges: LatticeEdge[][] = [];
    const verticalEdges: LatticeEdge[][] = [];
    const allEdges: LatticeEdge[] = [];
    const attachments: Array<PointAttachment | null> = [];
    const slotForCell = (row: number, column: number) => {
      // v2 deliberately distributes every source clip across the field.
      // Earlier point modes retain the original two-source checkerboard.
      const source = pointText && pointTopology === 'pure-lattice'
        ? Math.floor(seeded((row + 1) * 193 + (column + 1) * 389 + columns * 17) * 4)
        : (column + row) % 2;
      const copy = (column * 3 + row * 5 + (column ^ row)) % copies;
      return source * copies + copy;
    };
    if (pointText) {
      for (let row = 0; row <= rows; row++) {
        horizontalEdges[row] = [];
        for (let column = 0; column < columns; column++) {
          const edge = {
            start: points[row * (columns + 1) + column],
            end: points[row * (columns + 1) + column + 1],
            textPoints: [],
            adjacentSlots: [
              ...(row > 0 ? [slotForCell(row - 1, column)] : []),
              ...(row < rows ? [slotForCell(row, column)] : []),
            ],
          };
          horizontalEdges[row][column] = edge;
          allEdges.push(edge);
        }
      }
      for (let row = 0; row < rows; row++) {
        verticalEdges[row] = [];
        for (let column = 0; column <= columns; column++) {
          const edge = {
            start: points[row * (columns + 1) + column],
            end: points[(row + 1) * (columns + 1) + column],
            textPoints: [],
            adjacentSlots: [
              ...(column > 0 ? [slotForCell(row, column - 1)] : []),
              ...(column < columns ? [slotForCell(row, column)] : []),
            ],
          };
          verticalEdges[row][column] = edge;
          allEdges.push(edge);
        }
      }
      const maxEdgeDistance = Math.min(22, Math.min(cellWidth, cellHeight) * 0.5);
      for (const localPoint of textPoints) {
        const textPoint = { x: wordX + localPoint.x, y: wordY + localPoint.y };
        let closest: LatticeEdge | null = null;
        let closestDistance = Number.POSITIVE_INFINITY;
        let closestProjection = 0;
        for (const edge of allEdges) {
          const candidate = segmentDistanceSquared(textPoint, edge.start, edge.end);
          if (candidate.projection < 0 || candidate.projection > 1) continue;
          if (candidate.distanceSquared > maxEdgeDistance ** 2) continue;
          if (candidate.distanceSquared < closestDistance) {
            closest = edge;
            closestDistance = candidate.distanceSquared;
            closestProjection = candidate.projection;
          }
        }
        if (!closest) {
          attachments.push(null);
        } else {
          closest.textPoints.push(textPoint);
          const projection = Math.max(0, Math.min(1, closestProjection));
          attachments.push({
            point: textPoint,
            foot: {
              x: closest.start.x + projection * (closest.end.x - closest.start.x),
              y: closest.start.y + projection * (closest.end.y - closest.start.y),
            },
            edge: closest,
          });
        }
      }
      for (const edge of allEdges) {
        const dx = edge.end.x - edge.start.x;
        const dy = edge.end.y - edge.start.y;
        const lengthSquared = dx * dx + dy * dy;
        edge.textPoints.sort((first, second) => {
          const firstProjection = lengthSquared === 0 ? 0
            : ((first.x - edge.start.x) * dx + (first.y - edge.start.y) * dy) / lengthSquared;
          const secondProjection = lengthSquared === 0 ? 0
            : ((second.x - edge.start.x) * dx + (second.y - edge.start.y) * dy) / lengthSquared;
          return firstProjection - secondProjection;
        });
        const maxPoints = Math.max(2, Math.min(10, Math.ceil(Math.sqrt(lengthSquared) / 24)));
        if (edge.textPoints.length > maxPoints) {
          const selected = [];
          for (let index = 0; index < maxPoints; index++) {
            const sourceIndex = maxPoints === 1
              ? 0
              : Math.round(index * (edge.textPoints.length - 1) / (maxPoints - 1));
            selected.push(edge.textPoints[sourceIndex]);
          }
          edge.textPoints = selected;
        }
      }
    }

    const edgePath = (edge: LatticeEdge) => {
      const dx = edge.end.x - edge.start.x;
      const dy = edge.end.y - edge.start.y;
      const length = Math.hypot(dx, dy);
      if (length === 0) return [edge.start];
      const path: Point2D[] = [edge.start];
      for (const textPoint of edge.textPoints) {
        if (pointTopology === 'lattice-tessellation') {
          // This sibling uses the inserted text point as the actual mesh
          // vertex. Avoid the small before/after detours used by the
          // accepted face-bridge mode; those detours turn dense points into
          // a row of distracting teeth.
          path.push(textPoint);
          continue;
        }
        const projection = Math.max(0, Math.min(1,
          ((textPoint.x - edge.start.x) * dx + (textPoint.y - edge.start.y) * dy) / (length * length)));
        const halfSpan = Math.min(12, Math.max(5, length * 0.06));
        const span = halfSpan / length;
        const beforeProjection = Math.max(0, projection - span);
        const afterProjection = Math.min(1, projection + span);
        path.push(
          {
            x: edge.start.x + beforeProjection * dx,
            y: edge.start.y + beforeProjection * dy,
          },
          textPoint,
          {
            x: edge.start.x + afterProjection * dx,
            y: edge.start.y + afterProjection * dy,
          },
        );
      }
      path.push(edge.end);
      return path;
    };
    // This sibling mode turns each contour segment into a real pair of mesh
    // triangles. The base cell is clipped around that face first, so the
    // contour face is part of the rasterized lattice rather than a later
    // highlight painted over it. Its two triangle IDs are selected from the
    // discrete text-region slots, with the nearby lattice copy retained; no
    // second text/ID image is introduced.
    const tessellationFaces: TessellationFace[] = [];
    if (pointText && pointTopology === 'lattice-tessellation') {
      let offset = 0;
      for (const contour of textContours) {
        const contourAttachments = attachments.slice(offset, offset + contour.length);
        offset += contour.length;
        for (let index = 0; index < contourAttachments.length; index++) {
          const first = contourAttachments[index];
          const second = contourAttachments[(index + 1) % contourAttachments.length];
          if (!first || !second) continue;
          if (Math.hypot(second.foot.x - first.foot.x, second.foot.y - first.foot.y) > 80) continue;
          const polygon = [first.foot, second.foot, second.point, first.point];
          const nearbySlot = first.edge.adjacentSlots[0] ?? second.edge.adjacentSlots[0] ?? 0;
          const textCopy = nearbySlot % copies;
          const textSlot = 2 * copies + textCopy;
          const slots = [textSlot, textSlot];
          tessellationFaces.push({
            polygon,
            slots: slots.slice(0, 2),
            minX: Math.min(...polygon.map((point) => point.x)),
            minY: Math.min(...polygon.map((point) => point.y)),
            maxX: Math.max(...polygon.map((point) => point.x)),
            maxY: Math.max(...polygon.map((point) => point.y)),
          });
        }
      }
    }

    const cellPolygon = (row: number, column: number) => {
      const topLeft = points[row * (columns + 1) + column];
      const topRight = points[row * (columns + 1) + column + 1];
      const bottomRight = points[(row + 1) * (columns + 1) + column + 1];
      const bottomLeft = points[(row + 1) * (columns + 1) + column];
      if (!pointText || pointTopology === 'pure-lattice') return [topLeft, topRight, bottomRight, bottomLeft];
      return [
        topLeft,
        ...edgePath(horizontalEdges[row][column]),
        ...edgePath(verticalEdges[row][column + 1]),
        ...edgePath(horizontalEdges[row + 1][column]).reverse(),
        ...edgePath(verticalEdges[row][column]).reverse(),
      ];
    };
    const addPolygonToPath = (polygon: Point2D[]) => {
      baseContext.moveTo(polygon[0].x, polygon[0].y);
      for (let index = 1; index < polygon.length; index++) {
        baseContext.lineTo(polygon[index].x, polygon[index].y);
      }
      baseContext.closePath();
    };
    const valueForSlot = (slot: number) => Math.round(slot * 255 / (4 * copies - 1));

    baseContext.fillStyle = 'rgb(0, 0, 0)';
    baseContext.fillRect(0, 0, MASK_WIDTH, MASK_HEIGHT);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const polygon = cellPolygon(row, column);
        const slot = slotForCell(row, column);
        const value = valueForSlot(slot);
        const useTessellation = tessellationFaces.length > 0;
        if (useTessellation) baseContext.save();
        baseContext.beginPath();
        addPolygonToPath(polygon);
        if (useTessellation) {
          const minX = Math.min(...polygon.map((point) => point.x));
          const minY = Math.min(...polygon.map((point) => point.y));
          const maxX = Math.max(...polygon.map((point) => point.x));
          const maxY = Math.max(...polygon.map((point) => point.y));
          for (const face of tessellationFaces) {
            if (face.maxX < minX || face.minX > maxX || face.maxY < minY || face.minY > maxY) continue;
            addPolygonToPath(face.polygon);
          }
          baseContext.clip('evenodd');
          baseContext.fillStyle = `rgb(${value}, ${value}, ${value})`;
          baseContext.fillRect(0, 0, MASK_WIDTH, MASK_HEIGHT);
          baseContext.restore();
        } else {
          baseContext.fillStyle = `rgb(${value}, ${value}, ${value})`;
          baseContext.fill();
        }
      }
    }

    let preTextPixels: Uint8ClampedArray | null = null;
    if (pointText && pointTopology === 'pure-lattice') {
      // Boolean-split every existing moving face that the word passes over.
      // The word path becomes a shared, straight-segment face boundary inside
      // those quads; there is no patch perimeter, overlay, or independent
      // mesh to leak across the neighbouring lattice.
      // Keep the exact normal field so unrevealed pixels can remain normal
      // during the fade rather than becoming an alpha-blended third value.
      preTextPixels = baseContext.getImageData(0, 0, MASK_WIDTH, MASK_HEIGHT).data;
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const polygon = cellPolygon(row, column);
          const minX = Math.min(...polygon.map((point) => point.x));
          const minY = Math.min(...polygon.map((point) => point.y));
          const maxX = Math.max(...polygon.map((point) => point.x));
          const maxY = Math.max(...polygon.map((point) => point.y));
          if (maxX < wordX || minX > wordX + word.width || maxY < wordY || minY > wordY + word.height) continue;
          const baseSlot = slotForCell(row, column);
          const baseSource = Math.floor(baseSlot / copies);
          const copy = baseSlot % copies;
          // Choose a different, spatially distributed source while retaining
          // the copy. Every contour segment separates two real clip faces.
          const sourceOffset = 1 + Math.floor(seeded((row + 1) * 431 + (column + 1) * 761) * 3);
          const wordSlot = (baseSource + sourceOffset) % 4 * copies + copy;
          baseContext.save();
          baseContext.beginPath();
          addPolygonToPath(polygon);
          baseContext.clip();
          baseContext.translate(wordX, wordY);
          baseContext.clip(textPath, 'evenodd');
          const value = valueForSlot(wordSlot);
          baseContext.fillStyle = `rgb(${value}, ${value}, ${value})`;
          baseContext.fillRect(-wordX, -wordY, MASK_WIDTH, MASK_HEIGHT);
          baseContext.restore();
        }
      }
    }

    // Inserted contour segments are mesh faces between the sampled vertices
    // and their nearest lattice-edge feet. The faces share their contour
    // boundaries with one another; no point marker, connector stroke, or
    // separate text layer is drawn.
    if (pointText && pointTopology === 'contour-faces') {
      const insertedFaceValue = Math.round((2 * copies) * 255 / (4 * copies - 1));
      let offset = 0;
      for (const contour of textContours) {
        const contourAttachments = attachments.slice(offset, offset + contour.length);
        offset += contour.length;
        for (let index = 0; index < contourAttachments.length; index++) {
          const first = contourAttachments[index];
          const second = contourAttachments[(index + 1) % contourAttachments.length];
          if (!first || !second) continue;
          if (Math.hypot(second.foot.x - first.foot.x, second.foot.y - first.foot.y) > 80) continue;
          baseContext.fillStyle = `rgb(${insertedFaceValue}, ${insertedFaceValue}, ${insertedFaceValue})`;
          baseContext.beginPath();
          baseContext.moveTo(first.foot.x, first.foot.y);
          baseContext.lineTo(second.foot.x, second.foot.y);
          baseContext.lineTo(second.point.x, second.point.y);
          baseContext.lineTo(first.point.x, first.point.y);
          baseContext.closePath();
          baseContext.fill();
        }
      }
    }
    if (pointText && pointTopology === 'lattice-tessellation') {
      const fillTriangle = (triangle: Point2D[], slot: number) => {
        const value = valueForSlot(slot);
        baseContext.fillStyle = `rgb(${value}, ${value}, ${value})`;
        baseContext.beginPath();
        baseContext.moveTo(triangle[0].x, triangle[0].y);
        baseContext.lineTo(triangle[1].x, triangle[1].y);
        baseContext.lineTo(triangle[2].x, triangle[2].y);
        baseContext.closePath();
        baseContext.fill();
      };
      for (const face of tessellationFaces) {
        fillTriangle([face.polygon[0], face.polygon[1], face.polygon[2]], face.slots[0]);
        fillTriangle([face.polygon[0], face.polygon[2], face.polygon[3]], face.slots[1]);
      }
    }

    const basePixels = baseContext.getImageData(0, 0, MASK_WIDTH, MASK_HEIGHT).data;
    for (let y = 0; y < MASK_HEIGHT; y++) {
      const wordYPosition = y - wordY;
      for (let x = 0; x < MASK_WIDTH; x++) {
        const offset = (y * MASK_WIDTH + x) * 4;
        const wordXPosition = x - wordX;
        let baseSlot = Math.max(0, Math.min(4 * copies - 1,
          Math.round(basePixels[offset] / 255 * (4 * copies - 1))));
        if (pointText && pointTopology === 'pure-lattice' && preTextPixels) {
          const withinWord = wordXPosition >= 0 && wordXPosition < word.width
            && wordYPosition >= 0 && wordYPosition < word.height
            && word.pixels[wordYPosition * word.width + wordXPosition] === 1;
          // Integer hash makes the reveal spatially distributed, stable at a
          // given pixel, and independent of the clip colors or alpha.
          const hash = (((x * 73856093) ^ (y * 19349663)) >>> 0) / 4294967296;
          if (!withinWord || hash > textCoverage) {
            baseSlot = Math.max(0, Math.min(4 * copies - 1,
              Math.round(preTextPixels[offset] / 255 * (4 * copies - 1))));
          }
        }
        const baseSource = Math.min(1, Math.floor(baseSlot / copies));
        const baseCopy = baseSlot % copies;
        const insideWord = !pointText && wordXPosition >= 0 && wordXPosition < word.width
          && wordYPosition >= 0 && wordYPosition < word.height
          && word.pixels[wordYPosition * word.width + wordXPosition] === 1;
        const slot = insideWord ? (baseSource + 2) * copies + baseCopy : baseSlot;
        const value = Math.round(slot * 255 / (4 * copies - 1));
        output.data[offset] = value;
        output.data[offset + 1] = value;
        output.data[offset + 2] = value;
        output.data[offset + 3] = 255;
      }
    }
    context.putImageData(output, 0, 0);
  }

  rebuild();
  return {
    canvas,
    setDensity(value) {
      const next = Math.max(12, Math.min(96, Math.round(value / 4) * 4));
      if (next !== density) {
        density = next;
        rebuild();
      }
    },
    setFluidity(value) {
      fluidity = Math.max(0, Math.min(100, value));
      if (pointText) draw(currentTime);
    },
    setCopies(value) {
      const next = Math.max(1, Math.min(MAX_COPIES, Math.round(value)));
      if (next !== copies) {
        copies = next;
        rebuild();
      }
    },
    setPointText(enabled) {
      if (enabled !== pointText) {
        pointText = enabled;
        rebuild();
      }
    },
    setPointTopology(value) {
      if (value !== pointTopology) {
        pointTopology = value;
        if (pointText) rebuild();
      }
    },
    renderAt: draw,
    // renderAt() builds one exact point-by-point frame for the offscreen bake.
    // Rebuilding the full raster mask on every animation tick overwhelms the
    // footage compositor, so playback uses the cached frame sequence below.
    update(timeSeconds) {
      if (!pointText) draw(timeSeconds);
    },
    snapshot() { return output.data; },
    get pieceCount() { return pieceCount; },
  };
}

const BAKED_FPS = 30;
const MAX_RUN_LENGTH = (1 << 20) - 1;

type PointMaskConfig = { density: number; fluidity: number; copies: number; topology: PointTopology };
type BakedMaskData = { frames: Uint32Array[]; fps: number; duration: number; key: string };
type BakedMask = { key: string; render: (timeSeconds: number) => boolean };

function encodeMaskFrame(pixels: Uint8ClampedArray) {
  const runs: number[] = [];
  let value = pixels[0];
  let count = 1;
  const addRun = () => runs.push((count << 8) | value);
  for (let pixel = 1; pixel < MASK_WIDTH * MASK_HEIGHT; pixel++) {
    const next = pixels[pixel * 4];
    if (next === value && count < MAX_RUN_LENGTH) {
      count++;
    } else {
      addRun();
      value = next;
      count = 1;
    }
  }
  addRun();
  return Uint32Array.from(runs);
}

function decodeMaskFrame(runs: Uint32Array, pixels: Uint8ClampedArray) {
  let pixel = 0;
  for (const packed of runs) {
    const count = packed >>> 8;
    const value = packed & 255;
    const end = pixel + count;
    for (; pixel < end; pixel++) {
      const offset = pixel * 4;
      pixels[offset] = value;
      pixels[offset + 1] = value;
      pixels[offset + 2] = value;
      pixels[offset + 3] = 255;
    }
  }
}

function createBakedMask(data: BakedMaskData, canvas: HTMLCanvasElement): BakedMask {
  const context = canvas.getContext('2d', { alpha: false })!;
  const image = context.createImageData(MASK_WIDTH, MASK_HEIGHT);
  let lastFrame = -1;
  return {
    key: data.key,
    render(timeSeconds) {
      const loopedTime = ((timeSeconds % data.duration) + data.duration) % data.duration;
      const frame = Math.min(data.frames.length - 1, Math.floor(loopedTime * data.fps));
      if (frame === lastFrame) return false;
      decodeMaskFrame(data.frames[frame], image.data);
      context.putImageData(image, 0, 0);
      lastFrame = frame;
      return true;
    },
  };
}

async function bakePointMask(
  word: WordMask,
  config: PointMaskConfig,
  duration: number,
  key: string,
  cancelled: () => boolean,
  progress: (complete: number, total: number) => void,
): Promise<BakedMaskData | null> {
  const canvas = document.createElement('canvas');
  canvas.width = MASK_WIDTH;
  canvas.height = MASK_HEIGHT;
  const baker = createGeneratedMask(canvas, word, config.copies, true, config.topology);
  baker.setDensity(config.density);
  baker.setFluidity(config.fluidity);
  const total = Math.max(1, Math.round(duration * BAKED_FPS));
  const frames: Uint32Array[] = [];
  for (let frame = 0; frame < total; frame++) {
    if (cancelled()) return null;
    baker.renderAt(frame / BAKED_FPS);
    frames.push(encodeMaskFrame(baker.snapshot()));
    progress(frame + 1, total);
    if (frame % 2 === 1) {
      await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
    }
  }
  return { frames, fps: BAKED_FPS, duration, key };
}

function createWebGLRenderer(canvas: HTMLCanvasElement, footage: HTMLVideoElement[]): Renderer | null {
  const gl = canvas.getContext('webgl', { alpha: false, antialias: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  const shader = (type: number, source: string) => {
    const value = gl.createShader(type)!;
    gl.shaderSource(value, source);
    gl.compileShader(value);
    if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value) ?? 'Shader compilation failed');
    return value;
  };
  const program = gl.createProgram()!;
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec2 position;
    varying vec2 uv;
    void main() { uv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }
  `));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec2 uv;
    uniform sampler2D maskOutput, clip0, clip1, clip2, clip3;
    uniform int view;
    uniform float slotCount, copies, atlasColumns, atlasRows;
    float clipID(vec2 p) { return floor(texture2D(maskOutput, p).r * (slotCount - 1.0) + 0.5); }
    vec2 atlasUV(vec2 p, float copy) {
      float column = mod(copy, atlasColumns);
      float row = floor(copy / atlasColumns);
      // Keep linear filtering inside a copy's tile. Without the half-texel
      // inset, samples at a tile edge pull pixels from the neighboring copy
      // and create false lines in the footage boundaries.
      vec2 inset = vec2(
        0.5 / (atlasColumns * ${ATLAS_WIDTH}.0),
        0.5 / (atlasRows * ${ATLAS_HEIGHT}.0)
      );
      vec2 local = mix(inset, vec2(1.0) - inset, p);
      return vec2((column + local.x) / atlasColumns, (atlasRows - 1.0 - row + local.y) / atlasRows);
    }
    void main() {
      float id = clipID(uv);
      float source = floor(id / copies);
      float copy = id - source * copies;
      vec3 color;
      if (view == 1) color = vec3(id / (slotCount - 1.0));
      else if (view == 2) {
        if (source < 0.5) color = vec3(0.94, 0.40, 0.20);
        else if (source < 1.5) color = vec3(0.15, 0.63, 0.85);
        else if (source < 2.5) color = vec3(0.73, 0.84, 0.31);
        else color = vec3(0.66, 0.43, 0.89);
      } else {
        if (source < 0.5) color = texture2D(clip0, atlasUV(uv, copy)).rgb;
        else if (source < 1.5) color = texture2D(clip1, atlasUV(uv, copy)).rgb;
        else if (source < 2.5) color = texture2D(clip2, atlasUV(uv, copy)).rgb;
        else color = texture2D(clip3, atlasUV(uv, copy)).rgb;
      }
      gl_FragColor = vec4(color, 1.0);
    }
  `));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'Shader linking failed');
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  // The mask is an ID texture, not a color image. Keep browser color and
  // alpha conversion out of its upload path, and never blend the result.
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.disable(gl.BLEND);

  const names = ['maskOutput', 'clip0', 'clip1', 'clip2', 'clip3'];
  const atlasCanvases = footage.map(() => {
    const atlas = document.createElement('canvas');
    atlas.width = ATLAS_WIDTH * ATLAS_COLUMNS;
    atlas.height = ATLAS_HEIGHT * ATLAS_ROWS;
    return atlas;
  });
  const atlasContexts = atlasCanvases.map((atlas) => atlas.getContext('2d')!);
  const textures = Array.from({ length: 5 }, (_, index) => {
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const filter = index === 0 ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.uniform1i(gl.getUniformLocation(program, names[index]), index);
    return texture;
  });
  const lastTimes = footage.map(() => -1);
  let lastCopies = -1;
  footage.forEach((video, index) => video.addEventListener('seeked', () => { lastTimes[index] = -1; }));
  const view = gl.getUniformLocation(program, 'view');
  const slotCount = gl.getUniformLocation(program, 'slotCount');
  const copies = gl.getUniformLocation(program, 'copies');
  const atlasColumns = gl.getUniformLocation(program, 'atlasColumns');
  const atlasRows = gl.getUniformLocation(program, 'atlasRows');
  return {
    draw(settings, maskSource, maskChanged = true) {
      if (maskChanged || (maskSource instanceof HTMLVideoElement && maskSource.readyState >= 2)) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, textures[0]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, maskSource);
      }
      footage.forEach((video, index) => {
        if (video.readyState < 2 || (video.currentTime === lastTimes[index] && settings.copies === lastCopies)) return;
        drawFootageAtlas(video, atlasContexts[index], index, settings.copies);
        const textureIndex = index + 1;
        gl.activeTexture(gl.TEXTURE0 + textureIndex);
        gl.bindTexture(gl.TEXTURE_2D, textures[textureIndex]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, atlasCanvases[index]);
        lastTimes[index] = video.currentTime;
      });
      lastCopies = settings.copies;
      gl.uniform1f(slotCount, 4 * settings.copies);
      gl.uniform1f(copies, settings.copies);
      gl.uniform1f(atlasColumns, ATLAS_COLUMNS);
      gl.uniform1f(atlasRows, ATLAS_ROWS);
      gl.uniform1i(view, settings.view);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    },
  };
}

// A smaller CPU compositor keeps the study usable when WebGL is unavailable.
function createCanvasRenderer(canvas: HTMLCanvasElement, videos: HTMLVideoElement[]): Renderer {
  const width = canvas.width = 640;
  const height = canvas.height = 360;
  const context = canvas.getContext('2d')!;
  const maskLayer = document.createElement('canvas');
  maskLayer.width = width;
  maskLayer.height = height;
  const maskContext = maskLayer.getContext('2d', { willReadFrequently: true })!;
  const atlasCanvases = videos.map(() => {
    const atlas = document.createElement('canvas');
    atlas.width = ATLAS_WIDTH * ATLAS_COLUMNS;
    atlas.height = ATLAS_HEIGHT * ATLAS_ROWS;
    return atlas;
  });
  const atlasContexts = atlasCanvases.map((atlas) => atlas.getContext('2d', { willReadFrequently: true })!);
  const atlasPixelWidth = ATLAS_WIDTH * ATLAS_COLUMNS;
  const atlasPixelHeight = ATLAS_HEIGHT * ATLAS_ROWS;
  const pixels = atlasCanvases.map(() => new Uint8ClampedArray(atlasPixelWidth * atlasPixelHeight * 4));
  let maskPixels = new Uint8ClampedArray(width * height * 4);
  const output = context.createImageData(width, height);
  const ids = new Uint8Array(width * height);
  const lastTimes = videos.map(() => -1);
  let lastCopies = -1;
  videos.forEach((video, index) => video.addEventListener('seeked', () => { lastTimes[index] = -1; }));
  return {
    draw(settings, maskSource, maskChanged = true) {
      if (maskChanged || (maskSource instanceof HTMLVideoElement && maskSource.readyState >= 2)) {
        if (maskSource instanceof HTMLCanvasElement
          && maskSource.width === MASK_WIDTH && maskSource.height === MASK_HEIGHT) {
          // Read the generated mask's own pixels and pick the corresponding
          // source pixel. Drawing it through a second scaled canvas can
          // interpolate the selector before it is decoded into an ID.
          const sourceContext = maskSource.getContext('2d', { willReadFrequently: true });
          if (!sourceContext) throw new Error('Unable to read generated mask');
          const sourcePixels = sourceContext.getImageData(0, 0, MASK_WIDTH, MASK_HEIGHT).data;
          for (let y = 0; y < height; y++) {
            const sourceY = Math.min(MASK_HEIGHT - 1, Math.floor(y * MASK_HEIGHT / height));
            for (let x = 0; x < width; x++) {
              const sourceX = Math.min(MASK_WIDTH - 1, Math.floor(x * MASK_WIDTH / width));
              maskPixels[(y * width + x) * 4] = sourcePixels[(sourceY * MASK_WIDTH + sourceX) * 4];
            }
          }
        } else {
          maskContext.imageSmoothingEnabled = false;
          maskContext.drawImage(maskSource, 0, 0, width, height);
          maskPixels = maskContext.getImageData(0, 0, width, height).data;
        }
      }
      videos.forEach((video, index) => {
        if (video.readyState < 2 || (video.currentTime === lastTimes[index] && settings.copies === lastCopies)) return;
        drawFootageAtlas(video, atlasContexts[index], index, settings.copies);
        pixels[index] = atlasContexts[index].getImageData(0, 0, atlasPixelWidth, atlasPixelHeight).data;
        lastTimes[index] = video.currentTime;
      });
      lastCopies = settings.copies;
      const slotCount = 4 * settings.copies;
      for (let p = 0; p < ids.length; p++) {
        ids[p] = Math.max(0, Math.min(slotCount - 1, Math.round(maskPixels[p * 4] / 255 * (slotCount - 1))));
      }
      for (let p = 0; p < ids.length; p++) {
        const id = ids[p];
        const source = Math.floor(id / settings.copies);
        const copy = id % settings.copies;
        const x = p % width;
        const y = Math.floor(p / width);
        const offset = p * 4;
        const atlasColumn = copy % ATLAS_COLUMNS;
        const atlasRow = Math.floor(copy / ATLAS_COLUMNS);
        const footageOffset = ((atlasRow * ATLAS_HEIGHT + y) * atlasPixelWidth + atlasColumn * ATLAS_WIDTH + x) * 4;
        for (let channel = 0; channel < 3; channel++) {
          const value = settings.view === 1
            ? id / (slotCount - 1) * 255
            : settings.view === 2
              ? COLORS[source][channel] * 255
              : pixels[source][footageOffset + channel];
          output.data[offset + channel] = value;
        }
        output.data[offset + 3] = 255;
      }
      context.putImageData(output, 0, 0);
    },
  };
}

function readyImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Unable to load ${src}`));
    image.src = src;
  });
}

/* ---------------------------------------------------------------------- */
/* Deformed-mesh treatment: the round-4 direction and its controls.         */
/* The selector-mask path above is kept intact behind the treatment switch, */
/* because it is the only thing that renders the round-1 and round-2        */
/* topologies.                                                              */
/* ---------------------------------------------------------------------- */

type MeshPanel = {
  frame: (delta: number) => void;
  setActive: (active: boolean) => void;
};

function createMeshPanel(
  canvas: HTMLCanvasElement,
  footage: HTMLVideoElement[],
  debugCanvas?: HTMLCanvasElement,
  nextCanvas?: HTMLCanvasElement,
  nextDebugCanvas?: HTMLCanvasElement,
  thirdCanvas?: HTMLCanvasElement,
  thirdDebugCanvas?: HTMLCanvasElement,
  fourthCanvas?: HTMLCanvasElement,
  fourthDebugCanvas?: HTMLCanvasElement,
): MeshPanel | null {
  const created = createMeshShowcase(canvas, footage, debugCanvas, true);
  if (!created) return null;
  const firstShowcase: MeshShowcase = created;
  // Created on demand, so the compiler must not narrow them to their `null`.
  let secondShowcase = null as MeshShowcase | null;
  let thirdShowcase = null as MeshShowcase | null;
  let fourthShowcase = null as MeshShowcase | null;
  let showcase = firstShowcase;
  let projectIndex = 0;
  const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
  const c = {
    preset: el<HTMLSelectElement>('m-preset'),
    sequence: el<HTMLSelectElement>('m-sequence'),
    project: el<HTMLOutputElement>('m-project-out'),
    projectCaption: el<HTMLElement>('m-project-caption'),
    text: el<HTMLInputElement>('m-text'),
    font: el<HTMLInputElement>('m-font'),
    fontAuto: el<HTMLInputElement>('m-font-auto'),
    offx: el<HTMLInputElement>('m-offx'),
    offy: el<HTMLInputElement>('m-offy'),
    drift: el<HTMLInputElement>('m-drift'),
    driftAmt: el<HTMLInputElement>('m-drift-amt'),
    cols: el<HTMLInputElement>('m-cols'),
    breath: el<HTMLInputElement>('m-breath'),
    breathSpeed: el<HTMLInputElement>('m-breath-speed'),
    algo: el<HTMLSelectElement>('m-algo'),
    correspondence: el<HTMLSelectElement>('m-correspondence'),
    gain: el<HTMLInputElement>('m-gain'),
    band: el<HTMLInputElement>('m-band'),
    spread: el<HTMLInputElement>('m-spread'),
    ampAuto: el<HTMLInputElement>('m-amp-auto'),
    amp: el<HTMLInputElement>('m-amp'),
    idle: el<HTMLInputElement>('m-idle'),
    settle: el<HTMLInputElement>('m-settle'),
    hold: el<HTMLInputElement>('m-hold'),
    release: el<HTMLInputElement>('m-release'),
    releaseShape: el<HTMLSelectElement>('m-release-shape'),
    play: el<HTMLButtonElement>('m-play'),
    time: el<HTMLInputElement>('m-time'),
    strength: el<HTMLInputElement>('m-strength'),
    schedule: el<HTMLSelectElement>('m-schedule'),
    flarePeak: el<HTMLInputElement>('m-flare-peak'),
    flareCentre: el<HTMLInputElement>('m-flare-centre'),
    flareWidth: el<HTMLInputElement>('m-flare-width'),
    polarity: el<HTMLSelectElement>('m-polarity'),
    smooth: el<HTMLInputElement>('m-smooth'),
    jumbleShading: el<HTMLSelectElement>('m-jumble-shading'),
    luma: el<HTMLInputElement>('m-luma'),
    clamp: el<HTMLInputElement>('m-clamp'),
    footage: el<HTMLSelectElement>('m-footage'),
    commit: el<HTMLInputElement>('m-commit'),
    commitTransition: el<HTMLSelectElement>('m-commit-transition'),
    clothFrontier: el<HTMLSelectElement>('m-cloth-frontier'),
    clothRelief: el<HTMLInputElement>('m-cloth-relief'),
    holdDrop: el<HTMLInputElement>('m-hold-drop'),
    dropBounce: el<HTMLInputElement>('m-drop-bounce'),
    idleField: el<HTMLSelectElement>('m-idle-field'),
    stagger: el<HTMLInputElement>('m-stagger'),
    clip: el<HTMLSelectElement>('m-clip'),
    family: el<HTMLInputElement>('m-family'),
    familyMode: el<HTMLSelectElement>('m-family-mode'),
    territoryRegions: el<HTMLInputElement>('m-territory-regions'),
    territoryGrowth: el<HTMLInputElement>('m-territory-growth'),
    territoryBreath: el<HTMLInputElement>('m-territory-breath'),
    view: el<HTMLSelectElement>('m-view'),
    curve: el<HTMLCanvasElement>('m-curve'),
    status: el<HTMLOutputElement>('m-status'),
    densityNote: el('m-density-note'),
    foldNote: el('m-fold-note'),
    gainLabel: el('m-gain-label'),
  };
  const out = (id: string) => el<HTMLOutputElement>(id);

  let params: MeshParams = { ...MESH_DEFAULTS };
  let clock = 0;
  /** Whole passes of the sequence, so `clock + laps * total` never wraps. */
  let laps = 0;
  let running = true;
  let active = false;
  let solveTimer: number | null = null;
  const n = (input: HTMLInputElement) => Number(input.value);
  const sequenceCount = () => c.sequence.value === 'quad' ? 4 : c.sequence.value === 'trio' ? 3 : c.sequence.value === 'duo' ? 2 : 1;
  const sequenced = () => sequenceCount() > 1;
  const projectNames = ['Yope3D', 'SpinStack', 'β-VAE', 'GlyphInterpreter'];
  const projectWords = ['Yope3D', 'SpinStack', 'β-VAE', 'Glyph\nInterpreter'];
  const projectClips = [0, 1, 2, 4];
  // Prototype 3 solved each name at its own size and stacked the long one,
  // which keeps every project on the same 80 columns.
  const territoryFonts = [230, 199, 230, 170];
  /** The canvas crossfade between projects. Territories and the flow need
   *  none: their release already ends on the next project's opening frame,
   *  so the two displays swap in a single paint. */
  const handoffSeconds = () => params.footage === 'territories' || params.footage === 'cloth' || params.footage === 'clothmesh' || params.footage === 'clothlift' || (params.footage === 'jumble' && params.commit && params.commitTransition === 'flow') ? 0 : 0.55;
  const backgroundWorkers = typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined';
  const smoothUnit = (value: number) => {
    const t = Math.max(0, Math.min(1, value));
    return t * t * (3 - 2 * t);
  };
  let handoffFrom = -1;
  let waitingFor = -1;

  function display(index: number) {
    return index === 3
      ? { canvas: fourthCanvas, debug: fourthDebugCanvas, engine: fourthShowcase }
      : index === 2
      ? { canvas: thirdCanvas, debug: thirdDebugCanvas, engine: thirdShowcase }
      : index === 1
        ? { canvas: nextCanvas, debug: nextDebugCanvas, engine: secondShowcase }
        : { canvas, debug: debugCanvas, engine: firstShowcase };
  }

  function showDisplay(index: number, visible: boolean, opacity = 1, zIndex = 1) {
    const item = display(index);
    item.engine?.setVisible(visible);
    if (item.canvas) {
      item.canvas.hidden = !visible;
      item.canvas.style.opacity = String(opacity);
      item.canvas.style.zIndex = String(zIndex);
    }
    if (item.debug) {
      item.debug.style.opacity = String(opacity);
      item.debug.style.zIndex = String(zIndex + 1);
    }
  }

  /** One project: the chosen clip leads the ring, so it is featured, with the
   *  next project on its right and the previous one on its left. */
  function territoryRingFor(clip: number): number[] {
    const index = projectClips.indexOf(clip);
    const around = (step: number) => projectClips[(index + step + projectClips.length) % projectClips.length];
    return index >= 0 ? projectClips.map((_, step) => around(step)) : [clip, ...projectClips.filter((projectClip) => projectClip !== clip)];
  }

  /** Left, featured and right clips a display opens on. */
  function territoryNeighbours(ring: number[], shown: number): number[] {
    const at = (step: number) => ring[(((shown + step) % ring.length) + ring.length) % ring.length];
    return [at(-1), at(0), at(1)];
  }

  function projectParams(base: MeshParams, index: number): MeshParams {
    if (!sequenced()) return base;
    const project = {
      ...base,
      text: projectWords[index],
      clip: projectClips[index],
      homeClip: projectClips[index],
      clothUnder: projectClips[(index + 1) % sequenceCount()],
      columns: base.columns,
      // The strip of regions repeats the sequence. Each handover moves every
      // region one place left along it, so the next project's regions are
      // always the ones right of the featured project's.
      territoryRing: projectClips.slice(0, sequenceCount()),
      territoryHandover: true,
    };
    return base.footage === 'territories'
      ? { ...project, fontSize: territoryFonts[index], fontAuto: false }
      : project;
  }

  function ensureProjects(forceThrough = -1) {
    if (sequenceCount() >= 2 && !secondShowcase && nextCanvas) {
      secondShowcase = createMeshShowcase(nextCanvas, footage, nextDebugCanvas, true);
      secondShowcase?.setVisible(false);
      secondShowcase?.setParams(projectParams(read(), 1));
    }
    // With workers both incoming words can prepare from the start. On older
    // browsers, let each incoming word finish before starting the next one.
    if (sequenceCount() >= 3 && !thirdShowcase && thirdCanvas && (forceThrough >= 2 || backgroundWorkers || (projectIndex > 0 && handoffFrom < 0 && secondShowcase && !secondShowcase.status().solving))) {
      thirdShowcase = createMeshShowcase(thirdCanvas, footage, thirdDebugCanvas, true);
      thirdShowcase?.setVisible(false);
      thirdShowcase?.setParams(projectParams(read(), 2));
    }
    if (sequenceCount() >= 4 && !fourthShowcase && fourthCanvas && (forceThrough >= 3 || backgroundWorkers || (projectIndex > 1 && handoffFrom < 0 && thirdShowcase && !thirdShowcase.status().solving))) {
      fourthShowcase = createMeshShowcase(fourthCanvas, footage, fourthDebugCanvas, true);
      fourthShowcase?.setVisible(false);
      fourthShowcase?.setParams(projectParams(read(), 3));
    }
  }

  function syncProject(index: number, keepPrevious = -1) {
    if (index === 1 && !secondShowcase) return;
    if (index === 2 && !thirdShowcase) return;
    if (index === 3 && !fourthShowcase) return;
    projectIndex = index;
    showcase = display(index).engine!;
    for (let project = 0; project < projectNames.length; project++) showDisplay(project, active && (index === project || keepPrevious === project));
    if (sequenced()) {
      c.text.value = projectNames[index];
      c.clip.value = String(projectClips[index]);
    }
    c.project.textContent = sequenced() ? projectNames[index] : c.text.value;
    caption();
  }

  function caption() {
    const treatment = c.footage.value === 'territories' ? 'breathing territories' : c.footage.value === 'cloth' ? 'cloth wipe' : 'deformed mesh';
    c.projectCaption.textContent = `${c.project.textContent} / ${treatment}`;
  }

  function selectPreset(name: MeshPresetName) {
    if (name === 'custom') return;
    const preset = MESH_PRESETS[name];
    const textValues: Record<string, unknown> = {
      ...preset,
      correspondence: preset.warpCorrespondence ?? 'offset',
    };
    const controlByKey: Record<string, HTMLInputElement | HTMLSelectElement> = {
      text: c.text, fontSize: c.font, offsetX: c.offx, offsetY: c.offy, driftAmount: c.driftAmt,
      columns: c.cols, breath: c.breath, breathSpeed: c.breathSpeed, algorithm: c.algo,
      warpCorrespondence: c.correspondence, gain: c.gain, band: c.band, spread: c.spread,
      amplitude: c.amp, idle: c.idle, settle: c.settle, hold: c.hold, release: c.release,
      releaseShape: c.releaseShape, strength: c.strength, schedule: c.schedule,
      flarePeak: c.flarePeak, flareCentre: c.flareCentre, flareWidth: c.flareWidth,
      polarity: c.polarity, smoothingPx: c.smooth, footage: c.footage, clip: c.clip,
      idleField: c.idleField, stagger: c.stagger, jumbleShading: c.jumbleShading, view: c.view,
      familyMode: c.familyMode, commitTransition: c.commitTransition, clothFrontier: c.clothFrontier, clothRelief: c.clothRelief, dropBounce: c.dropBounce,
      territoryGrowth: c.territoryGrowth, territoryBreath: c.territoryBreath, territoryRegions: c.territoryRegions,
    };
    for (const [key, value] of Object.entries(textValues)) {
      const control = controlByKey[key];
      if (!control || (typeof value !== 'string' && typeof value !== 'number')) continue;
      control.value = String(value);
    }
    c.fontAuto.checked = Boolean(preset.fontAuto ?? false);
    c.drift.checked = Boolean(preset.drift ?? false);
    c.holdDrop.checked = Boolean(preset.holdDrop ?? false);
    c.ampAuto.checked = Boolean(preset.amplitudeAuto ?? true);
    c.luma.checked = Boolean(preset.lumaAdaptive ?? false);
    c.clamp.checked = Boolean(preset.clampIdleTail ?? false);
    c.commit.checked = Boolean(preset.commit ?? false);
    c.family.checked = Boolean(preset.familySplit ?? false);
    c.preset.value = name;
    if (sequenced()) {
      c.text.value = projectNames[projectIndex];
      c.clip.value = String(projectClips[projectIndex]);
    }
    apply(false);
    markChanged();
  }

  // The panel is the only layer that knows what the clips are CALLED, so the
  // word-to-ground match lives here and the engine receives a plain index.
  const clipNames = footage.map((video, index) => video.dataset.name ?? `clip ${index}`);
  /** Every string that should ground on a clip: its name plus its aliases. */
  const clipKeys = footage.map((video, index) => [clipNames[index], ...(video.dataset.aliases ?? '').split('|')].filter(Boolean));

  // Greek survives the glyph pipeline intact — Arial Black covers it — but it
  // does not survive an [a-z0-9] strip, and a word like "βVAE" would reduce to
  // "vae" and match whatever else happens to contain those letters. Folding
  // each letter to its NAME first is what makes "βVAE" and "beta-VAE" the same
  // string, which is the whole point of matching from the word.
  const GREEK: Record<string, string> = {
    α: 'alpha', β: 'beta', γ: 'gamma', δ: 'delta', ε: 'epsilon', ζ: 'zeta', η: 'eta', θ: 'theta',
    ι: 'iota', κ: 'kappa', λ: 'lambda', μ: 'mu', ν: 'nu', ξ: 'xi', ο: 'omicron', π: 'pi',
    ρ: 'rho', σ: 'sigma', ς: 'sigma', τ: 'tau', υ: 'upsilon', φ: 'phi', χ: 'chi', ψ: 'psi', ω: 'omega',
  };
  const normalise = (value: string) =>
    Array.from(value.toLowerCase())
      .map((char) => GREEK[char] ?? char)
      .join('')
      .replace(/[^a-z0-9]/g, '');

  /** The clip whose name the word names. Longest match wins, so a word holding
   *  two names picks the more specific one. No match keeps the Clip select's
   *  choice rather than silently grounding on clip 0. */
  function matchHome(text: string, fallback: number): number {
    const word = normalise(text);
    if (!word) return fallback;
    let best = -1;
    let bestLength = 0;
    clipKeys.forEach((names, index) => {
      for (const name of names) {
        const key = normalise(name);
        if (!key) continue;
        // The word carries the name ("Yope3D", "Yope3D reel"), or the word is a
        // prefix of it ("Spin" for "SpinStack"). Score by how much of the name
        // was actually matched, so a longer, more specific name wins.
        const length = word.includes(key) ? key.length : word.length >= 3 && key.startsWith(word) ? word.length : 0;
        if (length > bestLength) {
          best = index;
          bestLength = length;
        }
      }
    });
    return best >= 0 ? best : fallback;
  }

  function homeClipFor(text: string, mode: FamilyMode, clip: number): number {
    return mode === 'origin-auto' ? matchHome(text, clip) : clip;
  }

  function read(): MeshParams {
    const clip = Number(c.clip.value);
    const familyMode = c.familyMode.value as FamilyMode;
    return {
      text: c.text.value,
      fontSize: n(c.font),
      fontAuto: c.fontAuto.checked,
      offsetX: n(c.offx),
      offsetY: n(c.offy),
      drift: c.drift.checked,
      driftAmount: n(c.driftAmt),
      columns: n(c.cols),
      breath: n(c.breath),
      breathSpeed: n(c.breathSpeed),
      algorithm: c.algo.value as Algorithm,
      warpCorrespondence: c.correspondence.value as WarpCorrespondence,
      gain: n(c.gain),
      band: n(c.band),
      spread: n(c.spread),
      amplitudeAuto: c.ampAuto.checked,
      amplitude: n(c.amp),
      idle: n(c.idle),
      settle: n(c.settle),
      hold: n(c.hold),
      release: n(c.release),
      releaseShape: c.releaseShape.value as ReleaseShape,
      strength: n(c.strength),
      schedule: c.schedule.value as Schedule,
      flarePeak: n(c.flarePeak),
      flareCentre: n(c.flareCentre),
      flareWidth: n(c.flareWidth),
      polarity: c.polarity.value as Polarity,
      lumaAdaptive: c.luma.checked,
      smoothing: n(c.smooth),
      smoothingPx: n(c.smooth),
      clampIdleTail: c.clamp.checked,
      jumbleShading: c.jumbleShading.value as JumbleShading,
      footage: c.footage.value as FootageMode,
      commit: c.commit.checked,
      commitTransition: c.commitTransition.value as CommitTransition,
      clothFrontier: c.clothFrontier.value as ClothFrontier,
      clothRelief: n(c.clothRelief),
      // The cloth wipe stacks the clips in sequence order: this project's
      // clip lies under the one before it, which the fall drags away.
      clothUnder: territoryRingFor(clip)[1],
      holdDrop: c.holdDrop.checked,
      dropBounce: n(c.dropBounce),
      idleField: c.idleField.value as IdleField,
      stagger: n(c.stagger),
      clip,
      familySplit: c.family.checked,
      familyMode,
      homeClip: homeClipFor(c.text.value, familyMode, clip),
      seed: MESH_DEFAULTS.seed,
      territoryRing: territoryRingFor(clip),
      territoryRegions: n(c.territoryRegions),
      territoryGrowth: n(c.territoryGrowth),
      territoryBreath: n(c.territoryBreath),
      // a lone project has no one to hand over to; the sequence turns it on
      territoryHandover: false,
      view: Number(c.view.value),
    };
  }

  /** Re-solving is the only expensive thing here, so it is debounced and the
   *  previous mesh keeps drawing until the replacement lands. */
  function apply(solveChanged: boolean) {
    const next = projectParams(read(), projectIndex);
    const sendParams = (value: MeshParams) => {
      firstShowcase.setParams(projectParams(value, 0));
      if (sequenced()) {
        ensureProjects();
        if (sequenceCount() >= 2) secondShowcase?.setParams(projectParams(value, 1));
        if (sequenceCount() >= 3) thirdShowcase?.setParams(projectParams(value, 2));
        if (sequenceCount() >= 4) fourthShowcase?.setParams(projectParams(value, 3));
      }
    };
    if (solveChanged) {
      next.text = params.text;
      next.fontSize = params.fontSize;
      next.columns = params.columns;
      next.algorithm = params.algorithm;
      next.gain = params.gain;
      next.band = params.band;
      if (solveTimer !== null) window.clearTimeout(solveTimer);
      solveTimer = window.setTimeout(() => {
        solveTimer = null;
        params = projectParams(read(), projectIndex);
        sendParams(params);
      }, 280);
    }
    params = next;
    sendParams(params);
    labels();
  }

  function labels() {
    const p = projectParams(read(), projectIndex);
    const status = showcase.status();
    const territories = p.footage === 'territories';
    // A territories sequence sets each project's word at the size prototype 3
    // solved it at, so the size controls have nothing left to steer.
    const fixedWords = territories && sequenced();
    out('m-text-out').textContent = `${p.text.replace('\n', '').length} chars${p.text.includes('\n') ? ' · 2 lines' : ''}`;
    // In auto the slider is inert, so the readout has to show the size that was
    // actually rasterised, not the one the slider still points at.
    out('m-font-out').textContent = p.fontAuto
      ? `auto ${status.fontPx > 0 ? Math.round(status.fontPx) : '…'} px`
      : `${p.fontSize} px${fixedWords ? ' · per project' : ''}`;
    c.font.disabled = p.fontAuto || fixedWords;
    out('m-offx-out').textContent = `${p.offsetX} px`;
    out('m-offy-out').textContent = `${p.offsetY} px`;
    out('m-drift-out').textContent = `±${p.driftAmount} px`;
    const cell = 1280 / p.columns;
    const rows = Math.round(720 / cell);
    out('m-cols-out').textContent = `${p.columns}×${rows} · ${cell.toFixed(1)} px`;
    out('m-breath-out').textContent = `${p.breath.toFixed(2)} cells`;
    out('m-breath-out2').textContent = `${p.breathSpeed.toFixed(2)}×`;
    out('m-algo-out').textContent = p.algorithm === 'settle' ? 'F settle' : 'G harmonic';
    c.gainLabel.textContent = p.algorithm === 'settle' ? 'Gain (attraction)' : 'Gain (offset/stroke)';
    out('m-correspondence-out').textContent = p.algorithm === 'warp'
      ? (p.warpCorrespondence === 'late-g-lens' ? 'late G · scale .82 · rot 25°' : 'offset contour · legacy')
      : 'unused by F';
    out('m-gain-out').textContent = p.algorithm === 'settle'
      ? `${p.gain.toFixed(2)} · k ${(6 * p.gain).toFixed(1)}`
      : `${p.gain.toFixed(2)} · stroke`;
    out('m-band-out').textContent = `${p.band.toFixed(2)} cells${p.band > 1.2 ? ' · shreds' : ''}`;
    // Spread is a browser-only window on top of G's field. Off is the probe.
    out('m-spread-out').textContent = p.spread > 0 ? `${p.spread.toFixed(2)} strokes` : 'off · probe field';
    // Manual amplitude ignores the cycle, which is indistinguishable from a
    // frozen cycle unless the panel says so outright.
    out('m-amp-out').textContent = p.amplitudeAuto
      ? `auto ${status.amplitude.toFixed(2)}`
      : `MANUAL ${p.amplitude.toFixed(2)} · cycle ignored`;
    el('m-amp-row').classList.toggle('manual', !p.amplitudeAuto);
    c.amp.disabled = p.amplitudeAuto;
    out('m-idle-out').textContent = `${p.idle.toFixed(1)} s`;
    out('m-settle-out').textContent = `${p.settle.toFixed(1)} s`;
    out('m-hold-out').textContent = `${p.hold.toFixed(1)} s`;
    out('m-release-out').textContent = `${p.release.toFixed(1)} s`;
    const cycle = cycleOf(p);
    const count = sequenceCount();
    const total = cycle.loop * count;
    out('m-loop-out').textContent = `${total.toFixed(1)} s loop${sequenced() ? ` · ${count} projects` : ''}`;
    c.time.max = total.toFixed(2);
    c.text.disabled = sequenced();
    c.clip.disabled = sequenced();
    c.cols.disabled = false;
    c.fontAuto.disabled = fixedWords;
    out('m-strength-out').textContent = p.strength.toFixed(2) + (p.strength > 1.1 ? ' · sticker' : p.strength < 0.35 ? ' · faint' : '');
    out('m-sched-out').textContent = `s ${status.strength.toFixed(2)}`;
    out('m-flare-peak-out').textContent = p.flarePeak.toFixed(2);
    out('m-flare-centre-out').textContent = `a ${p.flareCentre.toFixed(2)}`;
    out('m-flare-width-out').textContent = p.flareWidth.toFixed(2);
    const exponentSign = ['compression-bright', 'positive', 'embossed'].includes(p.polarity) ? '+' : '-';
    out('m-polarity-out').textContent = `exponent ${exponentSign}${p.strength.toFixed(2)}`;
    const smoothPx = p.smoothingPx ?? p.smoothing;
    out('m-smooth-out').textContent = `${smoothPx.toFixed(1)} px screen target`;
    out('m-jumble-shading-out').textContent = p.jumbleShading === 'on' ? 'stacks' : 'clips only';
    out('m-footage-out').textContent = p.footage === 'jumble'
      ? 'fixed mix'
      : territories
        ? `${sequenceCount()} regions`
        : p.footage === 'cloth' || p.footage === 'clothmesh' || p.footage === 'clothlift'
          ? `${clipNames[p.homeClip] ?? p.homeClip} → ${clipNames[p.clothUnder] ?? p.clothUnder}`
          : 'one clip';
    // Territories never commit: each region keeps one continuous clip.
    const committing = p.commit && !territories;
    c.commit.disabled = territories;
    c.idleField.disabled = !committing;
    c.commitTransition.disabled = !committing;
    out('m-commit-transition-out').textContent = territories
      ? 'unused by territories'
      : !p.commit
        ? 'needs commit'
        : p.commitTransition === 'flow'
          ? (p.footage === 'jumble' ? 'collage squeezed into the word' : 'needs the jumble')
          : p.commitTransition === 'push' ? 'far first, finishes at full strength' : p.commitTransition === 'squeeze' ? 'contracts into the new clip' : 'instant';
    const flowing = committing && p.footage === 'jumble' && p.commitTransition === 'flow';
    c.clothFrontier.disabled = !flowing;
    out('m-cloth-frontier-out').textContent = !flowing
      ? 'needs flow'
      : !hasClothBake(p.text)
        ? 'no bake for this word'
        : p.clothFrontier === 'off' ? 'plain distance' : p.clothFrontier === 'height' ? 'sag bends it' : p.clothFrontier === 'frontier' ? 'edge rides the fabric' : 'edge and collage ride it';
    // The cloth wipe is the bake on its own: one clip, no field, no glyph.
    const wiping = p.footage === 'cloth' || p.footage === 'clothmesh' || p.footage === 'clothlift';
    c.clothRelief.disabled = !wiping;
    out('m-cloth-relief-out').textContent = !wiping
      ? 'needs the cloth wipe'
      : !hasClothBake(p.text)
        ? 'no bake for this word'
        : p.clothRelief > 0.02 ? `relief ${(p.clothRelief * 100).toFixed(0)}%` : 'squeeze only';
    c.holdDrop.disabled = !flowing;
    c.dropBounce.disabled = !flowing || !p.holdDrop;
    out('m-drop-bounce-out').textContent = !flowing ? 'needs flow' : !p.holdDrop ? 'word stays put' : p.dropBounce < 0.05 ? 'lands dead' : `keeps ${(p.dropBounce * 100).toFixed(0)}% per bounce`;
    c.stagger.disabled = !committing;
    c.family.disabled = !committing;
    c.familyMode.disabled = !committing;
    c.territoryRegions.disabled = !territories;
    c.territoryGrowth.disabled = !territories;
    c.territoryBreath.disabled = !territories;
    // The featured project holds 40% of the width at rest and grows by twice
    // this on each side of every region it holds.
    const heldShare = Math.min(1, 0.4 + 2 * p.territoryGrowth);
    out('m-territory-growth-out').textContent = `${p.territoryGrowth.toFixed(2)} · 40→${Math.round(heldShare * 100)}%${p.territoryGrowth >= 0.3 ? ' · bands shut' : ''}`;
    out('m-territory-breath-out').textContent = p.territoryBreath > 0 ? `±${Math.round(40 * p.territoryBreath)} px` : 'still';
    const perProject = Math.round(p.territoryRegions);
    out('m-territory-regions-out').textContent = perProject === 1 ? '1 · prototype 3' : `${perProject} · bands interleave`;
    const [left, featured, right] = territoryNeighbours(p.territoryRing, sequenced() ? projectIndex : 0)
      .map((clip) => clipNames[clip] ?? `clip ${clip}`);
    out('m-territory-roles-out').textContent = territories ? `${left} | ${featured} | ${right}` : 'needs territories';
    caption();
    out('m-idle-field-out').textContent = p.idleField === 'jumble' ? 'collage' : 'one clip';
    out('m-stagger-out').textContent = p.stagger.toFixed(2);
    // In origin mode the Clip select stops meaning "the single clip" and starts
    // meaning "which project owns the background", so the readout has to say so.
    const selectedName = c.clip.selectedOptions[0]?.textContent ?? '';
    const groundName = clipNames[p.homeClip] ?? selectedName;
    const originActive = p.commit && p.familySplit && p.familyMode !== 'luminance';
    const matched = p.familyMode === 'origin-auto' && p.homeClip !== p.clip;
    out('m-family-mode-out').textContent = territories
      ? 'unused by territories'
      : !p.commit
        ? 'needs commit'
        : p.familyMode === 'luminance'
          ? 'dark out / bright in'
          : p.familyMode === 'origin-auto'
            ? `ground → ${groundName}${matched ? ' · from the word' : ' · no match, using Clip'}`
            : `ground → ${groundName}`;
    out('m-clip-out').textContent = territories
      ? `${selectedName} · featured`
      : originActive && p.familyMode === 'origin'
        ? `${selectedName} · home`
        : originActive && !matched
          ? `${selectedName} · home (fallback)`
          : selectedName;
    out('m-view-out').textContent = c.view.selectedOptions[0]?.textContent ?? '';

    // cells per stroke, live, against the floor for the ACTIVE mode
    const perStroke = status.strokePx > 0 ? status.strokePx / cell : 0;
    const hybridActive = p.footage === 'jumble' && p.commit && p.familySplit;
    const floor = hybridActive ? 1.14 : 2.5;
    const sweet = hybridActive ? '48 to 64 columns (family hybrid)' : '72 columns and up';
    c.densityNote.textContent = perStroke > 0
      ? `${perStroke.toFixed(2)} cells per stroke · stroke ${status.strokePx.toFixed(0)} px · floor ${floor} ${hybridActive ? '(hybrid contrast active)' : '(geometry)'}, comfortable at ${sweet}`
      : 'measuring the stroke…';
    c.densityNote.classList.toggle('warn', perStroke > 0 && perStroke < floor);
    c.foldNote.textContent = `${status.constraints} constraints · invalid cells ${(status.invalidRate * 100).toFixed(1)}% · BD diagonal ${(status.diagonalBdRate * 100).toFixed(1)}% · ${status.vertices.toLocaleString()} vertices · solve ${status.solveMs.toFixed(0)} ms`;
    c.foldNote.classList.toggle('warn', status.foldRate > 0.09);
  }

  function statusLine() {
    if (waitingFor >= 0) {
      const pending = display(waitingFor).engine?.status();
      c.status.textContent = pending
        ? `${pending.solving ? `Preparing ${projectNames[waitingFor]} ${(pending.progress * 100).toFixed(0)}%` : `Loading ${projectNames[waitingFor]} footage`} · keeping the collage visible`
        : `Preparing ${projectNames[waitingFor]} · keeping the collage visible`;
      return;
    }
    const s = showcase.status();
    const solving = s.solving ? `solving ${(s.progress * 100).toFixed(0)}% · ` : '';
    const featured = s.territoryShare > 0 ? `featured ${(s.territoryShare * 100).toFixed(0)}% · ` : '';
    c.status.textContent = `${solving}a ${s.amplitude.toFixed(2)} · s ${s.strength.toFixed(2)} · ${featured}t ${s.time.toFixed(1)}/${s.loop.toFixed(1)}s · ${s.frameMs.toFixed(1)} ms · ${s.note}`;
  }

  function drawCurve() {
    const ctx = c.curve.getContext('2d');
    if (!ctx) return;
    const w = c.curve.width;
    const h = c.curve.height;
    const cycle = cycleOf(params);
    const count = sequenceCount();
    const total = cycle.loop * count;
    const top = 10;
    const bottom = h - 16;
    const yOf = (v: number) => bottom - (Math.min(v, 2) / 2) * (bottom - top);
    const xOf = (t: number) => (t / total) * w;
    ctx.fillStyle = '#12151a';
    ctx.fillRect(0, 0, w, h);
    // the 0.35 to 0.75 strength window
    ctx.fillStyle = 'rgba(111, 159, 106, 0.16)';
    ctx.fillRect(0, yOf(0.75), w, yOf(0.35) - yOf(0.75));
    // phase boundaries
    ctx.strokeStyle = '#2b3037';
    ctx.lineWidth = 1;
    for (let project = 0; project < count; project++) {
      [cycle.t0, cycle.t1, cycle.t2, cycle.loop].forEach((t) => {
        ctx.beginPath();
        ctx.moveTo(Math.round(xOf(project * cycle.loop + t)) + 0.5, top);
        ctx.lineTo(Math.round(xOf(project * cycle.loop + t)) + 0.5, bottom);
        ctx.stroke();
      });
    }
    const plot = (fn: (t: number) => number, color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let project = 0; project < count; project++) {
        for (let i = 0; i <= 400; i++) {
          const t = (i / 400) * cycle.loop;
          const x = xOf(project * cycle.loop + t);
          const y = yOf(fn(t));
          if (i === 0 && project === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    };
    plot((t) => amplitudeAt(t, cycle, params.releaseShape), '#5f9bd8');
    plot((t) => strengthAt(t, cycle, params), '#d1603d');
    // playhead
    const x = Math.round(xOf(((clock % total) + total) % total)) + 0.5;
    ctx.strokeStyle = '#e2df8f';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, top - 6);
    ctx.lineTo(x, bottom + 6);
    ctx.stroke();
    ctx.fillStyle = '#7f8894';
    ctx.font = '16px ui-monospace, monospace';
    ctx.fillText('a(t)', 8, h - 2);
    ctx.fillStyle = '#d1603d';
    ctx.fillText('s(t)', 56, h - 2);
    if (sequenced()) {
      ctx.fillStyle = '#aeb7c2';
      ctx.font = '12px ui-monospace, monospace';
      for (let project = 0; project < count; project++) {
        ctx.fillText(projectNames[project], (project / count) * w + 8, top + 12);
      }
    }
  }

  const solveControls: HTMLElement[] = [c.text, c.font, c.fontAuto, c.cols, c.algo, c.correspondence, c.gain, c.band];
  const everyControl: HTMLElement[] = [
    c.sequence, c.text, c.font, c.fontAuto, c.offx, c.offy, c.drift, c.driftAmt, c.cols, c.breath, c.breathSpeed,
    c.algo, c.correspondence, c.gain, c.band, c.spread, c.ampAuto, c.amp, c.idle, c.settle, c.hold, c.release,
    c.releaseShape, c.strength, c.schedule, c.flarePeak, c.flareCentre, c.flareWidth,
    c.polarity, c.smooth, c.jumbleShading, c.luma, c.clamp, c.footage, c.idleField, c.familyMode,
    c.stagger, c.clip, c.family, c.commit, c.commitTransition, c.clothFrontier, c.clothRelief, c.holdDrop, c.dropBounce, c.territoryRegions, c.territoryGrowth, c.territoryBreath, c.view,
  ];
  everyControl.forEach((control) => {
    if (control === c.sequence) return;
    const solveChanged = solveControls.indexOf(control) >= 0;
    control.addEventListener('input', () => { c.preset.value = 'custom'; apply(solveChanged); markChanged(); });
    control.addEventListener('change', () => { c.preset.value = 'custom'; apply(solveChanged); markChanged(); });
  });

  /* Default tracking. A control that has drifted from its default looks
     identical to one that has not, which makes an ordinary changed setting
     read as a broken effect. Every row that differs is marked, each panel
     carries a count, and one button puts the whole panel back. Defaults are
     captured here, before any preset or query string overrides them. */
  const valueOf = (control: HTMLElement) =>
    control instanceof HTMLInputElement && control.type === 'checkbox'
      ? String(control.checked)
      : (control as HTMLInputElement | HTMLSelectElement).value;
  const setValue = (control: HTMLElement, value: string) => {
    if (control instanceof HTMLInputElement && control.type === 'checkbox') control.checked = value === 'true';
    else (control as HTMLInputElement | HTMLSelectElement).value = value;
  };
  const defaultOf = new Map<HTMLElement, string>(everyControl.map((control) => [control, valueOf(control)]));
  const panel = el<HTMLElement>('mesh-controls');
  const changedOut = el<HTMLOutputElement>('m-changed-out');

  function markChanged() {
    const perPanel = new Map<HTMLElement, number>();
    let total = 0;
    everyControl.forEach((control) => {
      const drifted = valueOf(control) !== defaultOf.get(control);
      control.closest('.row')?.classList.toggle('changed', drifted);
      if (!drifted) return;
      total += 1;
      const group = control.closest('details');
      if (group) perPanel.set(group, (perPanel.get(group) ?? 0) + 1);
    });
    panel.querySelectorAll('details').forEach((group) => {
      const count = perPanel.get(group) ?? 0;
      const badge = group.querySelector('.badge');
      if (badge) badge.textContent = count > 0 ? String(count) : '';
    });
    changedOut.textContent = total > 0 ? `${total} changed from default` : 'all defaults';
    changedOut.classList.toggle('warn', total > 0);
  }

  // The four-project territories treatment is this page's starting point.
  // Capture it as the reset target after the controls have been populated.
  selectPreset(DEFAULT_PRESET);
  everyControl.forEach((control) => defaultOf.set(control, valueOf(control)));
  markChanged();

  el<HTMLButtonElement>('m-reset').addEventListener('click', () => {
    everyControl.forEach((control) => setValue(control, defaultOf.get(control) ?? valueOf(control)));
    c.preset.value = DEFAULT_PRESET;
    clock = 0;
    laps = 0;
    handoffFrom = -1;
    waitingFor = -1;
    syncProject(0);
    apply(true);
    markChanged();
  });

  /* The two algorithms were tuned and measured separately, so F's settings are
     not a fair starting point for G and vice versa. Switching restores the
     configuration each one was actually judged at. Word, cycle, footage and
     view are deliberately left alone: they are orthogonal to the algorithm and
     resetting them would throw away the comparison you were setting up.
     G's figures come from probe-g-global-warp.py, whose round-4 stages all run
     at 112x63 with WOB = 0.35, and whose image-warp peak is the late-G lens
     rather than the offset contour. F's are the page defaults. */
  const algoDefaults: Record<string, Partial<Record<keyof typeof c, string>>> = {
    settle: {
      gain: '1', band: '0.85', spread: '1', cols: '88', breath: '0.3',
      correspondence: 'offset', polarity: 'compression-dark',
    },
    warp: {
      // The offset contour at 0.30 stroke widths is G's WIREFRAME optimum and
      // is what probe-g-global-warp.py's acceptance-test stage renders. The
      // lens is the IMAGE-WARP peak and deliberately trades the crease away,
      // so it reads on footage and not on the wireframe. Default to the one
      // that passes the acceptance test; switch correspondence to compare.
      gain: '0.3', band: '0.85', spread: '0', cols: '112', breath: '0.35',
      correspondence: 'offset', polarity: 'compression-bright',
    },
  };
  c.correspondence.addEventListener('change', () => {
    // In offset mode gain IS the offset as a fraction of stroke width, where
    // 0.30 is the probe optimum. In lens mode it is a bare displacement
    // multiplier that the solver divides by 0.8, so 0.8 is neutral.
    if (c.algo.value !== 'warp') return;
    c.gain.value = c.correspondence.value === 'late-g-lens' ? '0.8' : '0.3';
    apply(true);
    markChanged();
  });

  c.algo.addEventListener('change', () => {
    const chosen = algoDefaults[c.algo.value];
    if (chosen) {
      (Object.keys(chosen) as (keyof typeof c)[]).forEach((key) => {
        const control = c[key] as HTMLElement;
        const value = chosen[key];
        if (control && value !== undefined) setValue(control, value);
      });
    }
    apply(true);
    markChanged();
  });
  c.correspondence.addEventListener('change', () => apply(true));
  c.preset.addEventListener('change', () => selectPreset(c.preset.value as MeshPresetName));
  c.sequence.addEventListener('change', () => {
    if (solveTimer !== null) window.clearTimeout(solveTimer);
    solveTimer = null;
    clock = 0;
    laps = 0;
    handoffFrom = -1;
    waitingFor = -1;
    if (!sequenced()) {
      c.text.value = projectNames[0];
      c.clip.value = '0';
    }
    syncProject(0);
    if (sequenced()) selectPreset(DEFAULT_PRESET);
    else apply(false);
    markChanged();
  });
  c.play.addEventListener('click', () => {
    running = !running;
    c.play.textContent = running ? 'Hold cycle' : 'Run cycle';
  });
  c.time.addEventListener('input', () => {
    running = false;
    c.play.textContent = 'Run cycle';
    clock = Number(c.time.value);
    laps = 0;
    const wanted = Math.min(sequenceCount() - 1, Math.floor(clock / cycleOf(params).loop));
    if (wanted === 1) secondShowcase?.solveNow();
    if (wanted === 2) {
      ensureProjects(2);
      thirdShowcase?.solveNow();
    }
    if (wanted === 3) {
      ensureProjects(3);
      fourthShowcase?.solveNow();
    }
  });

  // The cloth lift's glyph glow, tuned live; ?bevel= sets where it starts.
  const bevelSlider = document.getElementById('m-cloth-bevel') as HTMLInputElement | null;
  const bevelOut = document.getElementById('m-cloth-bevel-out');
  if (bevelSlider) {
    bevelSlider.value = String(liftBevel());
    const showBevel = () => {
      setLiftBevel(Number(bevelSlider.value));
      if (bevelOut) bevelOut.textContent = Number(bevelSlider.value).toFixed(2);
    };
    bevelSlider.addEventListener('input', showBevel);
    showBevel();
  }

  // A query string can preset the panel and freeze a still, which is how a
  // headless Firefox screenshot can catch the hold rather than the idle.
  const query = new URLSearchParams(window.location.search);
  const preset = (key: string, control: HTMLInputElement | HTMLSelectElement) => {
    const value = query.get(key);
    if (value !== null) control.value = value;
  };
  preset('text', c.text);
  preset('cols', c.cols);
  preset('view', c.view);
  preset('footage', c.footage);
  preset('strength', c.strength);
  preset('algo', c.algo);
  preset('polarity', c.polarity);
  preset('sequence', c.sequence);
  const queryPreset = query.get('preset') as MeshPresetName | null;
  if (queryPreset && queryPreset !== 'custom' && queryPreset in MESH_PRESETS) selectPreset(queryPreset);
  // after the preset, which would put them back
  preset('regions', c.territoryRegions);
  preset('growth', c.territoryGrowth);
  preset('cloth', c.clothFrontier);
  preset('relief', c.clothRelief);
  preset('bounce', c.dropBounce);
  if (query.has('drop')) c.holdDrop.checked = query.get('drop') === '1';
  if (query.has('amp')) {
    c.ampAuto.checked = false;
    c.amp.value = query.get('amp') ?? '1';
  }
  if (query.has('t')) {
    clock = Number(query.get('t'));
    running = false;
    c.play.textContent = 'Run cycle';
  }
  if (query.get('run') === '1') {
    running = true;
    c.play.textContent = 'Hold cycle';
  }
  apply(false);
  markChanged();
  if (query.has('shot')) {
    ensureProjects(3);
    firstShowcase.solveNow();
    secondShowcase?.solveNow();
    thirdShowcase?.solveNow();
    fourthShowcase?.solveNow();
  }

  function paint() {
    const cycle = cycleOf(params);
    const total = cycle.loop * sequenceCount();
    if (clock >= total) {
      laps += Math.floor(clock / total);
      clock %= total;
    } else if (clock < 0) {
      clock = ((clock % total) + total) % total;
    }
    const wanted = Math.min(sequenceCount() - 1, Math.floor(clock / cycle.loop));
    if (sequenced()) {
      ensureProjects(wanted);
      // Hidden displays advance their solves and prepare a first frame. The
      // outgoing display of a handoff is drawn below, in step with the next.
      if (projectIndex !== 0 && handoffFrom !== 0) firstShowcase.frame(0);
      if (projectIndex !== 1 && handoffFrom !== 1) secondShowcase?.frame(0);
      if (projectIndex !== 2 && handoffFrom < 0) thirdShowcase?.frame(0, 14);
      if (projectIndex !== 3 && handoffFrom < 0) fourthShowcase?.frame(0, 14);
    }
    const incoming = display(wanted).engine;
    const incomingStatus = incoming?.status();
    if (wanted !== projectIndex && (!incomingStatus || incomingStatus.solving || incomingStatus.vertices === 0 || !incomingStatus.ready)) {
      clock = wanted > projectIndex ? wanted * cycle.loop - 0.001 : total - 0.001;
      waitingFor = wanted;
    } else if (projectIndex !== wanted) {
      waitingFor = -1;
      const previous = projectIndex;
      const blend = running && active && sequenced() && handoffSeconds() > 0;
      syncProject(wanted, blend ? previous : -1);
      params = projectParams(read(), wanted);
      labels();
      if (blend) handoffFrom = previous;
    } else {
      waitingFor = -1;
    }
    const localTime = clock - projectIndex * cycle.loop;
    const motion = clock + laps * total;
    // How many projects have been shown before this one. A lone project keeps
    // its borders from one loop to the next, so it stays at zero.
    const shown = sequenced() ? laps * sequenceCount() + projectIndex : 0;
    if (handoffFrom >= 0) {
      const blend = handoffSeconds();
      if (running && localTime < blend) {
        // Blend only during idle, where both projects' fields are at zero. The
        // outgoing project is drawn at the incoming one's time, so the two
        // canvases share one lattice and the blend only crossfades footage.
        showDisplay(projectIndex, active, 1, 1);
        showDisplay(handoffFrom, active, 1 - smoothUnit(localTime / blend), 3);
        display(handoffFrom).engine?.frame(localTime, 9, motion, shown - 1);
      } else {
        showDisplay(handoffFrom, false);
        showDisplay(projectIndex, active);
        handoffFrom = -1;
      }
    }
    // Keep the last good outgoing frame in its preserved drawing buffer while
    // the next display prepares. Repeated redraws at the release boundary can
    // expose an empty video texture if Firefox's decoder is catching up.
    if (waitingFor < 0) showcase.frame(localTime, 9, motion, shown);
    c.time.value = clock.toFixed(2);
    out('m-time-out').textContent = `${clock.toFixed(1)} s${sequenced() ? ` · ${projectNames[projectIndex]}` : ''}`;
    statusLine();
    drawCurve();
  }

  return {
    frame(delta) {
      if (!active) return;
      // Idle until the first word's field lands, so the opening cycle never
      // plays without it and the word never pops in partway through forming.
      const first = showcase.status();
      if (running && !(first.solving && first.solveMs === 0)) clock += delta;
      paint();
      const s = showcase.status();
      out('m-sched-out').textContent = `s ${s.strength.toFixed(2)}`;
      if (params.amplitudeAuto) out('m-amp-out').textContent = `auto ${s.amplitude.toFixed(2)}`;
      c.foldNote.textContent = `${s.constraints} constraints · invalid cells ${(s.invalidRate * 100).toFixed(1)}% · BD diagonal ${(s.diagonalBdRate * 100).toFixed(1)}% · ${s.vertices.toLocaleString()} vertices · solve ${s.solveMs.toFixed(0)} ms`;
    },
    setActive(next) {
      active = next;
      handoffFrom = -1;
      waitingFor = -1;
      syncProject(projectIndex);
      if (next) {
        apply(false);
        labels();
        paint(); // draw at once, so switching treatment is not a blank frame
      }
    },
  };
}

export async function startShowcaseTest() {
  const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
  let canvas = element<HTMLCanvasElement>('showcase');
  const mask = element<HTMLVideoElement>('mask-video');
  const generatedMaskCanvas = element<HTMLCanvasElement>('generated-mask');
  const footage = Array.from(document.querySelectorAll<HTMLVideoElement>('video[data-clip]'));
  const meshFootage = [...footage, element<HTMLVideoElement>('media').querySelector<HTMLVideoElement>('video[data-mesh-clip]')!];
  const videos = [mask, ...meshFootage];
  const maskButton = element<HTMLButtonElement>('mask-toggle');
  const footageButton = element<HTMLButtonElement>('footage-toggle');
  const timeline = element<HTMLInputElement>('timeline');
  const time = element<HTMLOutputElement>('time');
  const speed = element<HTMLSelectElement>('speed');
  const density = element<HTMLInputElement>('density');
  const densityValue = element<HTMLOutputElement>('density-value');
  const fluidity = element<HTMLInputElement>('fluidity');
  const fluidityValue = element<HTMLOutputElement>('fluidity-value');
  const copies = element<HTMLInputElement>('copies');
  const copiesValue = element<HTMLOutputElement>('copies-value');
  const pointTextToggle = element<HTMLInputElement>('point-text');
  const pointTopologySelect = element<HTMLSelectElement>('point-topology');
  const status = element('status');
  const loading = element('loading');
  const scopeNote = element('scope-note');
  const meshCanvas = element<HTMLCanvasElement>('showcase-mesh');
  const meshDebugCanvas = element<HTMLCanvasElement>('showcase-mesh-debug');
  const meshNextCanvas = element<HTMLCanvasElement>('showcase-mesh-next');
  const meshNextDebugCanvas = element<HTMLCanvasElement>('showcase-mesh-next-debug');
  const meshThirdCanvas = element<HTMLCanvasElement>('showcase-mesh-third');
  const meshThirdDebugCanvas = element<HTMLCanvasElement>('showcase-mesh-third-debug');
  const meshFourthCanvas = element<HTMLCanvasElement>('showcase-mesh-fourth');
  const meshFourthDebugCanvas = element<HTMLCanvasElement>('showcase-mesh-fourth-debug');
  const treatment = element<HTMLSelectElement>('treatment');
  const legacyControls = element('legacy-controls');
  const meshControls = element('mesh-controls');
  const legacyNotes = element('legacy-notes');
  let meshPanel: MeshPanel | null = null;
  const meshActive = () => treatment.value === 'mesh' && meshPanel !== null;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let maskMoving = !reduced.matches;
  let footageMoving = !reduced.matches;
  let initialized = false;
  let dirty = true;
  let selectorDirty = true;
  let generatedMask: GeneratedMask | null = null;
  let bakedMask: BakedMask | null = null;
  let bakeTimer: number | null = null;
  let bakeVersion = 0;
  let bakePending = false;
  let baking = false;
  let maskAvailable = false;
  let maskDuration = MASK_DURATION;
  let fallbackMaskTime = 0;
  let fallbackMaskRate = 1;
  const settings = { view: 0, copies: Number(copies.value) };
  const currentMaskTime = () => maskAvailable && Number.isFinite(mask.currentTime)
    ? mask.currentTime
    : fallbackMaskTime;

  function updateControlState() {
    maskButton.disabled = !initialized;
    timeline.disabled = !initialized;
    speed.disabled = !initialized;
    density.disabled = !initialized;
    fluidity.disabled = !initialized;
    pointTopologySelect.disabled = !initialized || !pointTextToggle.checked;
  }

  function updatePieceLabels() {
    densityValue.textContent = `${generatedMask?.pieceCount ?? Number(density.value)} pieces`;
    fluidityValue.textContent = `${fluidity.value}%`;
    copiesValue.textContent = `${copies.value} ${Number(copies.value) === 1 ? 'copy' : 'copies'} / source`;
  }

  function labels() {
    maskButton.textContent = maskMoving ? 'Hold mask' : 'Move mask';
    footageButton.textContent = footageMoving ? 'Pause footage' : 'Play footage';
    if (meshActive()) {
      status.textContent = `Deformed mesh \u00b7 ${footageMoving ? 'footage playing' : 'footage paused'}`;
      scopeNote.textContent = 'The lattice is the geometry, not a selector: undeformed positions are texture coordinates, deformed positions are vertex positions. The deformation is solved once per word and lattice; per frame the page evaluates the breathing wobble, adds amplitude times the displacement, derives the shading term from the deformed cell areas, and draws one textured mesh. No mask, no clip-ID quantisation, no baking.';
      meshCanvas.setAttribute('aria-label', 'A fine lattice deforms until its own edges arrange into the word, with footage textured through the deformed mesh.');
      updatePieceLabels();
      return;
    }
    const maskStatus = !pointTextToggle.checked
      ? maskMoving ? 'Generated mask moving' : 'Generated mask held'
      : baking
        ? 'Baking point-by-point mask'
        : bakePending
          ? 'Point-by-point mask update queued'
          : bakedMask
            ? maskMoving ? 'Baked mask playing' : 'Baked mask held'
            : 'Point-by-point mask waiting';
    status.textContent = `${maskStatus} · ${footageMoving ? 'footage playing' : 'footage paused'}`;
    const topologyName = pointTopologySelect.selectedOptions[0]?.textContent ?? 'the selected point topology';
    scopeNote.textContent = pointTextToggle.checked
      ? `The selector mask is the sole clip-ID field. Point-by-point mode is using ${topologyName}; its complete animation is baked once and replayed while the footage plays.`
      : 'The selector mask is the sole clip-ID field, generated from the four-point clip field. These defaults reproduce the original static 12-piece mask; raise density, edge fluidity, or Copies / source to alter that same mask. Toggle Point-by-point text to compare the preserved contour-face bridge with the sibling lattice-topology variants.';
    canvas.setAttribute('aria-label', pointTextToggle.checked
      ? 'Yope3D lettering is represented by polygon faces inserted into a moving lattice over four videos.'
      : 'Yope3D lettering floats as a selector mask over a collage of four videos selected by a moving lattice.');
    updatePieceLabels();
  }

  function applyTreatment() {
    const mesh = meshActive();
    canvas.hidden = mesh;
    meshCanvas.hidden = !mesh;
    legacyControls.hidden = mesh;
    meshControls.hidden = !mesh;
    // The bottom notes and gray-level legend describe the selector mask.
    legacyNotes.hidden = mesh;
    meshPanel?.setActive(mesh);
    labels();
    dirty = true;
  }

  async function playback() {
    if (!initialized) return;
    const wanted = [maskMoving && maskAvailable, ...meshFootage.map(() => footageMoving)];
    const results = await Promise.allSettled(videos.map((video, index) => {
      if (wanted[index] && !document.hidden) return video.play();
      video.pause();
      return Promise.resolve();
    }));
    if (results[0].status === 'rejected') {
      const heldTime = Number.isFinite(mask.currentTime) ? mask.currentTime : fallbackMaskTime;
      maskAvailable = false;
      fallbackMaskTime = heldTime;
    }
    if (results.slice(1).every((result) => result.status === 'rejected')) {
      footageMoving = false;
      meshFootage.forEach((video) => video.pause());
    }
    labels();
    dirty = true;
  }

  try {
    videos.forEach((video) => { video.muted = true; });
    const updateMaskMetadata = () => {
      if (!Number.isFinite(mask.duration) || mask.duration <= 0) return;
      maskAvailable = true;
      maskDuration = mask.duration;
      timeline.max = String(maskDuration);
      dirty = true;
      if (initialized && maskMoving) void playback();
    };
    mask.addEventListener('loadedmetadata', updateMaskMetadata);
    mask.addEventListener('durationchange', updateMaskMetadata);
    mask.addEventListener('error', () => {
      maskAvailable = false;
      maskDuration = MASK_DURATION;
      dirty = true;
    });
    meshFootage.forEach((video) => video.addEventListener('loadeddata', () => {
      dirty = true;
      void playback();
    }));
    const reference = await readyImage('/media/showcase-test/letter-mask.png');
    updateMaskMetadata();
    const word = wordMaskFromReference(reference);
    generatedMask = createGeneratedMask(
      generatedMaskCanvas,
      word,
      settings.copies,
      pointTextToggle.checked,
      pointTopologySelect.value as PointTopology,
    );
    const readPointConfig = (): PointMaskConfig => ({
      density: Math.max(12, Math.min(96, Math.round(Number(density.value) / 4) * 4)),
      fluidity: Math.max(0, Math.min(100, Number(fluidity.value))),
      copies: Math.max(1, Math.min(MAX_COPIES, Math.round(Number(copies.value)))),
      topology: pointTopologySelect.value as PointTopology,
    });
    const pointMaskKey = () => {
      const config = readPointConfig();
      return `${config.density}:${config.fluidity}:${config.copies}:${config.topology}:${maskDuration}`;
    };
    const cancelPointBake = () => {
      bakeVersion++;
      if (bakeTimer !== null) {
        window.clearTimeout(bakeTimer);
        bakeTimer = null;
      }
      bakePending = false;
      baking = false;
    };
    const bakeCurrentPointMask = async () => {
      if (!pointTextToggle.checked || !generatedMask) return;
      const config = readPointConfig();
      const key = pointMaskKey();
      if (bakedMask?.key === key) {
        bakePending = false;
        labels();
        return;
      }
      const version = bakeVersion;
      baking = true;
      bakePending = false;
      labels();
      try {
        const data = await bakePointMask(
          word,
          config,
          maskDuration,
          key,
          () => version !== bakeVersion || !pointTextToggle.checked,
          (complete, total) => {
            if (version === bakeVersion) {
              status.textContent = `Baking point-by-point mask ${Math.round(complete / total * 100)}% · ${footageMoving ? 'footage playing' : 'footage paused'}`;
            }
          },
        );
        if (!data || version !== bakeVersion || !pointTextToggle.checked) return;
        bakedMask = createBakedMask(data, generatedMask.canvas);
        generatedMask.setDensity(config.density);
        generatedMask.setFluidity(config.fluidity);
        generatedMask.setCopies(config.copies);
        generatedMask.setPointTopology(config.topology);
        generatedMask.setPointText(true);
        settings.copies = config.copies;
        selectorDirty = true;
        dirty = true;
      } catch (error) {
        if (version === bakeVersion) console.error(error);
      } finally {
        if (version === bakeVersion) {
          baking = false;
          labels();
        }
      }
    };
    const schedulePointBake = (immediate = false) => {
      if (!pointTextToggle.checked) return;
      bakeVersion++;
      if (bakeTimer !== null) window.clearTimeout(bakeTimer);
      bakePending = true;
      bakeTimer = window.setTimeout(() => {
        bakeTimer = null;
        void bakeCurrentPointMask();
      }, immediate ? 0 : 260);
      labels();
    };
    let renderer: Renderer | null = null;
    try { renderer = createWebGLRenderer(canvas, footage); } catch { /* Try the CPU compositor below. */ }
    if (!renderer) {
      // A canvas that has acquired a GL context cannot acquire a 2D context.
      const replacement = canvas.cloneNode() as HTMLCanvasElement;
      canvas.replaceWith(replacement);
      canvas = replacement;
      renderer = createCanvasRenderer(canvas, footage);
    }
    const activeRenderer = renderer;
    try {
      meshPanel = createMeshPanel(meshCanvas, meshFootage, meshDebugCanvas, meshNextCanvas, meshNextDebugCanvas, meshThirdCanvas, meshThirdDebugCanvas, meshFourthCanvas, meshFourthDebugCanvas);
    } catch (error) {
      console.error(error);
      meshCanvas.dataset.meshError = error instanceof Error ? error.message : String(error);
      const meshError = error instanceof Error ? error.message : String(error);
      const meshStatus = document.getElementById('m-status');
      if (meshStatus) meshStatus.textContent = `mesh error: ${meshError}`;
      meshPanel = null;
    }
    if (!meshPanel) {
      const meshOption = treatment.querySelector('option[value="mesh"]');
      if (meshOption instanceof HTMLOptionElement) meshOption.disabled = true;
      treatment.value = 'mask';
    }
    treatment.addEventListener('change', () => applyTreatment());
    applyTreatment();
    initialized = true;
    footageButton.disabled = false;
    updateControlState();
    timeline.max = String(maskDuration);
    renderer.draw(settings, generatedMask.canvas, true);
    selectorDirty = false;
    loading.hidden = true;
    element('stage').setAttribute('aria-busy', 'false');
    element('stage').dataset.ready = 'true';
    await playback();

    maskButton.addEventListener('click', () => { maskMoving = !maskMoving; void playback(); });
    footageButton.addEventListener('click', () => { footageMoving = !footageMoving; void playback(); });
    timeline.addEventListener('input', () => {
      maskMoving = false;
      const nextTime = Math.min(Number(timeline.value), Math.max(0, maskDuration - 0.04));
      if (maskAvailable) {
        mask.pause();
        mask.currentTime = nextTime;
      } else {
        fallbackMaskTime = nextTime;
      }
      if (!pointTextToggle.checked) selectorDirty = true;
      labels();
      dirty = true;
    });
    density.addEventListener('input', (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      if (pointTextToggle.checked) {
        schedulePointBake();
      } else {
        generatedMask?.setDensity(value);
        selectorDirty = true;
      }
      labels();
      dirty = true;
    });
    fluidity.addEventListener('input', (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      if (pointTextToggle.checked) {
        schedulePointBake();
      } else {
        generatedMask?.setFluidity(value);
        selectorDirty = true;
      }
      labels();
      dirty = true;
    });
    copies.addEventListener('input', (event) => {
      const value = Math.max(1, Math.min(MAX_COPIES, Math.round(Number((event.target as HTMLInputElement).value))));
      if (pointTextToggle.checked) {
        schedulePointBake();
      } else {
        settings.copies = value;
        generatedMask?.setCopies(settings.copies);
        selectorDirty = true;
      }
      labels();
      dirty = true;
    });
    pointTextToggle.addEventListener('change', (event) => {
      const enabled = (event.target as HTMLInputElement).checked;
      if (enabled) {
        generatedMask?.setPointTopology(pointTopologySelect.value as PointTopology);
        generatedMask?.setPointText(true);
        selectorDirty = true;
        schedulePointBake(true);
      } else {
        cancelPointBake();
        const config = readPointConfig();
        generatedMask?.setDensity(config.density);
        generatedMask?.setFluidity(config.fluidity);
        settings.copies = config.copies;
        generatedMask?.setCopies(settings.copies);
        generatedMask?.setPointTopology(config.topology);
        generatedMask?.setPointText(false);
        selectorDirty = true;
      }
      labels();
      updateControlState();
      dirty = true;
    });
    pointTopologySelect.addEventListener('change', () => {
      const topology = pointTopologySelect.value as PointTopology;
      generatedMask?.setPointTopology(topology);
      if (pointTextToggle.checked) schedulePointBake(true);
      labels();
      dirty = true;
    });
    videos.forEach((video, index) => video.addEventListener('seeked', () => {
      dirty = true;
      if (index === 0 && !pointTextToggle.checked) selectorDirty = true;
    }));
    speed.addEventListener('change', (event) => {
      fallbackMaskRate = Number((event.target as HTMLSelectElement).value);
      if (maskAvailable) mask.playbackRate = fallbackMaskRate;
    });
    element<HTMLSelectElement>('view').addEventListener('change', (event) => {
      settings.view = Number((event.target as HTMLSelectElement).value);
      dirty = true;
    });
    document.addEventListener('visibilitychange', () => { void playback(); });
    reduced.addEventListener('change', () => {
      if (reduced.matches) {
        maskMoving = false;
        footageMoving = false;
      } else {
        maskMoving = true;
        footageMoving = true;
      }
      void playback();
    });
    let lastFrame = 0;
    const tick = (now: number) => {
      if (!document.hidden && meshActive() && meshPanel) {
        meshPanel.frame(lastFrame > 0 ? Math.min(0.1, (now - lastFrame) / 1000) : 0);
        lastFrame = now;
      } else if (!document.hidden && now - lastFrame >= 1000 / 30 && (dirty || maskMoving || footageMoving)) {
        if (!maskAvailable && maskMoving && lastFrame > 0) {
          fallbackMaskTime = (fallbackMaskTime + (now - lastFrame) / 1000 * fallbackMaskRate) % maskDuration;
        }
        const maskTime = currentMaskTime();
        let maskChanged = selectorDirty;
        if (pointTextToggle.checked) {
          // Keep the last completed bake playing while a debounced slider
          // update is producing its replacement.
          if (bakedMask) maskChanged = bakedMask.render(maskTime) || maskChanged;
        } else if (maskMoving || selectorDirty) {
          generatedMask?.update(maskTime);
          maskChanged = true;
        }
        activeRenderer.draw(settings, generatedMask!.canvas, maskChanged);
        selectorDirty = false;
        timeline.value = String(maskTime);
        time.textContent = `${maskTime.toFixed(1)} / ${maskDuration.toFixed(1)}s`;
        dirty = false;
        lastFrame = now;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  } catch (error) {
    console.error(error);
    loading.textContent = 'The preview could not load. Try reloading, or open the mask movie below.';
    status.textContent = 'Preview unavailable';
    element('stage').setAttribute('aria-busy', 'false');
  }
}
