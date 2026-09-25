"""Agent D probe: contour-seeded organic tessellation for the opening showcase.

The quad lattice is replaced entirely.  The selector mask is the Voronoi
diagram of a moving point set.  Sites are seeded in *pairs that straddle the
glyph contour*: for a contour sample ``p`` with inward unit normal ``n`` the
pair is ``p + n*h`` (inside) and ``p - n*h`` (outside).  The perpendicular
bisector of that pair is the tangent line at ``p``, so a Voronoi edge lands
exactly on the letter outline.  Everywhere else the field is free sites,
Lloyd-relaxed into an organic, even, non-splintery cell field.

Consequences that matter for the brief:

* Voronoi cell boundaries are straight segments by construction, so the
  "no spline boundaries" rule is satisfied without post-processing.
* The letter is a *union of many cells* (one per contour sample on each
  side, plus free sites that fall deep inside thick strokes), not a stamp.
* Sliding a pair tangentially along the contour leaves the bisector where it
  is: the pieces grow and shrink along the letter edge while the outline
  stays exact.  That is the boiling character without losing legibility.
* Deleting the contour seeds leaves a plain organic field, so the name can
  form and dissolve with the same machinery.

This file is self-contained.  It does not import or modify
``scripts/inspect-point-mask.py``.

Tasks:
    --task matrix   render the site-count x copies x word matrix + contact sheets
    --task single   one configuration: mask, composite, edges, no-seed frame
    --task video    one 12 s loop (lossless VP9 mask + H.264 composite preview)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = Path('/System/Library/Fonts/Supplemental/Arial Black.ttf')
UI_FONT = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
UI_FONT_BOLD = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')

WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 30, 12
TAU = math.tau

# Same crop/zoom/offset table the browser compositor uses for copy variants.
COPY_TRANSFORMS = [
    (1.00, 0.00, 0.00), (1.12, -0.10, 0.08), (1.24, 0.12, -0.08),
    (1.36, -0.15, -0.12), (1.48, 0.08, 0.12), (1.60, -0.04, 0.02),
    (1.18, 0.16, 0.14), (1.30, -0.18, 0.04), (1.44, 0.05, -0.16),
    (1.56, 0.14, -0.02), (1.70, -0.12, 0.12), (1.84, 0.02, -0.12),
]

PIXEL_XY = None          # lazily built (H*W, 2) sample grid
_VARIANT_CACHE: dict = {}
_CLIP_CACHE: dict = {}


# --------------------------------------------------------------------------
# footage
# --------------------------------------------------------------------------

def clip_frames(frames_dir: Path, equalize: bool = False):
    """Return four 1280x720 RGB uint8 arrays, one still per source clip."""
    key = ('eq' if equalize else 'raw')
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]
    clips = []
    for index in range(4):
        image = Image.open(frames_dir / f'clip{index}.png').convert('RGB')
        array = np.asarray(image.resize((WIDTH, HEIGHT), Image.LANCZOS)).astype(np.float32)
        if equalize:
            # Only for the diagnostic panel: the shipped gravity clip is
            # essentially pure black, which hides the geometry being judged.
            luma = array.mean(axis=2)
            mean, std = luma.mean(), max(luma.std(), 1.0)
            array = (array - mean) / std * 46.0 + 118.0
        clips.append(np.clip(array, 0, 255).astype(np.uint8))
    _CLIP_CACHE[key] = clips
    return clips


def variant(source: int, copy: int, frames_dir: Path, equalize: bool = False) -> np.ndarray:
    """One copy variant of one source, framed like the browser's atlas tile."""
    key = (source, copy, equalize)
    if key in _VARIANT_CACHE:
        return _VARIANT_CACHE[key]
    zoom, offset_x, offset_y = COPY_TRANSFORMS[(copy + source) % len(COPY_TRANSFORMS)]
    base = Image.fromarray(clip_frames(frames_dir, equalize)[source])
    scale = zoom
    width, height = int(round(WIDTH * scale)), int(round(HEIGHT * scale))
    scaled = base.resize((width, height), Image.LANCZOS)
    canvas = Image.new('RGB', (WIDTH, HEIGHT), (11, 13, 16))
    canvas.paste(scaled, (int(round((WIDTH - width) / 2 + offset_x * WIDTH)),
                          int(round((HEIGHT - height) / 2 + offset_y * HEIGHT))))
    array = np.asarray(canvas)
    if len(_VARIANT_CACHE) > 120:
        _VARIANT_CACHE.clear()
    _VARIANT_CACHE[key] = array
    return array


def slot_luma(copies: int, frames_dir: Path, equalize: bool = False) -> np.ndarray:
    """Mean luminance of each slot's footage; used by the optional repair."""
    means = np.zeros(4 * copies)
    for source in range(4):
        for copy in range(copies):
            means[source * copies + copy] = variant(source, copy, frames_dir, equalize).mean()
    return means


# --------------------------------------------------------------------------
# glyph raster and straight-segment contours
# --------------------------------------------------------------------------

def render_word(word: str, max_width: int = 1150, max_height: int = 300):
    """Rasterize the word with Arial Black, largest size that fits the box."""
    chosen = None
    for size in range(300, 23, -2):
        font = ImageFont.truetype(str(FONT_PATH), size)
        left, top, right, bottom = font.getbbox(word)
        if right - left <= max_width and bottom - top <= max_height:
            chosen = (size, font, (left, top, right, bottom))
            break
    if chosen is None:
        raise SystemExit(f'cannot fit {word!r}')
    size, font, (left, top, right, bottom) = chosen
    pad = 6
    image = Image.new('L', (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    ImageDraw.Draw(image).text((pad - left, pad - top), word, font=font, fill=255)
    return (np.asarray(image) >= 128), size


_MARCHING_CASES = {
    1: [('top', 'left')], 2: [('top', 'right')], 3: [('left', 'right')],
    4: [('right', 'bottom')], 5: [('top', 'left'), ('right', 'bottom')],
    6: [('top', 'bottom')], 7: [('left', 'bottom')], 8: [('left', 'bottom')],
    9: [('top', 'bottom')], 10: [('top', 'right'), ('left', 'bottom')],
    11: [('right', 'bottom')], 12: [('left', 'right')],
    13: [('top', 'right')], 14: [('top', 'left')],
}


def trace_contours(filled: np.ndarray):
    """Marching-squares loops of the glyph, in half-pixel integer coordinates."""
    grid = np.pad(filled.astype(np.uint8), 1)
    code = (grid[:-1, :-1] * 1 + grid[:-1, 1:] * 2
            + grid[1:, 1:] * 4 + grid[1:, :-1] * 8)
    segments: list[tuple] = []
    nodes: dict = {}

    def edge_point(kind, x, y):
        if kind == 'top':
            return (2 * x + 1, 2 * y)
        if kind == 'right':
            return (2 * x + 2, 2 * y + 1)
        if kind == 'bottom':
            return (2 * x + 1, 2 * y + 2)
        return (2 * x, 2 * y + 1)

    for value, pairs in _MARCHING_CASES.items():
        ys, xs = np.nonzero(code == value)
        for x, y in zip(xs.tolist(), ys.tolist()):
            for first, second in pairs:
                a, b = edge_point(first, x, y), edge_point(second, x, y)
                index = len(segments)
                segments.append((a, b))
                nodes.setdefault(a, []).append(index)
                nodes.setdefault(b, []).append(index)

    unused = set(range(len(segments)))
    loops = []
    while unused:
        first_index = next(iter(unused))
        start = segments[first_index][0]
        current, previous, loop = start, None, []
        while True:
            candidates = [i for i in nodes[current] if i in unused]
            if previous is not None and len(candidates) > 1:
                candidates = [i for i in candidates if i != previous] or candidates
            if not candidates:
                break
            index = candidates[0]
            unused.discard(index)
            a, b = segments[index]
            loop.append(((current[0] - 2) / 2.0, (current[1] - 2) / 2.0))
            current = b if a == current else a
            previous = index
            if current == start:
                break
        if current == start and len(loop) >= 4:
            loops.append(np.asarray(loop, dtype=np.float64))
    return loops


def resample_loop(loop: np.ndarray, spacing: float, minimum: int = 8) -> np.ndarray:
    """Even arc-length resampling; every closed loop keeps at least `minimum`."""
    closed = np.vstack([loop, loop[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    total = arc[-1]
    if total <= 1e-6:
        return loop[:1]
    count = max(minimum, int(round(total / spacing)))
    targets = np.linspace(0.0, total, count, endpoint=False)
    xs = np.interp(targets, arc, closed[:, 0])
    ys = np.interp(targets, arc, closed[:, 1])
    return np.stack([xs, ys], axis=1)


def sample_nearest(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, field.shape[0] - 1)
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, field.shape[1] - 1)
    return field[ys, xs]


def contour_sites(word: str, spacing_scale: float, depth: float,
                  spacing_override: float | None = None):
    """Contour samples with inward normals, per-sample depth and arc length.

    Depth is solved symmetrically: the pair offset `h` is shrunk until both
    sites sit at least `h` from *any* wall.  A symmetric `h` keeps the pair's
    bisector on the contour; an asymmetric one would push it off.
    """
    filled, font_size = render_word(word)
    inside_distance = distance_transform_edt(filled)
    outside_distance = distance_transform_edt(~filled)
    stroke_half = float(np.percentile(inside_distance[filled], 72)) if filled.any() else 10.0

    base_spacing = spacing_override if spacing_override else max(
        9.0, min(40.0, 0.95 * stroke_half * spacing_scale))
    max_depth = min(depth, 0.42 * stroke_half)

    points, normals, arcs, loop_ids = [], [], [], []
    for loop_index, loop in enumerate(trace_contours(filled)):
        sampled = resample_loop(loop, base_spacing)
        if len(sampled) < 4:
            continue
        forward = np.roll(sampled, -1, axis=0) - np.roll(sampled, 1, axis=0)
        length = np.linalg.norm(forward, axis=1)
        length[length == 0] = 1.0
        tangent = forward / length[:, None]
        candidate = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        probe_plus = sample_nearest(filled, sampled + candidate * 2.5)
        probe_minus = sample_nearest(filled, sampled - candidate * 2.5)
        sign = np.where(probe_plus & ~probe_minus, 1.0,
                        np.where(probe_minus & ~probe_plus, -1.0, 0.0))
        if (sign == 0).all():
            continue
        # Fill ambiguous samples (thin spots) from the loop's majority sign.
        majority = 1.0 if sign.sum() >= 0 else -1.0
        sign = np.where(sign == 0, majority, sign)
        inward = candidate * sign[:, None]

        steps = np.linalg.norm(np.diff(np.vstack([sampled, sampled[:1]]), axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(steps)])[:-1]
        arc = arc / max(arc[-1] + steps[-1], 1e-6)

        points.append(sampled)
        normals.append(inward)
        arcs.append(arc)
        loop_ids.append(np.full(len(sampled), loop_index))

    if not points:
        raise SystemExit(f'no contours for {word!r}')
    points = np.vstack(points)
    normals = np.vstack(normals)
    arcs = np.concatenate(arcs)
    loop_ids = np.concatenate(loop_ids)

    # Symmetric depth solve over a descending ladder of candidate depths.
    depths = np.full(len(points), 2.0)
    for factor in (1.0, 0.85, 0.72, 0.6, 0.5, 0.42, 0.35, 0.28, 0.22):
        h = max_depth * factor
        if h < 2.0:
            break
        inner = sample_nearest(inside_distance, points + normals * h)
        outer = sample_nearest(outside_distance, points - normals * h)
        ok = (inner >= h * 0.9) & (outer >= h * 0.9) & (depths <= 2.0 + 1e-9)
        depths = np.where(ok, h, depths)
    depths = np.maximum(depths, 2.0)

    return {
        'filled': filled,
        'font_size': font_size,
        'stroke_half': stroke_half,
        'spacing': base_spacing,
        'points': points,
        'normals': normals,
        'arcs': arcs,
        'loops': loop_ids,
        'depths': depths,
    }


# --------------------------------------------------------------------------
# site field
# --------------------------------------------------------------------------

def hashed(values: np.ndarray, salt: float) -> np.ndarray:
    raw = np.sin(values * 12.9898 + salt * 78.233) * 43758.5453
    return raw - np.floor(raw)


def organic_sources(xy: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    """Four overlapping low-frequency fields; the winner picks the source.

    This is a single field evaluated identically for every site, inside the
    letters and outside them.  There is no inside/outside palette family:
    all four sources occur on both sides of the contour.
    """
    x = xy[:, 0] / WIDTH
    y = xy[:, 1] / HEIGHT
    fields = np.empty((4, len(xy)))
    for k in range(4):
        a = 2.2 + 0.9 * k
        b = 1.7 + 1.3 * ((k * 3) % 4)
        fields[k] = (np.sin(TAU * (a * x + 0.35 * b * y) + 1.7 * k)
                     + 0.70 * np.sin(TAU * (0.5 * b * x - 1.1 * a * y) + 2.9 * k + 1.1)
                     + 0.45 * np.sin(TAU * (1.9 * a * x + 1.5 * b * y) + 0.8 * k))
    fields += (jitter - 0.5)[None, :] * 1.15
    return np.argmax(fields, axis=0)


def free_sites(count: int, keep_out: np.ndarray | None, rng: np.random.Generator,
               clearance: float, relax_steps: int = 4, pinned: np.ndarray | None = None):
    """Jittered-grid seeding, Lloyd relaxation, contour clearance enforced."""
    columns = max(1, int(round(math.sqrt(max(count, 1) * WIDTH / HEIGHT))))
    rows = max(1, int(math.ceil(count / columns)))
    xs = (np.arange(columns) + 0.5) / columns * WIDTH
    ys = (np.arange(rows) + 0.5) / rows * HEIGHT
    grid = np.stack(np.meshgrid(xs, ys, indexing='xy'), axis=-1).reshape(-1, 2)
    grid = grid + rng.uniform(-0.42, 0.42, grid.shape) * np.array(
        [WIDTH / columns, HEIGHT / rows])
    if len(grid) > count:
        grid = grid[rng.permutation(len(grid))[:count]]
    sites = np.clip(grid, [6, 6], [WIDTH - 6, HEIGHT - 6])

    keep_tree = cKDTree(keep_out) if keep_out is not None and len(keep_out) else None

    def push_out(points):
        if keep_tree is None:
            return points
        distance, index = keep_tree.query(points)
        close = distance < clearance
        if close.any():
            direction = points[close] - keep_out[index[close]]
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            norm[norm < 1e-6] = 1.0
            direction = direction / norm
            points = points.copy()
            points[close] = keep_out[index[close]] + direction * clearance
        return np.clip(points, [4, 4], [WIDTH - 4, HEIGHT - 4])

    sites = push_out(sites)
    if relax_steps and len(sites):
        # Discrete Lloyd on a coarse grid; pinned pair sites take part in the
        # tessellation but never move.
        gx, gy = np.meshgrid(np.linspace(0, WIDTH - 1, 256),
                             np.linspace(0, HEIGHT - 1, 144), indexing='xy')
        probe = np.stack([gx.ravel(), gy.ravel()], axis=1)
        for _ in range(relax_steps):
            allsites = sites if pinned is None or not len(pinned) else np.vstack([sites, pinned])
            _, nearest = cKDTree(allsites).query(probe, workers=-1)
            counts = np.bincount(nearest, minlength=len(allsites))[:len(sites)]
            sum_x = np.bincount(nearest, weights=probe[:, 0], minlength=len(allsites))[:len(sites)]
            sum_y = np.bincount(nearest, weights=probe[:, 1], minlength=len(allsites))[:len(sites)]
            moved = sites.copy()
            live = counts > 0
            moved[live, 0] = sum_x[live] / counts[live]
            moved[live, 1] = sum_y[live] / counts[live]
            sites = push_out(0.5 * sites + 0.5 * moved)
    return sites


def build_field(word: str, density: int, copies: int, seed: int = 7,
                depth: float = 9.0, spacing_scale: float | None = None,
                spacing_override: float | None = None, with_contour: bool = True,
                repair: str = 'source', frames_dir: Path | None = None,
                equalize: bool = False):
    """Assemble the static description of one configuration."""
    rng = np.random.default_rng(seed)
    if spacing_scale is None:
        spacing_scale = (12.0 / max(density, 1)) ** 0.20

    contour = contour_sites(word, spacing_scale, depth, spacing_override) if with_contour else None
    glyph = contour['filled'] if contour is not None else None

    if contour is not None:
        pair_count = len(contour['points'])
        clearance = float(np.hypot(contour['depths'].max(),
                                   contour['spacing'] / 2) * 1.3 + 4.0)
    else:
        pair_count = 0
        clearance = 0.0

    word_h, word_w = (glyph.shape if glyph is not None else (0, 0))
    base_word = np.array([(WIDTH - word_w) / 2.0, (HEIGHT - word_h) / 2.0])

    keep_out = (contour['points'] + base_word) if contour is not None else None
    free = free_sites(density, keep_out, rng, clearance, 4,
                      pinned=None)

    total = len(free) + 2 * pair_count
    jitter = hashed(np.arange(total).astype(float) + 0.5, seed)
    if contour is not None:
        reference = np.vstack([
            free,
            contour['points'] + base_word + contour['normals'] * contour['depths'][:, None],
            contour['points'] + base_word - contour['normals'] * contour['depths'][:, None],
        ])
    else:
        reference = free
    sources = organic_sources(reference, jitter)
    copy_index = (np.floor(hashed(np.arange(total).astype(float) + 11.0, seed + 3) * copies)
                  ).astype(int) % max(copies, 1)

    inner_slice = slice(len(free), len(free) + pair_count)
    outer_slice = slice(len(free) + pair_count, total)
    repaired = 0
    if contour is not None and repair != 'none':
        inner = sources[inner_slice].copy()
        outer = sources[outer_slice]
        if repair == 'source':
            clash = inner == outer
            bump = 1 + (np.floor(hashed(np.arange(pair_count).astype(float), seed + 5) * 3)
                        ).astype(int)
            inner[clash] = (outer[clash] + bump[clash]) % 4
            repaired = int(clash.sum())
        elif repair == 'contrast':
            assert frames_dir is not None
            luma = slot_luma(copies, frames_dir, equalize).reshape(4, copies).mean(axis=1)
            order = np.argsort(hashed(np.arange(pair_count).astype(float), seed + 5))
            for i in range(pair_count):
                out_source = outer[i]
                gaps = np.abs(luma - luma[out_source])
                allowed = np.nonzero(gaps >= 24.0)[0]
                if len(allowed) == 0:
                    allowed = np.nonzero(np.arange(4) != out_source)[0]
                if inner[i] not in allowed:
                    inner[i] = allowed[order[i] % len(allowed)]
                    repaired += 1
        sources = sources.copy()
        sources[inner_slice] = inner

    slots = sources * copies + copy_index
    values = np.round(slots * 255.0 / (4 * copies - 1)).astype(np.uint8)

    return {
        'word': word,
        'density': density,
        'copies': copies,
        'contour': contour,
        'glyph': glyph,
        'base_word': base_word,
        'free': free,
        'pair_count': pair_count,
        'clearance': clearance,
        'sources': sources,
        'slots': slots,
        'values': values,
        'repaired': repaired,
        'seed': seed,
        'inner_slice': inner_slice,
        'outer_slice': outer_slice,
    }


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------

def word_offset(t: float) -> np.ndarray:
    phase = t / SECONDS * TAU
    return np.array([125.0 * math.sin(phase), 120.0 * math.sin(phase * 2)])


def sites_at(field: dict, t: float, fluidity: float = 1.0):
    """Positions of every site at time t.  Loop-periodic, so 12 s cycles."""
    omega = TAU / SECONDS
    free = field['free']
    seed = field['seed']
    n = len(free)
    if n:
        ph = hashed(np.arange(n).astype(float), seed + 21) * TAU
        qh = hashed(np.arange(n).astype(float), seed + 33) * TAU
        cell = math.sqrt(WIDTH * HEIGHT / max(n, 1))
        amplitude = 0.26 * cell * fluidity
        dx = amplitude * (0.68 * np.sin(omega * t + ph) + 0.32 * np.sin(2 * omega * t + qh))
        dy = amplitude * (0.68 * np.sin(omega * t + qh) + 0.32 * np.cos(2 * omega * t + ph))
        drift = np.array([17.0 * math.sin(omega * t), 11.0 * math.sin(2 * omega * t + 0.9)])
        moving_free = free + np.stack([dx, dy], axis=1) * 1.0 + drift * fluidity
    else:
        moving_free = free

    contour = field['contour']
    if contour is None:
        return moving_free, None, None

    points = contour['points']
    normals = contour['normals']
    arcs = contour['arcs']
    depths = contour['depths']
    spacing = contour['spacing']

    # Travelling waves along the contour: neighbours slide together, so cells
    # grow and shrink instead of swapping places (no popping).
    slide = spacing * 0.55 * fluidity * (
        0.62 * np.sin(TAU * 5 * arcs + omega * t)
        + 0.38 * np.sin(TAU * 8 * arcs - 1.6 * omega * t + 2.1))
    pulse = 1.0 + 0.30 * fluidity * np.sin(TAU * 3 * arcs + omega * t * 1.0 + 0.7)

    tangent = np.stack([normals[:, 1], -normals[:, 0]], axis=1)
    anchor = points + tangent * slide[:, None] + field['base_word'] + word_offset(t)
    offset = normals * (depths * pulse)[:, None]
    return moving_free, anchor + offset, anchor - offset


def pixel_grid() -> np.ndarray:
    global PIXEL_XY
    if PIXEL_XY is None:
        gx, gy = np.meshgrid(np.arange(WIDTH, dtype=np.float32),
                             np.arange(HEIGHT, dtype=np.float32), indexing='xy')
        PIXEL_XY = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return PIXEL_XY


def rasterize(field: dict, t: float, fluidity: float = 1.0):
    free, inner, outer = sites_at(field, t, fluidity)
    parts = [free] + ([inner, outer] if inner is not None else [])
    sites = np.vstack(parts)
    _, labels = cKDTree(sites).query(pixel_grid(), workers=-1)
    labels = labels.reshape(HEIGHT, WIDTH)
    values = field['values'][labels]
    return labels, values.astype(np.uint8), sites


def composite(field: dict, values: np.ndarray, frames_dir: Path,
              equalize: bool = False) -> np.ndarray:
    copies = field['copies']
    slot_image = np.round(values.astype(np.float32) / 255.0 * (4 * copies - 1)).astype(int)
    out = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for slot in np.unique(slot_image):
        mask = slot_image == slot
        out[mask] = variant(int(slot) // copies, int(slot) % copies, frames_dir, equalize)[mask]
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def glyph_mask(field: dict, t: float) -> np.ndarray:
    glyph = field['glyph']
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    if glyph is None:
        return mask
    origin = field['base_word'] + word_offset(t)
    x0, y0 = int(round(origin[0])), int(round(origin[1]))
    h, w = glyph.shape
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(WIDTH, x0 + w), min(HEIGHT, y0 + h)
    if sx1 <= sx0 or sy1 <= sy0:
        return mask
    mask[sy0:sy1, sx0:sx1] = glyph[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    return mask


def score(field: dict, t: float, labels: np.ndarray, values: np.ndarray,
          preview: np.ndarray, preview_eq: np.ndarray | None = None) -> dict:
    contour = field['contour']
    result = {}
    luma = preview.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    if contour is not None:
        _, inner_sites, outer_sites = sites_at(field, t, 1.0)
        mid = 0.5 * (inner_sites + outer_sites)
        direction = inner_sites - outer_sites
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        norm[norm < 1e-6] = 1.0
        direction = direction / norm
        probe_in = mid + direction * 2.2
        probe_out = mid - direction * 2.2
        value_in = sample_nearest(values, probe_in).astype(int)
        value_out = sample_nearest(values, probe_out).astype(int)
        result['contour_coverage'] = float((value_in != value_out).mean())
        luma_in = sample_nearest(luma, probe_in)
        luma_out = sample_nearest(luma, probe_out)
        result['visible_coverage'] = float((np.abs(luma_in - luma_out) >= 18.0).mean())
    else:
        result['contour_coverage'] = 0.0
        result['visible_coverage'] = 0.0

    mask = glyph_mask(field, t)
    area = int(mask.sum())
    result['glyph_area'] = area
    if area:
        counts = np.bincount(labels[mask], minlength=labels.max() + 1)
        result['interior_pieces'] = int((counts >= 40).sum())
        result['stamp_ratio'] = float(counts.max() / area)
    else:
        result['interior_pieces'] = 0
        result['stamp_ratio'] = 1.0

    # slivers: pair cells smaller than a readable piece
    if contour is not None:
        total_counts = np.bincount(labels.ravel(), minlength=len(field['values']))
        pair = total_counts[len(field['free']):]
        result['sliver_frac'] = float((pair < 55).mean())
        result['median_pair_area'] = float(np.median(pair))
    else:
        result['sliver_frac'] = 0.0
        result['median_pair_area'] = 0.0

    def squint(image_luma):
        if not area:
            return 0.0
        ys, xs = np.nonzero(mask)
        y0, y1 = max(0, ys.min() - 60), min(HEIGHT, ys.max() + 60)
        x0, x1 = max(0, xs.min() - 60), min(WIDTH, xs.max() + 60)
        target = gaussian_filter(mask.astype(np.float32), 7.0)[y0:y1, x0:x1][::4, ::4]
        got = gaussian_filter(image_luma, 7.0)[y0:y1, x0:x1][::4, ::4]
        a = target.ravel() - target.mean()
        b = got.ravel() - got.mean()
        denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        return 0.0 if denom < 1e-6 else float((a * b).sum() / denom)

    result['squint'] = squint(luma)
    if preview_eq is not None:
        luma_eq = preview_eq.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722],
                                                           dtype=np.float32)
        result['squint_eq'] = squint(luma_eq)
    result['cells'] = int(len(field['values']))
    result['pairs'] = int(field['pair_count'])
    result['spacing'] = float(contour['spacing']) if contour is not None else 0.0
    result['repaired'] = int(field['repaired'])
    return result


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------

def edge_overlay(base: np.ndarray, labels: np.ndarray,
                 color=(255, 90, 40)) -> np.ndarray:
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    out = base.copy()
    if out.ndim == 2:
        out = np.repeat(out[:, :, None], 3, axis=2)
    out[edges] = np.array(color, dtype=np.uint8)
    return out


def ui_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = UI_FONT_BOLD if bold and UI_FONT_BOLD.exists() else UI_FONT
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def contact_sheet(tiles, columns, tile_width, title, row_labels=None,
                  column_labels=None, caption_lines=2, extras=None) -> Image.Image:
    tile_height = int(round(tile_width * HEIGHT / WIDTH))
    caption_height = 15 * caption_lines + 8
    gap, label_width, header = 10, (170 if row_labels else 0), 56
    rows = int(math.ceil(len(tiles) / columns))
    extras = extras or []
    extra_width = columns * tile_width + (columns - 1) * gap
    extra_tile_w = (extra_width - (len(extras) - 1) * gap) // max(len(extras), 1) if extras else 0
    extra_tile_h = int(round(extra_tile_w * HEIGHT / WIDTH)) if extras else 0
    extra_block = (extra_tile_h + caption_height + gap + 26) if extras else 0

    width = label_width + columns * tile_width + (columns - 1) * gap + 2 * gap
    height = header + rows * (tile_height + caption_height + gap) + gap + extra_block
    sheet = Image.new('RGB', (width, height), (18, 19, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), title, font=ui_font(21, True), fill=(240, 240, 240))

    if column_labels:
        for index, text in enumerate(column_labels):
            x = label_width + gap + index * (tile_width + gap)
            draw.text((x, 36), text, font=ui_font(14, True), fill=(150, 200, 255))

    for index, (image, caption) in enumerate(tiles):
        row, column = divmod(index, columns)
        x = label_width + gap + column * (tile_width + gap)
        y = header + row * (tile_height + caption_height + gap)
        thumb = Image.fromarray(image).resize((tile_width, tile_height), Image.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + tile_width - 1, y + tile_height - 1], outline=(70, 72, 78))
        for line_index, line in enumerate(caption.split('\n')[:caption_lines]):
            draw.text((x + 2, y + tile_height + 3 + line_index * 15), line,
                      font=ui_font(12, line_index == 0),
                      fill=(235, 235, 235) if line_index == 0 else (165, 168, 175))

    if row_labels:
        for row, text in enumerate(row_labels):
            y = header + row * (tile_height + caption_height + gap)
            for line_index, line in enumerate(text.split('\n')):
                draw.text((gap, y + 6 + line_index * 17), line,
                          font=ui_font(14, line_index == 0),
                          fill=(255, 220, 150) if line_index == 0 else (170, 172, 180))

    if extras:
        y = header + rows * (tile_height + caption_height + gap) + gap
        draw.text((label_width + gap, y), 'inspection panels', font=ui_font(15, True),
                  fill=(150, 200, 255))
        y += 24
        for index, (image, caption) in enumerate(extras):
            x = label_width + gap + index * (extra_tile_w + gap)
            thumb = Image.fromarray(image).resize((extra_tile_w, extra_tile_h), Image.LANCZOS)
            sheet.paste(thumb, (x, y))
            draw.rectangle([x, y, x + extra_tile_w - 1, y + extra_tile_h - 1],
                           outline=(70, 72, 78))
            for line_index, line in enumerate(caption.split('\n')[:caption_lines]):
                draw.text((x + 2, y + extra_tile_h + 3 + line_index * 15), line,
                          font=ui_font(12, line_index == 0),
                          fill=(235, 235, 235) if line_index == 0 else (165, 168, 175))
    return sheet


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

def task_matrix(args):
    out = args.out
    frames = args.frames
    densities = [12, 24, 48, 96, 200]
    copy_counts = [1, 4, 12]
    words = ['Yope3D', 'SpinStack', '3D Gravity Simulator']
    t = args.time

    tiles, mask_tiles, rows_labels, records = [], [], [], []
    for word in words:
        for copies in copy_counts:
            for density in densities:
                field = build_field(word, density, copies, seed=args.seed,
                                    depth=args.depth, repair=args.repair,
                                    frames_dir=frames)
                labels, values, _ = rasterize(field, t, args.fluidity)
                preview = composite(field, values, frames)
                preview_eq = composite(field, values, frames, equalize=True)
                metrics = score(field, t, labels, values, preview, preview_eq)
                metrics.update(word=word, density=density, copies=copies)
                records.append(metrics)
                caption = (f'sites {density}+{2 * metrics["pairs"]}  copies {copies}\n'
                           f'cc {metrics["contour_coverage"]:.2f} '
                           f'vc {metrics["visible_coverage"]:.2f} '
                           f'pc {metrics["interior_pieces"]} '
                           f'sq {metrics["squint"]:+.2f}')
                tiles.append((preview, caption))
                mask_tiles.append((values, caption))
                print(f'{word:22s} d={density:4d} c={copies:2d}  ' + json.dumps(
                    {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in metrics.items()
                     if k in ('contour_coverage', 'visible_coverage', 'interior_pieces',
                              'stamp_ratio', 'sliver_frac', 'squint', 'squint_eq', 'cells')}),
                      flush=True)
            rows_labels.append(f'{word}\ncopies {copies}')

    column_labels = [f'free sites {d}' for d in densities]

    # inspection panels
    field = build_field('Yope3D', args.panel_density, 4, seed=args.seed,
                        depth=args.depth, repair=args.repair, frames_dir=frames)
    labels, values, _ = rasterize(field, t, args.fluidity)
    preview = composite(field, values, frames)
    edges_mask = edge_overlay(values, labels)
    edges_preview = edge_overlay(preview, labels, (255, 255, 255))
    plain = build_field('Yope3D', args.panel_density + 2 * field['pair_count'], 4,
                        seed=args.seed, depth=args.depth, with_contour=False,
                        repair='none', frames_dir=frames)
    plain_labels, plain_values, _ = rasterize(plain, t, args.fluidity)
    plain_preview = composite(plain, plain_values, frames)

    sheet = contact_sheet(tiles, len(densities), 372,
                          'Agent D - contour-seeded Voronoi tessellation - composite preview '
                          f'(t={t:.0f}s, real clip stills; gravity clip is black footage)',
                          rows_labels, column_labels,
                          extras=[(edges_preview,
                                   'cell edges drawn (white)\nYope3D  free 48  copies 4'),
                                  (plain_preview,
                                   'NO contour seeds - plain organic field\n'
                                   'same machinery, seeds removed'),
                                  (composite(field, values, frames, equalize=True),
                                   'same mask, luminance-equalised footage\n'
                                   'isolates the tessellation from the black clips')])
    sheet.save(out / 'contact-sheet.png')

    mask_sheet = contact_sheet(mask_tiles, len(densities), 372,
                               'Agent D - raw grayscale selector masks (clip IDs) '
                               f'(t={t:.0f}s)', rows_labels, column_labels,
                               extras=[(edges_mask, 'mask with cell edges drawn\n'
                                                    'Yope3D  free 48  copies 4'),
                                       (np.repeat(plain_values[:, :, None], 3, axis=2),
                                        'NO contour seeds - plain organic mask\n'
                                        'letters absent, field unchanged'),
                                       (np.repeat(values[:, :, None], 3, axis=2),
                                        'mask, no overlay\nYope3D  free 48  copies 4')])
    mask_sheet.save(out / 'contact-sheet-mask.png')

    (out / 'metrics.json').write_text(json.dumps(records, indent=1))
    print(f'wrote {out / "contact-sheet.png"} and {out / "contact-sheet-mask.png"}')


def task_single(args):
    out, frames = args.out, args.frames
    field = build_field(args.word, args.density, args.copies, seed=args.seed,
                        depth=args.depth, repair=args.repair, frames_dir=frames,
                        spacing_override=args.spacing)
    labels, values, sites = rasterize(field, args.time, args.fluidity)
    preview = composite(field, values, frames)
    preview_eq = composite(field, values, frames, equalize=True)
    metrics = score(field, args.time, labels, values, preview, preview_eq)
    tag = args.tag or f'{args.word.replace(" ", "_")}-d{args.density}-c{args.copies}'
    Image.fromarray(values).save(out / f'single-{tag}-mask.png')
    Image.fromarray(preview).save(out / f'single-{tag}-composite.png')
    Image.fromarray(edge_overlay(preview, labels, (255, 255, 255))).save(
        out / f'single-{tag}-edges.png')
    Image.fromarray(preview_eq).save(out / f'single-{tag}-composite-eq.png')
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in metrics.items()}, indent=1))


def task_video(args):
    out, frames = args.out, args.frames
    field = build_field(args.word, args.density, args.copies, seed=args.seed,
                        depth=args.depth, repair=args.repair, frames_dir=frames,
                        spacing_override=args.spacing)
    tag = args.tag or f'{args.word.replace(" ", "_")}-d{args.density}-c{args.copies}'
    mask_path = out / f'loop-{tag}-mask.webm'
    preview_path = out / f'loop-{tag}-composite.mp4'
    mask_encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libvpx-vp9', '-lossless', '1', '-b:v', '0', '-cpu-used', '4',
        '-pix_fmt', 'yuv420p', str(mask_path)], stdin=subprocess.PIPE)
    preview_encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(preview_path)],
        stdin=subprocess.PIPE)

    frame_count = int(round(FPS * args.seconds))
    previous_counts = None
    max_jump, pops, samples = 0.0, 0, []
    try:
        for index in range(frame_count):
            t = index / FPS
            labels, values, _ = rasterize(field, t, args.fluidity)
            preview = composite(field, values, frames)
            mask_encoder.stdin.write(values.tobytes())
            preview_encoder.stdin.write(np.ascontiguousarray(preview).tobytes())
            counts = np.bincount(labels.ravel(), minlength=len(field['values'])).astype(float)
            if previous_counts is not None:
                live = (counts > 200) | (previous_counts > 200)
                if live.any():
                    jump = np.abs(counts[live] - previous_counts[live]) / np.maximum(
                        previous_counts[live], 1.0)
                    max_jump = max(max_jump, float(jump.max()))
                pops += int(((previous_counts > 150) & (counts == 0)).sum())
            previous_counts = counts
            if index % (FPS) == 0:
                metrics = score(field, t, labels, values, preview)
                samples.append({'t': t, **{k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in metrics.items()
                                           if k in ('contour_coverage', 'visible_coverage',
                                                    'interior_pieces', 'stamp_ratio',
                                                    'squint')}})
                Image.fromarray(values).save(out / f'loop-{tag}-{int(t)}s-mask.png')
                Image.fromarray(preview).save(out / f'loop-{tag}-{int(t)}s.png')
                print(f't={t:4.1f}s ' + json.dumps(samples[-1]), flush=True)
    finally:
        mask_encoder.stdin.close()
        preview_encoder.stdin.close()
    mask_encoder.wait()
    preview_encoder.wait()
    stability = {'max_relative_area_jump_per_frame': round(max_jump, 4),
                 'cells_that_vanished': pops, 'per_second': samples}
    (out / f'loop-{tag}-stability.json').write_text(json.dumps(stability, indent=1))
    print(json.dumps({k: v for k, v in stability.items() if k != 'per_second'}, indent=1))
    print(f'{mask_path} {mask_path.stat().st_size / 1024:.0f} KB')
    print(f'{preview_path} {preview_path.stat().st_size / 1024:.0f} KB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('matrix', 'single', 'video'), default='single')
    parser.add_argument('--word', default='Yope3D')
    parser.add_argument('--density', type=int, default=48, help='free (background) site count')
    parser.add_argument('--copies', type=int, default=4)
    parser.add_argument('--depth', type=float, default=9.0, help='pair offset h in pixels')
    parser.add_argument('--spacing', type=float, default=None,
                        help='contour sample spacing override in pixels')
    parser.add_argument('--fluidity', type=float, default=1.0)
    parser.add_argument('--time', type=float, default=0.0)
    parser.add_argument('--seconds', type=float, default=SECONDS)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--repair', choices=('none', 'source', 'contrast'), default='source')
    parser.add_argument('--panel-density', type=int, default=48)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--frames', type=Path, required=True,
                        help='directory holding clip0..clip3.png stills')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {'matrix': task_matrix, 'single': task_single, 'video': task_video}[args.task](args)


if __name__ == '__main__':
    main()
