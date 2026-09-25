#!/usr/bin/env python3
"""Agent C probe: make the project name legible through PIECE STATISTICS.

The showcase selector stores a clip ID per pixel; the name has to emerge from
the arrangement of clip regions alone.  Existing topologies either draw the
glyph as a thin sliver of seam (unreadable) or stamp the interior as one flat
ID (readable, but no longer a collage).

This probe takes a third route: keep the glyph interior a *full* collage of
many moving, wobbling pieces drawn from the *same shared palette* as the
background, and let figure/ground come from second-order statistics --
the texture properties a viewer reads at a glance:

  scale        finer mosaic inside the glyph than outside (spatial frequency)
  orientation  pieces cut into strips along the local stroke direction inside,
               a fixed oblique direction outside (equal piece count)
  motion       inside vertices calm and phase-locked, outside boiling
               incoherently (or inverted) -- only visible in motion
  contrast     per-seam clip-ID assignment chosen to maximise the REAL footage
               luminance difference across seams near the glyph contour, and
               optionally to minimise it elsewhere

Deliberately NOT used (other agents own those): a reserved sub-range of the
ID palette for letter pieces, a lattice retriangulated to follow the contour,
Voronoi/organic substrates.  Every polygon edge here is a lattice edge or a
straight cut inside a lattice cell; the glyph only decides *statistics*.

Subcommands:
    matrix   density x copies x word sweep -> contact-sheet.png + scores
    cues     one cue at a time at fixed density -> contact-sheet-cues.png
    fluidity fluidity sweep -> scores
    one      a single configuration, full-size stills
    video    12 s composite + lossless ID webm of one configuration
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path('/Users/me/Desktop/dev/yugumishra.github.io')
OUT_DIR = Path(
    '/private/tmp/claude-501/-Users-me-Desktop-dev-yugumishra-github-io'
    '/ea9e4c74-aeb0-4767-b13b-b99660eec2a7/scratchpad/agent-c'
)
CACHE = OUT_DIR / 'cache'
WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 30, 12
TAU = math.pi * 2
FONT_PATH = Path('/System/Library/Fonts/Supplemental/Arial Black.ttf')
UI_FONT_PATH = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
UI_FONT_BOLD = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')

# source order must match the compositor: 0 cloth, 1 spinstack, 2 digits, 3 gravity
CLIP_PATHS = [
    ROOT / 'public/media/yope_cloth_b.mp4',
    ROOT / 'public/media/showcase-test/spinstack.mp4',
    ROOT / 'public/media/showcase-test/digits.mp4',
    ROOT / 'public/media/showcase-test/gravity.mp4',
]
CLIP_NAMES = ['yope', 'spinstack', 'digits', 'gravity']
CLIP_SECONDS = [5.4, 6.666667, 5.0, 4.166667]

# copied verbatim from src/scripts/showcase-test.ts
COPY_TRANSFORMS = [
    (1.00, 0.00, 0.00), (1.12, -0.10, 0.08), (1.24, 0.12, -0.08),
    (1.36, -0.15, -0.12), (1.48, 0.08, 0.12), (1.60, -0.04, 0.02),
    (1.18, 0.16, 0.14), (1.30, -0.18, 0.04), (1.44, 0.05, -0.16),
    (1.56, 0.14, -0.02), (1.70, -0.12, 0.12), (1.84, 0.02, -0.12),
]
ORIGINAL_VERTICES = [
    [(0, 0), (290, 0), (581, 0), (986, 0), (1280, 0)],
    [(0, 184), (326, 223), (565, 241), (881, 231), (1280, 184)],
    [(0, 427), (307, 522), (576, 444), (982, 538), (1280, 490)],
    [(0, 720), (302, 720), (721, 720), (883, 720), (1280, 720)],
]


# ---------------------------------------------------------------- primitives

def js_round(value):
    return math.floor(value + 0.5)


def seeded(seed):
    value = math.sin(seed * 12.9898 + 78.233) * 43758.5453
    return value - math.floor(value)


def grid_for_density(target):
    aspect = WIDTH / HEIGHT
    best = (4, 3, float('inf'))
    for columns in range(2, 21):
        for rows in range(2, 15):
            score = abs(columns * rows - target) * 6 + abs(columns / rows - aspect)
            if score < best[2]:
                best = (columns, rows, score)
    return best[:2]


# ------------------------------------------------------------------- glyph

class Word:
    """Binary glyph raster plus a per-pixel stroke-orientation field."""

    def __init__(self, text, max_width=1010, max_height=330, wrap_below=120):
        lines = [text]
        size = self._fit(lines, max_width, max_height)
        if size < wrap_below and ' ' in text:
            words = text.split(' ')
            split = max(range(1, len(words)), key=lambda i: -abs(
                sum(len(w) for w in words[:i]) - sum(len(w) for w in words[i:])))
            lines = [' '.join(words[:split]), ' '.join(words[split:])]
            size = self._fit(lines, max_width, max_height)
        self.text, self.lines, self.size = text, lines, size
        font = ImageFont.truetype(str(FONT_PATH), size)
        boxes = [font.getbbox(line) for line in lines]
        widths = [b[2] - b[0] for b in boxes]
        line_height = max(b[3] - b[1] for b in boxes)
        gap = int(size * 0.14)
        w = max(widths)
        h = line_height * len(lines) + gap * (len(lines) - 1)
        image = Image.new('L', (w, h), 0)
        drawer = ImageDraw.Draw(image)
        for index, (line, box) in enumerate(zip(lines, boxes)):
            x = (w - (box[2] - box[0])) // 2 - box[0]
            drawer.text((x, index * (line_height + gap) - box[1]), line, font=font, fill=255)
        binary = (np.asarray(image) >= 128)
        self.width, self.height = w, h
        self.mask = binary
        # soft coverage: 3x supersample of the same raster, box-averaged
        big = Image.new('L', (w * 3, h * 3), 0)
        drawer = ImageDraw.Draw(big)
        font3 = ImageFont.truetype(str(FONT_PATH), size * 3)
        boxes3 = [font3.getbbox(line) for line in lines]
        lh3 = max(b[3] - b[1] for b in boxes3)
        for index, (line, box) in enumerate(zip(lines, boxes3)):
            x = (w * 3 - (box[2] - box[0])) // 2 - box[0]
            drawer.text((x, index * (lh3 + gap * 3) - box[1]), line, font=font3, fill=255)
        cover = np.asarray(big, dtype=np.float32)[:h * 3, :w * 3] / 255.0
        self.cover = cover.reshape(h, 3, w, 3).mean(axis=(1, 3))
        self.stroke = self._stroke_width()
        self.theta = self._orientation()

    @staticmethod
    def _fit(lines, max_width, max_height):
        size = 230
        for _ in range(60):
            font = ImageFont.truetype(str(FONT_PATH), size)
            boxes = [font.getbbox(line) for line in lines]
            w = max(b[2] - b[0] for b in boxes)
            lh = max(b[3] - b[1] for b in boxes)
            h = lh * len(lines) + int(size * 0.14) * (len(lines) - 1)
            if w <= max_width and h <= max_height:
                break
            size = int(size * min(max_width / w, max_height / h) * 0.98)
        return max(24, size)

    def _stroke_width(self):
        """Median stroke thickness: 2x the distance transform on the skeleton."""
        distance = ndimage.distance_transform_edt(self.mask)
        ridge = distance[distance > 0]
        if ridge.size == 0:
            return 1.0
        return float(2 * np.percentile(ridge, 85))

    def _orientation(self):
        """Per-pixel stroke direction from a heavily smoothed structure tensor."""
        smooth = ndimage.gaussian_filter(self.mask.astype(np.float32), 2.0)
        gy = ndimage.sobel(smooth, axis=0)
        gx = ndimage.sobel(smooth, axis=1)
        sigma = max(6.0, self.stroke * 0.7)
        jxx = ndimage.gaussian_filter(gx * gx, sigma)
        jyy = ndimage.gaussian_filter(gy * gy, sigma)
        jxy = ndimage.gaussian_filter(gx * gy, sigma)
        # dominant gradient orientation; the stroke runs perpendicular to it
        grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
        return (grad + math.pi / 2).astype(np.float32)

    def offset(self, time_seconds):
        phase = time_seconds / SECONDS * TAU
        travel_y = min(120.0, max(0.0, (HEIGHT - self.height) / 2 - 24))
        travel_x = min(125.0, max(0.0, (WIDTH - self.width) / 2 - 24))
        return (js_round((WIDTH - self.width) / 2 + travel_x * math.sin(phase)),
                js_round((HEIGHT - self.height) / 2 + travel_y * math.sin(phase * 2)))


def paste_field(array, word_array, ox, oy, fill=0.0):
    out = np.full((HEIGHT, WIDTH), fill, dtype=np.float32)
    h, w = word_array.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(WIDTH, ox + w), min(HEIGHT, oy + h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = word_array[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return out


# ------------------------------------------------------------------- config

@dataclass
class Cfg:
    word: str = 'Yope3D'
    density: int = 48
    fluidity: float = 60.0
    copies: int = 4
    scale: bool = False
    orient: bool = False
    motion: bool = False
    contrast: str = 'off'            # off | differ | contrast
    substrate: str = 'lattice'       # lattice | uniform-fine
    fine_px: float = 22.0
    fine_ratio: int = 0              # if > 0 use a fixed subdivision factor
    strips: int = 3
    motion_in_amp: float = 0.10
    motion_in_rate: float = 0.28
    motion_lock: float = 0.92
    motion_invert: bool = False
    suppress: float = 0.6
    label: str = ''

    def fine_factor(self, cell_w, cell_h):
        if self.fine_ratio:
            return max(2, self.fine_ratio)
        return int(min(26, max(2, round(min(cell_w, cell_h) / self.fine_px))))


# ------------------------------------------------------------------ lattice

def lattice_points(cfg, time_seconds, word, cover):
    """Base lattice, with per-vertex motion statistics modulated by membership."""
    piece_columns, piece_rows = grid_for_density(cfg.density)
    columns, rows = piece_columns, piece_rows
    coarse = []
    for coarse_row in range(piece_rows + 1):
        for coarse_column in range(piece_columns + 1):
            original = (ORIGINAL_VERTICES[coarse_row][coarse_column]
                        if piece_columns == 4 and piece_rows == 3 else None)
            coarse.append((original[0] if original else coarse_column * WIDTH / piece_columns,
                           original[1] if original else coarse_row * HEIGHT / piece_rows))
    amount = cfg.fluidity / 100.0
    cell_w, cell_h = WIDTH / columns, HEIGHT / rows
    amplitude = min(cell_w, cell_h) * 0.24 * amount
    motion = 0.55 + amount * 0.85
    points = []
    for row in range(rows + 1):
        line = []
        for column in range(columns + 1):
            base = coarse[row * (piece_columns + 1) + column]
            boundary = column in (0, columns) or row in (0, rows)
            if boundary or amplitude == 0:
                line.append(base)
                continue
            seed = row * 97 + column * 53 + columns * 11 + rows * 7
            member = membership(cover, base[0], base[1]) if cfg.motion else 0.0
            x, y = wobble(seed, time_seconds, motion, amplitude, member, cfg)
            line.append((base[0] + x, base[1] + y))
        points.append(line)
    return columns, rows, points


def wobble(seed, time_seconds, rate, amplitude, member, cfg):
    """The showcase's two-term vertex wobble, with statistics modulated by member."""
    phase_x = seeded(seed + 1) * TAU
    phase_y = seeded(seed + 2) * TAU
    speed_x = 0.72 + seeded(seed + 3) * 0.56
    speed_y = 0.72 + seeded(seed + 4) * 0.56
    if member > 0:
        target = member if not cfg.motion_invert else 0.0
        other = 0.0 if not cfg.motion_invert else member
        # inside: calm, slow, phase-locked. outside: untouched (or swapped)
        lock = cfg.motion_lock * (target - other + 1) * 0.5 if cfg.motion_invert else cfg.motion_lock * member
        lock = max(0.0, min(1.0, lock))
        gain = 1 + (cfg.motion_in_amp - 1) * member
        speed_gain = 1 + (cfg.motion_in_rate - 1) * member
        if cfg.motion_invert:
            gain = 1 + (1 / max(cfg.motion_in_amp, 0.05) - 1) * 0 + (1 - member) * 0
            gain = cfg.motion_in_amp + (1 - cfg.motion_in_amp) * member
            speed_gain = cfg.motion_in_rate + (1 - cfg.motion_in_rate) * member
            lock = cfg.motion_lock * (1 - member)
        phase_x *= (1 - lock)
        phase_y *= (1 - lock)
        speed_x += (1.0 - speed_x) * lock
        speed_y += (1.0 - speed_y) * lock
        amplitude *= gain
        rate *= speed_gain
    phase = time_seconds * rate
    wobble_x = (math.sin(phase * speed_x + phase_x) * 0.68
                + math.sin(phase * 0.61 + phase_y) * 0.32)
    wobble_y = (math.sin(phase * speed_y + phase_y) * 0.68
                + math.cos(phase * 0.57 + phase_x) * 0.32)
    return wobble_x * amplitude, wobble_y * amplitude


def membership(cover, x, y):
    xi = int(max(0, min(WIDTH - 1, x)))
    yi = int(max(0, min(HEIGHT - 1, y)))
    return float(cover[yi, xi])


def bilinear(quad, u, v):
    top_left, top_right, bottom_right, bottom_left = quad
    top = (top_left[0] + (top_right[0] - top_left[0]) * u,
           top_left[1] + (top_right[1] - top_left[1]) * u)
    bottom = (bottom_left[0] + (bottom_right[0] - bottom_left[0]) * u,
              bottom_left[1] + (bottom_right[1] - bottom_left[1]) * u)
    return (top[0] + (bottom[0] - top[0]) * v, top[1] + (bottom[1] - top[1]) * v)


def clip_half(polygon, nx, ny, c):
    """Sutherland-Hodgman half-plane clip, keeping nx*x + ny*y <= c."""
    out = []
    count = len(polygon)
    for index in range(count):
        a = polygon[index]
        b = polygon[(index + 1) % count]
        da = nx * a[0] + ny * a[1] - c
        db = nx * b[0] + ny * b[1] - c
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def strip_cut(quad, theta, count, seed, time_seconds, jitter):
    """Cut a convex quad into `count` parallel strips running along theta."""
    nx, ny = -math.sin(theta), math.cos(theta)
    projections = [nx * p[0] + ny * p[1] for p in quad]
    low, high = min(projections), max(projections)
    if high - low < 1e-6 or count < 2:
        return [quad]
    cuts = [low]
    for index in range(1, count):
        base = low + (high - low) * index / count
        wig = math.sin(time_seconds * 0.9 + seeded(seed + index * 7) * TAU)
        cuts.append(base + wig * jitter * (high - low) / count)
    cuts.append(high)
    cuts = sorted(cuts)
    strips = []
    for index in range(count):
        piece = clip_half(quad, nx, ny, cuts[index + 1])
        piece = clip_half(piece, -nx, -ny, -cuts[index])
        if len(piece) >= 3:
            strips.append(piece)
    return strips


# ------------------------------------------------------------- piece builder

def sub_node(quad, i, j, nu, nv, cell_seed, time_seconds, amplitude, cover, cfg):
    u, v = i / nu, j / nv
    x, y = bilinear(quad, u, v)
    if 0 < i < nu and 0 < j < nv and amplitude > 0:
        seed = (cell_seed * 31 + i * 17 + j * 53) % 9973
        member = membership(cover, x, y) if cfg.motion else 0.0
        dx, dy = wobble(seed, time_seconds, 0.55 + cfg.fluidity / 100 * 0.85,
                        amplitude, member, cfg)
        x, y = x + dx, y + dy
    return (x, y)


def sub_cover(quad, i, j, nu, nv, cover):
    total = 0.0
    for a in (0.17, 0.5, 0.83):
        for b in (0.17, 0.5, 0.83):
            x, y = bilinear(quad, (i + a) / nu, (j + b) / nv)
            total += membership(cover, x, y)
    return total / 9.0


def sub_theta(quad, i, j, nu, nv, theta_field):
    x, y = bilinear(quad, (i + 0.5) / nu, (j + 0.5) / nv)
    xi = int(max(0, min(WIDTH - 1, x)))
    yi = int(max(0, min(HEIGHT - 1, y)))
    return float(theta_field[yi, xi])


def build_pieces(cfg, time_seconds, word, cover, theta_field):
    """Return a painter-ordered list of pieces: (polygon, key, inside)."""
    columns, rows, points = lattice_points(cfg, time_seconds, word, cover)
    cell_w, cell_h = WIDTH / columns, HEIGHT / rows
    fine = cfg.fine_factor(cell_w, cell_h)
    amount = cfg.fluidity / 100.0
    uniform = cfg.substrate == 'uniform-fine'
    n_out = fine if uniform else 1
    n_in = fine if (cfg.scale or uniform) else n_out
    pieces = []

    def emit(quad, n, key_prefix, cell_seed, only_inside, oriented_inside):
        amplitude = min(cell_w / n, cell_h / n) * 0.35 * amount
        nodes = {}

        def node(i, j):
            if (i, j) not in nodes:
                nodes[(i, j)] = sub_node(quad, i, j, n, n, cell_seed,
                                         time_seconds, amplitude, cover, cfg)
            return nodes[(i, j)]

        for j in range(n):
            for i in range(n):
                member = sub_cover(quad, i, j, n, n, cover) if (only_inside or cfg.orient) else 0.0
                inside = member >= 0.5
                if only_inside and not inside:
                    continue
                cell = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
                key = (key_prefix, cell_seed, i, j)
                if cfg.orient:
                    if inside and oriented_inside:
                        theta = sub_theta(quad, i, j, n, n, theta_field)
                    else:
                        theta = math.pi / 4
                    seed = (cell_seed * 7 + i * 31 + j * 11) % 9973
                    for index, strip in enumerate(strip_cut(cell, theta, cfg.strips,
                                                            seed, time_seconds, 0.18 * amount)):
                        pieces.append((strip, key + (index,), inside))
                else:
                    pieces.append((cell, key, inside))

    for row in range(rows):
        for column in range(columns):
            quad = [points[row][column], points[row][column + 1],
                    points[row + 1][column + 1], points[row + 1][column]]
            cell_seed = row * 211 + column * 97
            emit(quad, n_out, 0, cell_seed, False, uniform)
            if n_in > n_out:
                emit(quad, n_in, 1, cell_seed, True, True)
    return pieces, columns, rows, fine


# ---------------------------------------------------------------- clip IDs

def default_slot(key, copies):
    digest = 0
    for value in key:
        digest = (digest * 131 + int(value) + 7) % 100003
    source = min(3, int(seeded(digest * 0.37 + 11.0) * 4))
    copy = digest % copies
    return source * copies + copy


def index_map(pieces, scale=2):
    """Rasterise piece indices (1-based) at reduced resolution."""
    w, h = WIDTH // scale, HEIGHT // scale
    image = Image.new('I', (w, h), 0)
    drawer = ImageDraw.Draw(image)
    for index, (polygon, _key, _inside) in enumerate(pieces):
        drawer.polygon([(x / scale, y / scale) for x, y in polygon], fill=index + 1)
    return np.asarray(image, dtype=np.int32), w, h


def adjacency(indices, boundary_weight_map):
    """Seam pairs with pixel length and mean glyph-contour proximity weight."""
    pairs = {}
    for shift_y, shift_x in ((0, 1), (1, 0)):
        a = indices[:indices.shape[0] - shift_y, :indices.shape[1] - shift_x]
        b = indices[shift_y:, shift_x:]
        w = boundary_weight_map[:boundary_weight_map.shape[0] - shift_y,
                                :boundary_weight_map.shape[1] - shift_x]
        differs = (a != b) & (a > 0) & (b > 0)
        for i, j, weight in zip(a[differs], b[differs], w[differs]):
            key = (int(i) - 1, int(j) - 1) if i < j else (int(j) - 1, int(i) - 1)
            entry = pairs.get(key)
            if entry is None:
                pairs[key] = [1, float(weight)]
            else:
                entry[0] += 1
                entry[1] += float(weight)
    return [(i, j, n, total / n) for (i, j), (n, total) in pairs.items()]


def assign_slots(cfg, pieces, slot_luma, indices, seams, n_pieces):
    """off: hash rule. differ: repair neighbours. contrast: seam-aware ICM."""
    slots = np.array([default_slot(key, cfg.copies) for _poly, key, _in in pieces],
                     dtype=np.int32)
    if cfg.contrast == 'off' or n_pieces == 0:
        return slots
    neighbours = [[] for _ in range(n_pieces)]
    for i, j, length, weight in seams:
        if i < n_pieces and j < n_pieces:
            neighbours[i].append((j, length, weight))
            neighbours[j].append((i, length, weight))
    if cfg.contrast == 'differ':
        for index in range(n_pieces):
            taken = {slots[j] for j, _l, _w in neighbours[index]}
            if slots[index] in taken:
                for step in range(1, cfg.copies * 4):
                    candidate = (slots[index] + step) % (4 * cfg.copies)
                    if candidate not in taken:
                        slots[index] = candidate
                        break
        return slots
    total_slots = 4 * cfg.copies
    usage = np.bincount(slots[:n_pieces], minlength=total_slots).astype(np.float64)
    quota = n_pieces / total_slots
    order = list(range(n_pieces))
    for sweep in range(3):
        order = order[::-1] if sweep % 2 else order
        for index in order:
            if not neighbours[index]:
                continue
            score = np.zeros(total_slots)
            for j, length, weight in neighbours[index]:
                other = slots[j]
                difference = np.abs(slot_luma[:, index] - slot_luma[other, j])
                # near the glyph contour maximise the real footage difference;
                # elsewhere (optionally) prefer a quiet seam.
                gain = weight - cfg.suppress * (1.0 - weight)
                score += gain * length * difference
                score[other] -= 4.0 * length * max(weight, 0.25)
            score -= 0.35 * np.maximum(0.0, usage - quota) / max(quota, 1.0) * \
                np.mean(np.abs(slot_luma[:, index] - slot_luma[:, index].mean()))
            best = int(np.argmax(score))
            if best != slots[index]:
                usage[slots[index]] -= 1
                usage[best] += 1
                slots[index] = best
    return slots


def render_mask(pieces, slots, copies):
    mask = Image.new('L', (WIDTH, HEIGHT), 0)
    drawer = ImageDraw.Draw(mask)
    denominator = 4 * copies - 1
    for (polygon, _key, _inside), slot in zip(pieces, slots):
        value = js_round(int(slot) * 255 / denominator)
        drawer.polygon(polygon, fill=value, outline=value, width=1)
    return mask


# ----------------------------------------------------------------- footage

_frame_cache = {}


def clip_frame(source, time_seconds):
    key = (source, round(time_seconds, 3))
    if key in _frame_cache:
        return _frame_cache[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{CLIP_NAMES[source]}-{round(time_seconds * 1000)}.png'
    if not path.exists():
        moment = time_seconds % CLIP_SECONDS[source]
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{moment:.3f}',
                        '-i', str(CLIP_PATHS[source]), '-frames:v', '1', str(path)],
                       check=True)
    image = Image.open(path).convert('RGB')
    _frame_cache[key] = image
    if len(_frame_cache) > 64:
        _frame_cache.pop(next(iter(_frame_cache)))
    return image


def variant_images(copies, time_seconds=2.0):
    """Full-screen variant rasters, matching drawFootageAtlas()'s cover-fit."""
    images = []
    for source in range(4):
        frame = clip_frame(source, time_seconds)
        vw, vh = frame.size
        for copy in range(copies):
            zoom, ox, oy = COPY_TRANSFORMS[(copy + source) % len(COPY_TRANSFORMS)]
            scale = max(WIDTH / vw, HEIGHT / vh) * zoom
            w, h = int(vw * scale), int(vh * scale)
            canvas = Image.new('RGB', (WIDTH, HEIGHT), (11, 13, 16))
            canvas.paste(frame.resize((w, h), Image.BILINEAR),
                         (int((WIDTH - w) / 2 + ox * WIDTH), int((HEIGHT - h) / 2 + oy * HEIGHT)))
            images.append(np.asarray(canvas, dtype=np.uint8))
    return np.stack(images)


def composite(mask, variants, copies):
    ids = np.rint(np.asarray(mask, dtype=np.float32) / 255 * (4 * copies - 1)).astype(np.int32)
    ids = np.clip(ids, 0, 4 * copies - 1)
    return variants[ids, np.arange(HEIGHT)[:, None], np.arange(WIDTH)[None, :]]


def luma(rgb):
    a = rgb.astype(np.float32) / 255.0
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


# ------------------------------------------------------------------ metrics

def edge_energy(gray):
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    return ndimage.uniform_filter(np.hypot(gx, gy), 9)


def squint(field, ideal, sigma=10.0, step=8):
    a = ndimage.gaussian_filter(field.astype(np.float32), sigma)[::step, ::step].ravel()
    b = ndimage.gaussian_filter(ideal.astype(np.float32), sigma)[::step, ::step].ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def dprime(field, inside, ring):
    a, b = field[inside], field[ring]
    if a.size < 10 or b.size < 10:
        return 0.0
    spread = math.sqrt(0.5 * (a.var() + b.var()))
    return 0.0 if spread < 1e-8 else float((a.mean() - b.mean()) / spread)


def glyph_regions(region):
    inside = region > 0.5
    ring = (ndimage.binary_dilation(inside, iterations=26) & ~
            ndimage.binary_dilation(inside, iterations=4))
    return inside, ring


def flat_fraction(mask, inside):
    ids = np.asarray(mask)[::2, ::2]
    small = inside[::2, ::2]
    if small.sum() == 0:
        return 0.0
    best = 0
    for value in np.unique(ids[small]):
        labelled, count = ndimage.label((ids == value) & small)
        if count:
            sizes = np.bincount(labelled.ravel())[1:]
            best = max(best, int(sizes.max()))
    return best / float(small.sum())


def palette_divergence(mask, inside, copies):
    ids = np.rint(np.asarray(mask, dtype=np.float32) / 255 * (4 * copies - 1)).astype(int)
    bins = 4 * copies
    a = np.bincount(ids[inside], minlength=bins).astype(float)
    b = np.bincount(ids[~inside], minlength=bins).astype(float)
    if a.sum() == 0 or b.sum() == 0:
        return 1.0
    a, b = a / a.sum(), b / b.sum()
    m = 0.5 * (a + b)

    def kl(p, q):
        keep = p > 0
        return float(np.sum(p[keep] * np.log2(p[keep] / q[keep])))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


# ---------------------------------------------------------------- rendering

_word_cache = {}


def get_word(text):
    if text not in _word_cache:
        _word_cache[text] = Word(text)
    return _word_cache[text]


def render(cfg, time_seconds=3.0, variants=None, want_composite=True, footage_time=2.0):
    word = get_word(cfg.word)
    ox, oy = word.offset(time_seconds)
    cover = paste_field(None, word.cover, ox, oy)
    theta_field = paste_field(None, word.theta, ox, oy, fill=math.pi / 2)
    region = paste_field(None, word.mask.astype(np.float32), ox, oy)
    pieces, columns, rows, fine = build_pieces(cfg, time_seconds, word, cover, theta_field)
    n_pieces = len(pieces)
    slot_luma = None
    seams = []
    if cfg.contrast != 'off':
        indices, w, h = index_map(pieces, scale=4)
        contour = np.abs(ndimage.gaussian_filter(region, 2.0)[::4, ::4] - 0.5)
        near = np.clip(1.0 - contour * 2.0, 0.0, 1.0)
        near = np.clip(ndimage.maximum_filter(near, 5), 0.0, 1.0)
        seams = adjacency(indices, near.astype(np.float32))
        if cfg.contrast == 'contrast':
            if variants is None:
                variants = variant_images(cfg.copies, footage_time)
            small = np.stack([ndimage.uniform_filter(luma(v), 5)[::4, ::4] for v in variants])
            flat = indices.ravel()
            counts = np.bincount(flat, minlength=n_pieces + 1).astype(np.float64)
            counts[counts == 0] = 1
            slot_luma = np.zeros((4 * cfg.copies, n_pieces))
            for s in range(4 * cfg.copies):
                sums = np.bincount(flat, weights=small[s].ravel(), minlength=n_pieces + 1)
                slot_luma[s] = (sums / counts)[1:]
        else:
            slot_luma = np.zeros((4 * cfg.copies, n_pieces))
    slots = assign_slots(cfg, pieces, slot_luma, None, seams, n_pieces)
    mask = render_mask(pieces, slots, cfg.copies)
    result = {'mask': mask, 'pieces': pieces, 'slots': slots, 'region': region,
              'columns': columns, 'rows': rows, 'fine': fine, 'word': word}
    if want_composite:
        if variants is None:
            variants = variant_images(cfg.copies, footage_time)
        result['composite'] = composite(mask, variants, cfg.copies)
        result['variants'] = variants
    return result


def score(cfg, result, temporal=None):
    region = result['region']
    inside, ring = glyph_regions(region)
    gray = luma(result['composite'])
    energy = edge_energy(gray)
    row = {
        'label': cfg.label or describe(cfg),
        'word': cfg.word, 'density': cfg.density, 'copies': cfg.copies,
        'fluidity': cfg.fluidity, 'cues': cue_string(cfg),
        'squint_L': abs(squint(gray, region)),
        'squint_E': abs(squint(energy, region)),
        'dprime_E': dprime(energy, inside, ring),
        'pieces': len(result['pieces']),
        'fine': result['fine'],
        'stroke_px': round(result['word'].stroke, 1),
        'flat_frac': flat_fraction(result['mask'], inside),
        'palette_js': palette_divergence(result['mask'], inside, cfg.copies),
    }
    row['LEG'] = max(row['squint_L'], row['squint_E'])
    if temporal is not None:
        row['squint_T'] = abs(squint(temporal, region))
        row['dprime_T'] = dprime(temporal, inside, ring)
    return row


def cue_string(cfg):
    tags = []
    if cfg.scale:
        tags.append('scale')
    if cfg.orient:
        tags.append('orient')
    if cfg.motion:
        tags.append('motion' + ('!' if cfg.motion_invert else ''))
    if cfg.contrast != 'off':
        tags.append(cfg.contrast)
    if cfg.substrate == 'uniform-fine':
        tags.append('uf')
    return '+'.join(tags) or 'none'


def describe(cfg):
    return f'{cfg.word} d{cfg.density} c{cfg.copies} f{int(cfg.fluidity)} [{cue_string(cfg)}]'


# ------------------------------------------------------------ contact sheets

def ui_font(size, bold=False):
    path = UI_FONT_BOLD if (bold and UI_FONT_BOLD.exists()) else UI_FONT_PATH
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def contact_sheet(tiles, columns, path, title, tile_w=320, caption_h=34, header_h=52):
    tile_h = int(tile_w * HEIGHT / WIDTH)
    rows = math.ceil(len(tiles) / columns)
    pad = 6
    sheet_w = columns * (tile_w + pad) + pad
    sheet_h = header_h + rows * (tile_h + caption_h + pad) + pad
    sheet = Image.new('RGB', (sheet_w, sheet_h), (16, 17, 20))
    drawer = ImageDraw.Draw(sheet)
    drawer.text((pad + 4, 14), title, font=ui_font(22, True), fill=(238, 238, 238))
    small = ui_font(13)
    tiny = ui_font(12)
    for index, tile in enumerate(tiles):
        image, caption, subcaption, accent = tile
        col, row = index % columns, index // columns
        x = pad + col * (tile_w + pad)
        y = header_h + row * (tile_h + caption_h + pad)
        sheet.paste(image.resize((tile_w, tile_h), Image.LANCZOS), (x, y))
        drawer.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=accent, width=2)
        drawer.rectangle([x, y + tile_h, x + tile_w - 1, y + tile_h + caption_h - 1],
                         fill=(30, 31, 36))
        drawer.text((x + 5, y + tile_h + 3), caption, font=small, fill=(240, 240, 240))
        drawer.text((x + 5, y + tile_h + 18), subcaption, font=tiny, fill=(165, 172, 185))
    sheet.save(path)
    return sheet


def accent_for(value):
    value = max(0.0, min(1.0, value / 0.6))
    return (int(200 - 160 * value), int(60 + 160 * value), int(70 + 60 * value))


def to_image(array):
    return Image.fromarray(array.astype(np.uint8))


def heat(field):
    f = field - field.min()
    f = f / (f.max() + 1e-9)
    f = np.clip(f, 0, 1)
    rgb = np.stack([np.clip(1.6 * f - 0.3, 0, 1), np.clip(1.4 * f - 0.15, 0, 1), f * 0.85], -1)
    return to_image(rgb * 255)


# -------------------------------------------------------------- experiments

BEST = dict(scale=True, orient=True, motion=True, contrast='contrast')


def temporal_activity(cfg, centre=3.0, frames=7, step=1 / 30, variants=None):
    """Mean absolute frame difference of the composite over a short window."""
    if variants is None:
        variants = variant_images(cfg.copies, 2.0)
    stack = []
    for index in range(frames):
        t = centre + (index - frames // 2) * step
        out = render(cfg, t, variants=variants, want_composite=True)
        stack.append(luma(out['composite']))
    diffs = [np.abs(stack[i + 1] - stack[i]) for i in range(len(stack) - 1)]
    return ndimage.uniform_filter(np.mean(diffs, axis=0), 11)


def run_matrix(args):
    words = ['Yope3D', 'SpinStack', '3D Gravity Simulator']
    densities = [12, 24, 48, 96]
    copy_counts = [1, 4, 12]
    rows = []
    tiles = []
    variant_cache = {}

    def variants_for(copies):
        if copies not in variant_cache:
            variant_cache[copies] = variant_images(copies, 2.0)
        return variant_cache[copies]

    # honest baseline strip: the plain lattice with no cue at all
    for density in densities:
        cfg = Cfg(word='Yope3D', density=density, copies=4, fluidity=60.0)
        out = render(cfg, 3.0, variants=variants_for(4))
        row = score(cfg, out)
        row['group'] = 'baseline'
        rows.append(row)
        tiles.append((to_image(out['composite']),
                      f'BASELINE  d{density} c4  Yope3D',
                      f"no cue - LEG {row['LEG']:.3f}  pieces {row['pieces']}",
                      (90, 92, 100)))
    for word in words:
        for density in densities:
            for copies in copy_counts:
                cfg = Cfg(word=word, density=density, copies=copies, fluidity=60.0, **BEST)
                out = render(cfg, 3.0, variants=variants_for(copies))
                row = score(cfg, out)
                row['group'] = 'matrix'
                rows.append(row)
                tiles.append((to_image(out['composite']),
                              f'{word[:22]}  d{density} c{copies}',
                              f"LEG {row['LEG']:.3f}  L {row['squint_L']:.2f} "
                              f"E {row['squint_E']:.2f}  n={row['pieces']} F={row['fine']}",
                              accent_for(row['LEG'])))
                print(f"  {row['label']}: LEG={row['LEG']:.3f} "
                      f"L={row['squint_L']:.3f} E={row['squint_E']:.3f} "
                      f"d'={row['dprime_E']:.2f} n={row['pieces']}", flush=True)
    contact_sheet(tiles, 5, OUT_DIR / 'contact-sheet.png',
                  'Agent C - figure/ground by piece statistics - composite previews '
                  '(row 1: no cue baseline; rest: scale+orient+motion+contrast-aware IDs, fluidity 60, t=3s)')
    write_scores(rows, 'scores-matrix')
    return rows


def cue_configs(density=48, copies=4, word='Yope3D', fluidity=60.0):
    return [
        ('baseline', Cfg(word=word, density=density, copies=copies, fluidity=fluidity)),
        ('scale only', Cfg(word=word, density=density, copies=copies, fluidity=fluidity,
                           scale=True)),
        ('orientation only', Cfg(word=word, density=density, copies=copies, fluidity=fluidity,
                                 orient=True, substrate='uniform-fine')),
        ('motion only', Cfg(word=word, density=density, copies=copies, fluidity=fluidity,
                            motion=True, substrate='uniform-fine')),
        ('contrast IDs only', Cfg(word=word, density=density, copies=copies, fluidity=fluidity,
                                  contrast='contrast', substrate='uniform-fine')),
        ('neighbour-differ only', Cfg(word=word, density=density, copies=copies,
                                      fluidity=fluidity, contrast='differ',
                                      substrate='uniform-fine')),
        ('scale + contrast', Cfg(word=word, density=density, copies=copies, fluidity=fluidity,
                                 scale=True, contrast='contrast')),
        ('all cues', Cfg(word=word, density=density, copies=copies, fluidity=fluidity, **BEST)),
    ]


def run_cues(args):
    variants = variant_images(4, 2.0)
    entries = cue_configs()
    rows = []
    composites, masks, energies, temporals = [], [], [], []
    for name, cfg in entries:
        cfg = replace(cfg, label=name)
        out = render(cfg, 3.0, variants=variants)
        activity = temporal_activity(cfg, 3.0, 7, 1 / 30, variants)
        row = score(cfg, out, temporal=activity)
        row['group'] = 'cues'
        rows.append(row)
        print(f"  {name}: LEG={row['LEG']:.3f} E={row['squint_E']:.3f} "
              f"L={row['squint_L']:.3f} T={row['squint_T']:.3f} "
              f"d'E={row['dprime_E']:.2f} d'T={row['dprime_T']:.2f} n={row['pieces']}",
              flush=True)
        accent = accent_for(max(row['LEG'], row['squint_T']))
        composites.append((to_image(out['composite']), name,
                           f"composite - LEG {row['LEG']:.3f}  n={row['pieces']}", accent))
        masks.append((out['mask'].convert('RGB'), name,
                      f"clip-ID field - flat {row['flat_frac']:.2f}  JS {row['palette_js']:.2f}",
                      (70, 72, 80)))
        energies.append((heat(edge_energy(luma(out['composite']))), name,
                         f"edge energy - squint_E {row['squint_E']:.3f} "
                         f"d' {row['dprime_E']:.2f}", (70, 72, 80)))
        temporals.append((heat(activity), name,
                          f"temporal activity - squint_T {row['squint_T']:.3f} "
                          f"d' {row['dprime_T']:.2f}", (70, 72, 80)))
    contact_sheet(composites + masks + energies + temporals, len(entries),
                  OUT_DIR / 'contact-sheet-cues.png',
                  'Agent C - cues isolated at density 48, copies 4, fluidity 60, "Yope3D" '
                  '(rows: composite / clip-ID field / edge-energy map / temporal-activity map)',
                  tile_w=300)
    write_scores(rows, 'scores-cues')
    return rows


def run_fluidity(args):
    variants = variant_images(4, 2.0)
    rows = []
    tiles = []
    for fluidity in (0.0, 60.0, 100.0):
        for name, base in (('all cues', Cfg(**BEST)), ('motion only',
                           Cfg(motion=True, substrate='uniform-fine'))):
            cfg = replace(base, word='Yope3D', density=48, copies=4,
                          fluidity=fluidity, label=f'{name} f{int(fluidity)}')
            out = render(cfg, 3.0, variants=variants)
            activity = temporal_activity(cfg, 3.0, 7, 1 / 30, variants)
            row = score(cfg, out, temporal=activity)
            row['group'] = 'fluidity'
            rows.append(row)
            print(f"  {cfg.label}: LEG={row['LEG']:.3f} T={row['squint_T']:.3f} "
                  f"d'T={row['dprime_T']:.2f}", flush=True)
            tiles.append((to_image(out['composite']), cfg.label,
                          f"LEG {row['LEG']:.3f}  squint_T {row['squint_T']:.3f}",
                          accent_for(row['LEG'])))
    contact_sheet(tiles, 3, OUT_DIR / 'contact-sheet-fluidity.png',
                  'Agent C - fluidity sweep, density 48, copies 4, "Yope3D"')
    write_scores(rows, 'scores-fluidity')
    return rows


def run_one(args):
    cfg = Cfg(word=args.word, density=args.density, copies=args.copies,
              fluidity=args.fluidity, scale=args.scale, orient=args.orient,
              motion=args.motion, contrast=args.contrast,
              substrate=args.substrate, fine_px=args.fine_px,
              fine_ratio=args.fine_ratio, motion_invert=args.motion_invert,
              suppress=args.suppress)
    out = render(cfg, args.time)
    stem = args.name or 'one'
    to_image(out['composite']).save(OUT_DIR / f'{stem}-composite.png')
    out['mask'].save(OUT_DIR / f'{stem}-mask.png')
    heat(edge_energy(luma(out['composite']))).save(OUT_DIR / f'{stem}-energy.png')
    row = score(cfg, out)
    print(json.dumps(row, indent=2))
    return [row]


def run_video(args):
    cfg = Cfg(word=args.word, density=args.density, copies=args.copies,
              fluidity=args.fluidity, scale=args.scale, orient=args.orient,
              motion=args.motion, contrast=args.contrast,
              substrate=args.substrate, motion_invert=args.motion_invert)
    stem = args.name or 'best'
    mask_path = OUT_DIR / f'{stem}-mask.webm'
    comp_path = OUT_DIR / f'{stem}-composite.webm'
    frames = max(1, int(round(FPS * args.seconds)))
    mask_encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libvpx-vp9', '-lossless', '1', '-b:v', '0', '-cpu-used', '4',
        '-pix_fmt', 'yuv420p', str(mask_path)], stdin=subprocess.PIPE)
    comp_encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libvpx-vp9', '-b:v', '3M', '-cpu-used', '5',
        '-pix_fmt', 'yuv420p', str(comp_path)], stdin=subprocess.PIPE)
    try:
        for frame in range(frames):
            t = frame / FPS
            variants = variant_images(cfg.copies, t)
            out = render(cfg, t, variants=variants, want_composite=True, footage_time=t)
            mask_encoder.stdin.write(out['mask'].tobytes())
            comp_encoder.stdin.write(np.ascontiguousarray(out['composite']).tobytes())
            if frame % 30 == 0:
                to_image(out['composite']).save(OUT_DIR / f'{stem}-{frame // 30}s.png')
                print(f'  frame {frame}/{frames}', flush=True)
    finally:
        mask_encoder.stdin.close()
        comp_encoder.stdin.close()
    mask_encoder.wait()
    comp_encoder.wait()
    print(f'{mask_path}: {mask_path.stat().st_size / 1024:.0f} KB')
    print(f'{comp_path}: {comp_path.stat().st_size / 1024:.0f} KB')
    return []


def write_scores(rows, stem):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with open(OUT_DIR / f'{stem}.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (OUT_DIR / f'{stem}.json').write_text(json.dumps(rows, indent=1))
    print(f'wrote {OUT_DIR / stem}.csv ({len(rows)} rows)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('matrix', 'cues', 'fluidity', 'one', 'video'))
    parser.add_argument('--word', default='Yope3D')
    parser.add_argument('--density', type=int, default=48)
    parser.add_argument('--copies', type=int, default=4)
    parser.add_argument('--fluidity', type=float, default=60.0)
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--motion-invert', action='store_true')
    parser.add_argument('--contrast', choices=('off', 'differ', 'contrast'), default='off')
    parser.add_argument('--substrate', choices=('lattice', 'uniform-fine'), default='lattice')
    parser.add_argument('--fine-px', type=float, default=22.0)
    parser.add_argument('--fine-ratio', type=int, default=0)
    parser.add_argument('--suppress', type=float, default=0.6)
    parser.add_argument('--time', type=float, default=3.0)
    parser.add_argument('--seconds', type=float, default=SECONDS)
    parser.add_argument('--name', default='')
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    {'matrix': run_matrix, 'cues': run_cues, 'fluidity': run_fluidity,
     'one': run_one, 'video': run_video}[args.command](args)


if __name__ == '__main__':
    main()
