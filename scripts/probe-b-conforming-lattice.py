"""Agent B probe: a contour-conforming moving lattice for the opening showcase.

The showcase mask is a grayscale clip-ID field.  The project name has to become
readable purely from the seams between neighbouring clip regions.  Earlier
topologies either produced thin teeth clinging to lattice edges (unreadable) or
stamped the whole glyph interior with one flat ID (readable, but the letters
stop being a collage).

This probe solves it as geometry.  Two variants are implemented:

``cut``  -- the recommended one.  The glyph boundary becomes a real shared edge
            of the moving lattice by taking an exact per-cell boolean: every
            lattice cell C is split into ``C inside-glyph`` and ``C
            outside-glyph`` and the two halves carry different clip IDs.  The
            lattice is refined locally so a glyph stroke is crossed by several
            whole cells, which keeps the letters a moving collage with internal
            seams.  Counters fall out of even/odd filling.  No pinned vertices,
            so flips and slivers are structurally impossible.

``snap``  -- the literal reading of "snap lattice vertices onto the contour":
            nearby lattice vertices are projected onto the glyph outline, slide
            tangentially as they wobble, and the whole point set (lattice +
            contour vertices) is re-triangulated with Delaunay.  Kept as an
            honest comparison; it is the variant that flips and drops strokes.

Nothing here writes into the site.  It only renders stills, contact sheets,
composite previews and a lossless VP9 probe video.

Examples:
    python scripts/probe-b-conforming-lattice.py single --density 48 --copies 4 \
        --fluidity 60 --out /tmp/b.png
    python scripts/probe-b-conforming-lattice.py matrix --out-dir /tmp/agent-b
    python scripts/probe-b-conforming-lattice.py video --density 48 --copies 4 \
        --fluidity 60 --out /tmp/agent-b/best.webm
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy.spatial import Delaunay, cKDTree

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 30, 12
FONT_PATH = Path('/System/Library/Fonts/Supplemental/Arial Black.ttf')
UI_FONT = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
UI_FONT_REGULAR = Path('/System/Library/Fonts/Supplemental/Arial.ttf')

CLIP_FRAMES = {
    0: 'clip0.png',   # Yope3D cloth
    1: 'clip1.png',   # SpinStack
    2: 'clip2.png',   # MNIST digits
    3: 'clip3.png',   # gravity
}
ATLAS_WIDTH, ATLAS_HEIGHT = 640, 360
COPY_TRANSFORMS = [
    (1.00, 0.00, 0.00), (1.12, -0.10, 0.08), (1.24, 0.12, -0.08),
    (1.36, -0.15, -0.12), (1.48, 0.08, 0.12), (1.60, -0.04, 0.02),
    (1.18, 0.16, 0.14), (1.30, -0.18, 0.04), (1.44, 0.05, -0.16),
    (1.56, 0.14, -0.02), (1.70, -0.12, 0.12), (1.84, 0.02, -0.12),
]

# The 4x3 hand-placed field the study starts from, kept so density 12 still
# reproduces the original piece arrangement.
ORIGINAL_VERTICES = [
    [(0, 0), (290, 0), (581, 0), (986, 0), (1280, 0)],
    [(0, 184), (326, 223), (565, 241), (881, 231), (1280, 184)],
    [(0, 427), (307, 522), (576, 444), (982, 538), (1280, 490)],
    [(0, 720), (302, 720), (721, 720), (883, 720), (1280, 720)],
]


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


# ---------------------------------------------------------------------------
# Word raster and straight-segment contours
# ---------------------------------------------------------------------------

def word_bitmap(word, max_width=1120, max_height=300, base_size=230):
    """Rasterise the word with Arial Black, shrinking long names to fit."""
    size = base_size
    for _ in range(40):
        font = ImageFont.truetype(str(FONT_PATH), size)
        bounds = font.getbbox(word)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if width <= max_width and height <= max_height:
            break
        size = max(24, int(size * min(max_width / width, max_height / height) * 0.98))
    font = ImageFont.truetype(str(FONT_PATH), size)
    bounds = font.getbbox(word)
    image = Image.new('L', (bounds[2] - bounds[0], bounds[3] - bounds[1]), 0)
    ImageDraw.Draw(image).text((-bounds[0], -bounds[1]), word, font=font, fill=255)
    # Hard labels: no intermediate gray may become a clip ID.
    return image.point(lambda value: 255 if value >= 128 else 0), size


def trace_contours(bitmap):
    """Marching-squares chains around the glyphs, in bitmap coordinates."""
    width, height = bitmap.size
    pixels = bitmap.load()

    def filled(x, y):
        return 0 <= x < width and 0 <= y < height and pixels[x, y] >= 128

    edge_points = {
        'top': lambda x, y: (2 * x + 1, 2 * y),
        'right': lambda x, y: (2 * x + 2, 2 * y + 1),
        'bottom': lambda x, y: (2 * x + 1, 2 * y + 2),
        'left': lambda x, y: (2 * x, 2 * y + 1),
    }
    cases = {
        1: [('top', 'left')], 2: [('top', 'right')], 3: [('left', 'right')],
        4: [('right', 'bottom')], 5: [('top', 'left'), ('right', 'bottom')],
        6: [('top', 'bottom')], 7: [('left', 'bottom')], 8: [('left', 'bottom')],
        9: [('top', 'bottom')], 10: [('top', 'right'), ('left', 'bottom')],
        11: [('right', 'bottom')], 12: [('left', 'right')],
        13: [('top', 'right')], 14: [('top', 'left')],
    }
    segments = []
    nodes = {}

    def add_segment(a, b):
        index = len(segments)
        segments.append((a, b))
        nodes.setdefault(a, []).append(index)
        nodes.setdefault(b, []).append(index)

    for y in range(-1, height):
        for x in range(-1, width):
            code = (
                (1 if filled(x, y) else 0)
                | (2 if filled(x + 1, y) else 0)
                | (4 if filled(x + 1, y + 1) else 0)
                | (8 if filled(x, y + 1) else 0)
            )
            for first, second in cases.get(code, []):
                add_segment(edge_points[first](x, y), edge_points[second](x, y))

    unused = set(range(len(segments)))
    chains = []
    while unused:
        first_index = next(iter(unused))
        start = segments[first_index][0]
        current, previous, chain = start, None, []
        while True:
            candidates = [index for index in nodes[current] if index in unused]
            if previous is not None and len(candidates) > 1:
                candidates = [index for index in candidates if index != previous] or candidates
            if not candidates:
                break
            index = candidates[0]
            unused.remove(index)
            a, b = segments[index]
            chain.append((current[0] / 2, current[1] / 2))
            current = b if a == current else a
            previous = index
            if current == start:
                break
        if current == start and len(chain) >= 3:
            chains.append(chain)
    return chains


def rdp(points, epsilon):
    """Ramer-Douglas-Peucker: straight segments only, corners preserved."""
    if len(points) < 3:
        return list(points)
    start, end = points[0], points[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy)
    worst_index, worst = 0, -1.0
    for index in range(1, len(points) - 1):
        point = points[index]
        if norm == 0:
            distance = math.hypot(point[0] - start[0], point[1] - start[1])
        else:
            distance = abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / norm
        if distance > worst:
            worst_index, worst = index, distance
    if worst <= epsilon:
        return [start, end]
    left = rdp(points[:worst_index + 1], epsilon)
    right = rdp(points[worst_index:], epsilon)
    return left[:-1] + right


def subdivide(contour, max_segment):
    result = []
    count = len(contour)
    for index, point in enumerate(contour):
        end = contour[(index + 1) % count]
        result.append(point)
        length = math.hypot(end[0] - point[0], end[1] - point[1])
        if length > max_segment:
            steps = int(math.ceil(length / max_segment))
            for step in range(1, steps):
                ratio = step / steps
                result.append((point[0] + (end[0] - point[0]) * ratio,
                               point[1] + (end[1] - point[1]) * ratio))
    return result


def word_contours(word, tolerance=1.1, max_segment=26.0):
    bitmap, size = word_bitmap(word)
    contours = []
    for chain in trace_contours(bitmap):
        simplified = rdp(chain + [chain[0]], tolerance)
        if simplified[0] == simplified[-1]:
            simplified = simplified[:-1]
        if len(simplified) < 3:
            continue
        contours.append(subdivide(simplified, max_segment))
    area = float(np.count_nonzero(np.asarray(bitmap) >= 128))
    perimeter = sum(
        math.hypot(c[(i + 1) % len(c)][0] - p[0], c[(i + 1) % len(c)][1] - p[1])
        for c in contours for i, p in enumerate(c)
    )
    stroke = 2 * area / perimeter if perimeter else 40.0
    return {
        'contours': contours,
        'bitmap': bitmap,
        'width': bitmap.width,
        'height': bitmap.height,
        'font_size': size,
        'stroke': stroke,
    }


def word_position(word_info, time_seconds):
    phase = time_seconds / SECONDS * math.tau
    amplitude_x = min(125.0, max(0.0, (WIDTH - word_info['width']) / 2 - 8))
    amplitude_y = min(120.0, max(0.0, (HEIGHT - word_info['height']) / 2 - 8))
    return (
        js_round((WIDTH - word_info['width']) / 2 + amplitude_x * math.sin(phase)),
        js_round((HEIGHT - word_info['height']) / 2 + amplitude_y * math.sin(phase * 2)),
    )


def glyph_stencil(word_info, word_x, word_y):
    """Even/odd filled straight-segment word region; counters stay holes."""
    region = Image.new('1', (WIDTH, HEIGHT), 0)
    for contour in word_info['contours']:
        loop = Image.new('1', (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(loop).polygon([(word_x + x, word_y + y) for x, y in contour], fill=1)
        region = ImageChops.logical_xor(region, loop)
    return region


# ---------------------------------------------------------------------------
# The moving lattice
# ---------------------------------------------------------------------------

def refinement_factor(piece_columns, piece_rows, stroke, cap=8):
    """How finely to split coarse pieces so a stroke is crossed by whole cells."""
    coarse = min(WIDTH / piece_columns, HEIGHT / piece_rows)
    return int(max(1, min(cap, round(coarse / max(8.0, stroke * 1.2)))))


class Lattice:
    """A fine quad lattice whose shared corners wobble on continuous paths.

    Clip IDs are assigned per coarse block outside the word, so the visible
    background piece count still matches `density`; inside the word every fine
    cell gets its own ID, which is what keeps the letters a collage.
    """

    def __init__(self, density, fluidity, copies, stroke, refine=None, background='four'):
        self.piece_columns, self.piece_rows = grid_for_density(density)
        self.refine = refinement_factor(self.piece_columns, self.piece_rows, stroke) if refine is None else max(1, refine)
        self.columns = self.piece_columns * self.refine
        self.rows = self.piece_rows * self.refine
        self.copies = copies
        self.fluidity = fluidity
        self.background = background
        self.cell_width = WIDTH / self.columns
        self.cell_height = HEIGHT / self.rows
        self.amount = fluidity / 100.0
        self.amplitude = min(self.cell_width, self.cell_height) * 0.24 * self.amount
        self.motion = 0.55 + self.amount * 0.85
        self.fracture = min(self.cell_width, self.cell_height) * 0.18 * self.amount

        coarse_positions = []
        for coarse_row in range(self.piece_rows + 1):
            for coarse_column in range(self.piece_columns + 1):
                original = (ORIGINAL_VERTICES[coarse_row][coarse_column]
                            if self.piece_columns == 4 and self.piece_rows == 3 else None)
                coarse_positions.append((
                    original[0] if original else coarse_column * WIDTH / self.piece_columns,
                    original[1] if original else coarse_row * HEIGHT / self.piece_rows,
                ))

        def coarse_position(column, row):
            return coarse_positions[row * (self.piece_columns + 1) + column]

        self.vertices = []
        for row in range(self.rows + 1):
            for column in range(self.columns + 1):
                seed = row * 97 + column * 53 + self.columns * 11 + self.rows * 7
                coarse_x = column / self.columns * self.piece_columns
                coarse_y = row / self.rows * self.piece_rows
                left = min(self.piece_columns - 1, math.floor(coarse_x))
                top = min(self.piece_rows - 1, math.floor(coarse_y))
                fx, fy = coarse_x - left, coarse_y - top
                tl, tr = coarse_position(left, top), coarse_position(left + 1, top)
                bl, br = coarse_position(left, top + 1), coarse_position(left + 1, top + 1)
                self.vertices.append({
                    'x': tl[0] * (1 - fx) * (1 - fy) + tr[0] * fx * (1 - fy) + bl[0] * (1 - fx) * fy + br[0] * fx * fy,
                    'y': tl[1] * (1 - fx) * (1 - fy) + tr[1] * fx * (1 - fy) + bl[1] * (1 - fx) * fy + br[1] * fx * fy,
                    'phase_x': seeded(seed + 1) * math.tau,
                    'phase_y': seeded(seed + 2) * math.tau,
                    'speed_x': 0.72 + seeded(seed + 3) * 0.56,
                    'speed_y': 0.72 + seeded(seed + 4) * 0.56,
                })

        self.slot_count = 4 * copies
        self._outside = {}
        self._inside = {}
        for row in range(self.rows):
            for column in range(self.columns):
                self._outside[(row, column)] = self._outside_slot(row, column)
        for row in range(self.rows):
            for column in range(self.columns):
                self._inside[(row, column)] = self._inside_slot(row, column)

    # -- clip ID assignment -------------------------------------------------

    def _outside_source(self, row, column):
        coarse_row, coarse_column = row // self.refine, column // self.refine
        if self.background == 'checker':
            return (coarse_row + coarse_column) % 2
        # All four projects in the base field; this is a mix, not a reserved
        # partition: every source is available to the letters as well.
        return (coarse_column * 2 + coarse_row * 3 + ((coarse_column * coarse_row) % 3)) % 4

    def _outside_slot(self, row, column):
        coarse_row, coarse_column = row // self.refine, column // self.refine
        source = self._outside_source(row, column)
        copy = (coarse_column * 3 + coarse_row * 5 + (coarse_column ^ coarse_row)) % self.copies
        return source * self.copies + copy

    def _inside_slot(self, row, column):
        """Local conflict rule: the ID must actually change across the seam.

        Forbidden = the cell's own outside source plus the outside sources of
        the four neighbours it can touch when the contour runs along a lattice
        edge.  Falls back to the self-constraint when the neighbourhood already
        uses every source.
        """
        own = self._outside_source(row, column)
        forbidden = {own}
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, column + dc
            if 0 <= r < self.rows and 0 <= c < self.columns:
                forbidden.add(self._outside_source(r, c))
        candidates = [s for s in range(4) if s not in forbidden] or [s for s in range(4) if s != own]
        pick = candidates[(row * 7 + column * 11) % len(candidates)]
        copy = (row * 5 + column * 3 + ((row * column) % 7)) % self.copies
        return pick * self.copies + copy

    def value(self, slot):
        return js_round(slot * 255 / (self.slot_count - 1)) if self.slot_count > 1 else 0

    # -- geometry -----------------------------------------------------------

    def positions(self, time_seconds):
        points = []
        stride = self.columns + 1
        for index, vertex in enumerate(self.vertices):
            column, row = index % stride, index // stride
            if column in (0, self.columns) or row in (0, self.rows) or self.amplitude == 0:
                points.append((vertex['x'], vertex['y']))
                continue
            phase = time_seconds * self.motion
            wobble_x = (math.sin(phase * vertex['speed_x'] + vertex['phase_x']) * 0.68
                        + math.sin(phase * 0.61 + vertex['phase_y']) * 0.32)
            wobble_y = (math.sin(phase * vertex['speed_y'] + vertex['phase_y']) * 0.68
                        + math.cos(phase * 0.57 + vertex['phase_x']) * 0.32)
            points.append((vertex['x'] + wobble_x * self.amplitude,
                           vertex['y'] + wobble_y * self.amplitude))
        return points

    def edge_paths(self, points, time_seconds):
        """Fractured straight-segment chains, shared by both adjacent cells."""
        stride = self.columns + 1
        horizontal, vertical = {}, {}
        for row in range(self.rows + 1):
            for column in range(self.columns):
                key = row * 4093 + column * 131 + 17
                horizontal[(row, column)] = self._fracture(
                    points[row * stride + column], points[row * stride + column + 1], key, time_seconds)
        for row in range(self.rows):
            for column in range(self.columns + 1):
                key = row * 3571 + column * 197 + 8191
                vertical[(row, column)] = self._fracture(
                    points[row * stride + column], points[(row + 1) * stride + column], key, time_seconds)
        return horizontal, vertical

    def _fracture(self, start, end, key, time_seconds):
        if self.fracture <= 0.01:
            return [start, end]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 2:
            return [start, end]
        breaks = 1 + int(seeded(key) * 3)
        limit = min(self.fracture, length * 0.2)
        path = [start]
        for index in range(breaks):
            jitter = seeded(key * 7 + index * 13 + 3)
            u = (index + 0.5) / breaks + (jitter - 0.5) * 0.6 / breaks
            phase = seeded(key * 11 + index * 29 + 5) * math.tau
            speed = 0.5 + seeded(key * 17 + index * 31 + 9) * 0.9
            offset = math.sin(time_seconds * self.motion * speed + phase) * limit
            path.append((start[0] + dx * u - dy / length * offset,
                         start[1] + dy * u + dx / length * offset))
        path.append(end)
        return path

    def cell_polygon(self, row, column, horizontal, vertical):
        top = horizontal[(row, column)]
        right = vertical[(row, column + 1)]
        bottom = horizontal[(row + 1, column)]
        left = vertical[(row, column)]
        return top[:-1] + right[:-1] + list(reversed(bottom))[:-1] + list(reversed(left))[:-1]


def polygon_area(polygon):
    total = 0.0
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return total / 2


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

def make_frame_cut(lattice, word_info, time_seconds, want_geometry=False):
    """Exact per-cell boolean against the glyph region.

    ``outside`` and ``inside`` are the same lattice drawn with two different ID
    assignments; the glyph stencil selects between them.  Every seam in the
    result is therefore either a lattice edge or the glyph outline, and the
    glyph outline is a shared edge between two real faces.
    """
    points = lattice.positions(time_seconds)
    horizontal, vertical = lattice.edge_paths(points, time_seconds)
    word_x, word_y = word_position(word_info, time_seconds)

    outside = Image.new('L', (WIDTH, HEIGHT), 0)
    inside = Image.new('L', (WIDTH, HEIGHT), 0)
    outside_draw, inside_draw = ImageDraw.Draw(outside), ImageDraw.Draw(inside)
    polygons = []
    for row in range(lattice.rows):
        for column in range(lattice.columns):
            polygon = lattice.cell_polygon(row, column, horizontal, vertical)
            outside_draw.polygon(polygon, fill=lattice.value(lattice._outside[(row, column)]))
            inside_draw.polygon(polygon, fill=lattice.value(lattice._inside[(row, column)]))
            if want_geometry:
                polygons.append(polygon)

    stencil = glyph_stencil(word_info, word_x, word_y)
    mask = outside
    mask.paste(inside, mask=stencil)
    if want_geometry:
        return mask, stencil, polygons, (word_x, word_y)
    return mask, stencil


def make_frame_snap(lattice, word_info, time_seconds, want_geometry=False):
    """Snap nearby lattice vertices onto the outline, then re-triangulate.

    Snapped vertices keep boiling by sliding tangentially along the contour
    (arc-length offset) while being pinned normal to it.
    """
    points = lattice.positions(time_seconds)
    word_x, word_y = word_position(word_info, time_seconds)

    dense, arclen, owner = [], [], []
    contour_tables = []
    for index, contour in enumerate(word_info['contours']):
        placed = [(word_x + x, word_y + y) for x, y in contour]
        lengths = [0.0]
        for i in range(len(placed)):
            a, b = placed[i], placed[(i + 1) % len(placed)]
            lengths.append(lengths[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        contour_tables.append((placed, lengths))
        total = lengths[-1]
        steps = max(8, int(total / 3))
        for step in range(steps):
            s = step * total / steps
            dense.append(evaluate_arclen(placed, lengths, s))
            arclen.append(s)
            owner.append(index)

    tree = cKDTree(np.asarray(dense))
    radius = min(lattice.cell_width, lattice.cell_height) * 0.5
    snapped = list(points)
    stride = lattice.columns + 1
    for index, point in enumerate(points):
        column, row = index % stride, index // stride
        if column in (0, lattice.columns) or row in (0, lattice.rows):
            continue
        distance, nearest = tree.query(point)
        if distance > radius:
            continue
        contour_index = owner[nearest]
        placed, lengths = contour_tables[contour_index]
        slide = math.sin(time_seconds * lattice.motion * (0.6 + seeded(index) * 0.9)
                         + seeded(index + 7) * math.tau) * lattice.amplitude * 1.6
        snapped[index] = evaluate_arclen(placed, lengths, (arclen[nearest] + slide) % lengths[-1])

    vertices = list(snapped)
    for placed, _ in contour_tables:
        vertices.extend(placed)
    mesh = Delaunay(np.asarray(vertices))
    stencil = glyph_stencil(word_info, word_x, word_y)
    stencil_array = np.asarray(stencil)

    mask = Image.new('L', (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    polygons = []
    for simplex in mesh.simplices:
        triangle = [vertices[i] for i in simplex]
        cx = sum(p[0] for p in triangle) / 3
        cy = sum(p[1] for p in triangle) / 3
        sx = max(0, min(WIDTH - 1, int(cx)))
        sy = max(0, min(HEIGHT - 1, int(cy)))
        column = max(0, min(lattice.columns - 1, int(cx / lattice.cell_width)))
        row = max(0, min(lattice.rows - 1, int(cy / lattice.cell_height)))
        table = lattice._inside if stencil_array[sy, sx] else lattice._outside
        draw.polygon(triangle, fill=lattice.value(table[(row, column)]))
        if want_geometry:
            polygons.append(triangle)
    if want_geometry:
        return mask, stencil, polygons, (word_x, word_y)
    return mask, stencil


def evaluate_arclen(placed, lengths, target):
    total = lengths[-1]
    target = target % total if total else 0
    lo, hi = 0, len(lengths) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if lengths[mid] <= target:
            lo = mid
        else:
            hi = mid
    a, b = placed[lo], placed[(lo + 1) % len(placed)]
    span = lengths[lo + 1] - lengths[lo]
    ratio = 0 if span == 0 else (target - lengths[lo]) / span
    return (a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio)


def make_frame(variant, lattice, word_info, time_seconds, want_geometry=False):
    if variant == 'snap':
        return make_frame_snap(lattice, word_info, time_seconds, want_geometry)
    return make_frame_cut(lattice, word_info, time_seconds, want_geometry)


# ---------------------------------------------------------------------------
# Composite preview
# ---------------------------------------------------------------------------

_TILE_CACHE = {}


def clip_tiles(frames_dir, copies):
    key = copies
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]
    tiles = []
    for source in range(4):
        image = Image.open(Path(frames_dir) / CLIP_FRAMES[source]).convert('RGB')
        for copy in range(copies):
            zoom, ox, oy = COPY_TRANSFORMS[(copy + source) % len(COPY_TRANSFORMS)]
            scale = max(ATLAS_WIDTH / image.width, ATLAS_HEIGHT / image.height) * zoom
            resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
            tile = Image.new('RGB', (ATLAS_WIDTH, ATLAS_HEIGHT), (11, 13, 16))
            tile.paste(resized, (int((ATLAS_WIDTH - resized.width) / 2 + ox * ATLAS_WIDTH),
                                 int((ATLAS_HEIGHT - resized.height) / 2 + oy * ATLAS_HEIGHT)))
            tiles.append(np.asarray(tile.resize((WIDTH, HEIGHT), Image.BILINEAR)))
    stack = np.stack(tiles)
    _TILE_CACHE[key] = stack
    return stack


def composite(mask, copies, frames_dir):
    stack = clip_tiles(frames_dir, copies)
    slot_count = 4 * copies
    gray = np.asarray(mask).astype(np.float32)
    slots = np.clip(np.rint(gray * (slot_count - 1) / 255.0).astype(np.int32), 0, slot_count - 1)
    ys, xs = np.indices((HEIGHT, WIDTH))
    return Image.fromarray(stack[slots, ys, xs])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def contour_samples(word_info, word_x, word_y, step=3.0):
    points, normals = [], []
    for contour in word_info['contours']:
        count = len(contour)
        for index in range(count):
            ax, ay = contour[index]
            bx, by = contour[(index + 1) % count]
            length = math.hypot(bx - ax, by - ay)
            if length < 1e-6:
                continue
            nx, ny = -(by - ay) / length, (bx - ax) / length
            steps = max(1, int(length / step))
            for s in range(steps):
                ratio = (s + 0.5) / steps
                points.append((word_x + ax + (bx - ax) * ratio, word_y + ay + (by - ay) * ratio))
                normals.append((nx, ny))
    return np.asarray(points), np.asarray(normals)


def sample_at(array, xs, ys):
    xi = np.clip(np.rint(xs).astype(np.int32), 0, WIDTH - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, HEIGHT - 1)
    return array[yi, xi]


def score_frame(mask, stencil, preview, word_info, word_x, word_y, offset=3.0):
    gray = np.asarray(mask)
    region = np.asarray(stencil)
    luma = np.asarray(preview.convert('L')).astype(np.float32)
    points, normals = contour_samples(word_info, word_x, word_y)
    if len(points) == 0:
        return {}
    ax = points[:, 0] + normals[:, 0] * offset
    ay = points[:, 1] + normals[:, 1] * offset
    bx = points[:, 0] - normals[:, 0] * offset
    by = points[:, 1] - normals[:, 1] * offset
    id_a, id_b = sample_at(gray, ax, ay), sample_at(gray, bx, by)
    in_a, in_b = sample_at(region, ax, ay), sample_at(region, bx, by)
    valid = in_a != in_b                      # only where the probe really straddles
    id_change = float(np.mean((id_a != id_b)[valid])) if valid.any() else 0.0
    luma_a, luma_b = sample_at(luma, ax, ay), sample_at(luma, bx, by)
    visible = float(np.mean((np.abs(luma_a - luma_b) >= 12)[valid])) if valid.any() else 0.0

    # Interior pieces: connected components of constant ID inside the glyph.
    # Normalised by the number of glyph blobs, so "one flat ID per letter"
    # (a stamp) scores 1.0 and cannot be confused with a real collage.
    from scipy import ndimage
    pieces, slivers = 0, 0
    for value in np.unique(gray[region]):
        component, count = ndimage.label((gray == value) & region)
        if count == 0:
            continue
        sizes = ndimage.sum(np.ones_like(component), component, range(1, count + 1))
        pieces += int(np.count_nonzero(sizes >= 250))
        slivers += int(np.count_nonzero(sizes < 60))
    _, blob_count = ndimage.label(region)
    per_letter = pieces / max(1, blob_count)

    # Squint test.  Two readings, because the effect is an outline effect:
    #   fill  - does the glyph read as one brightness blob after blurring?
    #   edge  - do the composite's visible seams trace the glyph outline?
    def correlate(a_image, b_image):
        a = np.asarray(a_image, dtype=np.float32).ravel().copy()
        b = np.asarray(b_image, dtype=np.float32).ravel().copy()
        a -= a.mean()
        b -= b.mean()
        denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        return abs(float((a * b).sum() / denominator)) if denominator else 0.0

    def squash(array, blur=7):
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(blur)).resize((160, 90), Image.BILINEAR)

    region_u8 = (region * 255).astype(np.uint8)
    squint_fill = correlate(squash(luma), squash(region_u8))

    gy, gx = np.gradient(luma)
    composite_edges = np.hypot(gx, gy)
    ry, rx = np.gradient(region.astype(np.float32) * 255.0)
    glyph_edges = np.hypot(rx, ry)
    squint_edge = correlate(squash(composite_edges * 4.0, 9), squash(glyph_edges, 9))

    interior = min(1.0, pieces / 10.0)
    total = 0.35 * visible + 0.20 * id_change + 0.30 * squint_edge + 0.15 * interior
    return {
        'id_change': round(id_change, 3),
        'visible': round(visible, 3),
        'pieces': pieces,
        'slivers': slivers,
        'squint': round(squint_edge, 3),
        'squint_fill': round(squint_fill, 3),
        'score': round(total, 3),
    }


def flip_report(lattice, samples=40):
    worst, inverted = 1.0, 0
    for index in range(samples):
        time_seconds = index * SECONDS / samples
        points = lattice.positions(time_seconds)
        horizontal, vertical = lattice.edge_paths(points, time_seconds)
        nominal = lattice.cell_width * lattice.cell_height
        for row in range(lattice.rows):
            for column in range(lattice.columns):
                area = polygon_area(lattice.cell_polygon(row, column, horizontal, vertical))
                ratio = area / nominal
                worst = min(worst, ratio)
                if ratio <= 0:
                    inverted += 1
    return {'min_area_ratio': round(worst, 3), 'inverted_faces': inverted}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def wireframe(lattice, word_info, time_seconds, variant='cut'):
    """Inspectable topology: fine cells, visible-piece boundaries, glyph edge."""
    _, stencil, polygons, (word_x, word_y) = make_frame(variant, lattice, word_info, time_seconds, True)
    image = Image.new('RGB', (WIDTH, HEIGHT), (252, 252, 250))
    # Tint the glyph region so the per-cell boolean (cell-inside vs
    # cell-outside faces sharing the contour) is visible.
    tint = Image.new('RGB', (WIDTH, HEIGHT), (255, 232, 232))
    image.paste(tint, mask=stencil)
    draw = ImageDraw.Draw(image)

    if variant == 'cut':
        points = lattice.positions(time_seconds)
        horizontal, vertical = lattice.edge_paths(points, time_seconds)
        for row in range(lattice.rows):
            for column in range(lattice.columns):
                polygon = lattice.cell_polygon(row, column, horizontal, vertical)
                draw.line(list(polygon) + [polygon[0]], fill=(158, 186, 214), width=1)
        # Boundaries where the visible background piece actually changes.
        for row in range(lattice.rows):
            for column in range(lattice.columns):
                here = lattice._outside[(row, column)]
                if column + 1 < lattice.columns and lattice._outside[(row, column + 1)] != here:
                    draw.line(vertical[(row, column + 1)], fill=(40, 72, 122), width=3)
                if row + 1 < lattice.rows and lattice._outside[(row + 1, column)] != here:
                    draw.line(horizontal[(row + 1, column)], fill=(40, 72, 122), width=3)
    else:
        for polygon in polygons:
            draw.line(list(polygon) + [polygon[0]], fill=(158, 186, 214), width=1)

    for contour in word_info['contours']:
        placed = [(word_x + x, word_y + y) for x, y in contour]
        draw.line(placed + [placed[0]], fill=(214, 40, 40), width=3)
        for point in placed:
            draw.ellipse([point[0] - 2.4, point[1] - 2.4, point[0] + 2.4, point[1] + 2.4], fill=(214, 40, 40))
    legend = ImageDraw.Draw(image)
    legend.rectangle([8, 8, 690, 78], fill=(255, 255, 255), outline=(180, 180, 180))
    legend.text((16, 14), 'thin blue = fine lattice cell   thick blue = visible piece boundary',
                font=caption_font(17, False), fill=(40, 46, 56))
    legend.text((16, 44), 'red = glyph contour, a shared edge between two faces of the same cell',
                font=caption_font(17, False), fill=(150, 30, 30))
    return image


def caption_font(size, bold=True):
    path = UI_FONT if bold else UI_FONT_REGULAR
    return ImageFont.truetype(str(path), size)


def contact_sheet(tiles, columns, tile_size, title, subtitle=''):
    tile_width, tile_height = tile_size
    caption_height = 42
    pad = 10
    header = 78 if subtitle else 56
    rows = math.ceil(len(tiles) / columns)
    width = columns * (tile_width + pad) + pad
    height = header + rows * (tile_height + caption_height + pad) + pad
    sheet = Image.new('RGB', (width, height), (24, 25, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad + 2, 14), title, font=caption_font(26), fill=(245, 245, 245))
    if subtitle:
        draw.text((pad + 2, 46), subtitle, font=caption_font(15, False), fill=(165, 170, 178))
    small = caption_font(14)
    tiny = caption_font(13, False)
    for index, (image, line_one, line_two, accent) in enumerate(tiles):
        column, row = index % columns, index // columns
        x = pad + column * (tile_width + pad)
        y = header + row * (tile_height + caption_height + pad)
        sheet.paste(image.resize((tile_width, tile_height), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + tile_width - 1, y + tile_height - 1], outline=accent, width=2)
        draw.text((x + 3, y + tile_height + 5), line_one, font=small, fill=(238, 238, 240))
        draw.text((x + 3, y + tile_height + 23), line_two, font=tiny, fill=(160, 166, 175))
    return sheet


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def build(args, word):
    info = word_contours(word)
    lattice = Lattice(args.density, args.fluidity, args.copies, info['stroke'],
                      refine=args.refine, background=args.background)
    return info, lattice


def command_single(args):
    info, lattice = build(args, args.word)
    mask, stencil = make_frame(args.variant, lattice, info, args.time)
    preview = composite(mask, args.copies, args.frames)
    word_x, word_y = word_position(info, args.time)
    scores = score_frame(mask, stencil, preview, info, word_x, word_y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.out)
    mask.save(args.out.with_name(args.out.stem + '-mask.png'))
    print(json.dumps({'word': args.word, 'refine': lattice.refine,
                      'grid': f'{lattice.columns}x{lattice.rows}',
                      'stroke': round(info['stroke'], 1), **scores}, indent=2))


def command_wireframe(args):
    info, lattice = build(args, args.word)
    image = wireframe(lattice, info, args.time, args.variant)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f'{args.out}')


def command_video(args):
    info, lattice = build(args, args.word)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libvpx-vp9', '-lossless', '1', '-b:v', '0', '-cpu-used', '4',
        '-pix_fmt', 'yuv420p', str(args.out),
    ], stdin=subprocess.PIPE)
    preview_path = args.out.with_suffix('.preview.mp4')
    preview_encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        str(preview_path),
    ], stdin=subprocess.PIPE)
    totals = []
    frame_count = max(1, int(round(FPS * args.seconds)))
    try:
        for frame in range(frame_count):
            time_seconds = frame / FPS
            mask, stencil = make_frame(args.variant, lattice, info, time_seconds)
            preview = composite(mask, args.copies, args.frames)
            encoder.stdin.write(mask.tobytes())
            preview_encoder.stdin.write(preview.tobytes())
            if frame % (FPS // 2) == 0:
                word_x, word_y = word_position(info, time_seconds)
                scores = score_frame(mask, stencil, preview, info, word_x, word_y)
                totals.append((round(time_seconds, 2), scores))
                print(f'  t={time_seconds:5.2f}s  {scores}', flush=True)
            if frame % 30 == 0:
                mask.save(args.out.with_name(f'{args.out.stem}-{frame // FPS}s-mask.png'))
                preview.save(args.out.with_name(f'{args.out.stem}-{frame // FPS}s.png'))
    finally:
        encoder.stdin.close()
        preview_encoder.stdin.close()
    encoder.wait()
    preview_encoder.wait()
    worst = min(totals, key=lambda item: item[1]['score'])
    best = max(totals, key=lambda item: item[1]['score'])
    print(json.dumps({'video': str(args.out), 'worst': worst, 'best': best,
                      'flips': flip_report(lattice)}, indent=2))


MATRIX_ROWS = [
    ('Yope3D', 1, 0, 'cut', None),
    ('Yope3D', 4, 60, 'cut', None),
    ('Yope3D', 12, 60, 'cut', None),
    ('SpinStack', 4, 60, 'cut', None),
    ('3D Gravity Simulator', 4, 60, 'cut', None),
    ('Yope3D', 4, 60, 'cut', 1),      # control: no local refinement
    ('Yope3D', 4, 60, 'snap', None),  # control: vertex snapping + Delaunay
]
MATRIX_DENSITIES = [12, 24, 48, 96]


def command_matrix(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    composite_tiles, mask_tiles, rows_data = [], [], []
    info_cache = {}
    for word, copies, fluidity, variant, refine in MATRIX_ROWS:
        if word not in info_cache:
            info_cache[word] = word_contours(word)
        info = info_cache[word]
        for density in MATRIX_DENSITIES:
            lattice = Lattice(density, fluidity, copies, info['stroke'],
                              refine=refine if refine is not None else args.refine,
                              background=args.background)
            mask, stencil = make_frame(variant, lattice, info, args.time)
            preview = composite(mask, copies, args.frames)
            word_x, word_y = word_position(info, args.time)
            scores = score_frame(mask, stencil, preview, info, word_x, word_y)
            record = {'word': word, 'density': density, 'copies': copies,
                      'fluidity': fluidity, 'variant': variant,
                      'refine': lattice.refine, 'refine_forced': refine,
                      'grid': f'{lattice.columns}x{lattice.rows}', **scores}
            rows_data.append(record)
            good = scores['score'] >= 0.60
            accent = (86, 176, 110) if good else ((214, 170, 60) if scores['score'] >= 0.45 else (206, 76, 70))
            tag = 'snap' if variant == 'snap' else ('cut no-refine' if refine == 1 else 'cut')
            line_one = f"{word[:22]}  d{density} c{copies} f{fluidity}  [{tag}]"
            line_two = (f"r{lattice.refine} {lattice.columns}x{lattice.rows} | seam {scores['visible']:.2f} "
                        f"id {scores['id_change']:.2f} pcs {scores['pieces']} slv {scores['slivers']} "
                        f"sq {scores['squint']:.2f} = {scores['score']:.2f}")
            composite_tiles.append((preview, line_one, line_two, accent))
            mask_tiles.append((mask.convert('RGB'), line_one, line_two, accent))
            print(f'{line_one:42s} {line_two}', flush=True)

    subtitle = ('Composite previews at t=%.1fs.  seam = fraction of glyph perimeter with visible luma change; '
                'id = fraction with an ID change; pcs = whole cells inside the glyph; sq = blurred-correlation squint test.'
                % args.time)
    sheet = contact_sheet(composite_tiles, len(MATRIX_DENSITIES), (440, 248),
                          'Agent B - contour-conforming moving lattice (composite preview)', subtitle)
    sheet.save(out_dir / 'contact-sheet.png')

    mask_sheet = contact_sheet(mask_tiles, len(MATRIX_DENSITIES), (440, 248),
                               'Agent B - raw grayscale clip-ID masks (same cells)',
                               'Gray level = clip ID. Wireframe of the conforming mesh is appended below.')
    wire_info = info_cache['Yope3D']
    wire_lattice = Lattice(48, 60, 4, wire_info['stroke'], refine=args.refine, background=args.background)
    wire = wireframe(wire_lattice, wire_info, args.time, 'cut')
    wire.save(out_dir / 'wireframe.png')
    wire_small = wire.resize((1780, 1001), Image.LANCZOS)
    combined = Image.new('RGB', (max(mask_sheet.width, wire_small.width + 20),
                                 mask_sheet.height + wire_small.height + 60), (24, 25, 28))
    combined.paste(mask_sheet, (0, 0))
    draw = ImageDraw.Draw(combined)
    draw.text((12, mask_sheet.height + 14),
              'Wireframe: fine lattice cells (blue) + glyph contour as a shared mesh edge (red), Yope3D d48 c4 f60',
              font=caption_font(20), fill=(238, 238, 240))
    combined.paste(wire_small, (10, mask_sheet.height + 50))
    combined.save(out_dir / 'contact-sheet-mask.png')

    (out_dir / 'scores.json').write_text(json.dumps(rows_data, indent=2))
    print(f"\nwrote {out_dir / 'contact-sheet.png'}")
    print(f"wrote {out_dir / 'contact-sheet-mask.png'}")
    print(f"wrote {out_dir / 'wireframe.png'}")


def command_sweep(args):
    """Full density x copies x word x fluidity matrix, numbers only."""
    results = []
    info_cache = {}
    for word in ('Yope3D', 'SpinStack', '3D Gravity Simulator'):
        info_cache[word] = word_contours(word)
        for density in (12, 24, 48, 96):
            for copies in (1, 4, 12):
                for fluidity in (0, 60):
                    info = info_cache[word]
                    lattice = Lattice(density, fluidity, copies, info['stroke'],
                                      refine=args.refine, background=args.background)
                    frame_scores = []
                    for time_seconds in (0.0, 3.0, 7.5):
                        mask, stencil = make_frame(args.variant, lattice, info, time_seconds)
                        preview = composite(mask, copies, args.frames)
                        word_x, word_y = word_position(info, time_seconds)
                        frame_scores.append(score_frame(mask, stencil, preview, info, word_x, word_y))
                    mean = {key: round(float(np.mean([f[key] for f in frame_scores])), 3)
                            for key in frame_scores[0]}
                    record = {'word': word, 'density': density, 'copies': copies,
                              'fluidity': fluidity, 'refine': lattice.refine, **mean}
                    results.append(record)
                    print(f"{word[:20]:20s} d{density:3d} c{copies:2d} f{fluidity:2d} r{lattice.refine} "
                          f"seam {mean['visible']:.2f} id {mean['id_change']:.2f} "
                          f"pcs {mean['pieces']:.0f} slv {mean['slivers']:.0f} "
                          f"sq {mean['squint']:.2f} score {mean['score']:.2f}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'sweep.json').write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out_dir / 'sweep.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('single', 'matrix', 'sweep', 'video', 'wireframe'))
    parser.add_argument('--word', default='Yope3D')
    parser.add_argument('--density', type=int, default=48)
    parser.add_argument('--copies', type=int, default=4)
    parser.add_argument('--fluidity', type=float, default=60)
    parser.add_argument('--refine', type=int, default=None,
                        help='fine cells per coarse piece per axis (default: from stroke width)')
    parser.add_argument('--background', choices=('four', 'checker'), default='four')
    parser.add_argument('--variant', choices=('cut', 'snap'), default='cut')
    parser.add_argument('--time', type=float, default=0.0)
    parser.add_argument('--seconds', type=float, default=SECONDS)
    parser.add_argument('--frames', type=Path,
                        default=Path('/private/tmp/claude-501/-Users-me-Desktop-dev-yugumishra-github-io'
                                     '/ea9e4c74-aeb0-4767-b13b-b99660eec2a7/scratchpad/agent-b/frames'))
    parser.add_argument('--out', type=Path, default=Path('/private/tmp/agent-b-single.png'))
    parser.add_argument('--out-dir', type=Path,
                        default=Path('/private/tmp/claude-501/-Users-me-Desktop-dev-yugumishra-github-io'
                                     '/ea9e4c74-aeb0-4767-b13b-b99660eec2a7/scratchpad/agent-b'))
    args = parser.parse_args()
    {'single': command_single, 'matrix': command_matrix, 'sweep': command_sweep,
     'video': command_video, 'wireframe': command_wireframe}[args.command](args)


if __name__ == '__main__':
    main()
