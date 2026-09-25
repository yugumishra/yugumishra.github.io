"""Agent A probe: legibility of the opening showcase by ID-space partition.

The geometry is deliberately unchanged from ``inspect-point-mask.py``: the same
quad lattice, the same bilinear coarse-to-fine interpolation, the same
per-vertex wobble.  Nothing is retriangulated to follow the glyph and no piece
statistics are touched.  The only thing this probe changes is *which clip ID*
each lattice face receives.

Idea under test
---------------
Partition the ``4 * copies`` slot space into an INSIDE family and an OUTSIDE
family that share no member.  Fill the glyph region from a (possibly finer)
lattice restricted to the INSIDE family and everything else from the coarse
lattice restricted to the OUTSIDE family.  Every glyph contour pixel is then a
genuine boundary between two different clips by construction, while the letter
interior is still cut into several moving lattice faces holding several
different clips -- a collage, not a stamped shape.

This file is self-contained on purpose; ``inspect-point-mask.py`` is shared
with other experiments and is not modified.

Examples::

    python scripts/probe-a-id-partition.py cell --density 48 --copies 12 \
        --word Yope3D --partition luma-split --subdivide 3 --out /tmp/cell
    python scripts/probe-a-id-partition.py matrix --out-dir /tmp/agent-a
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1280, 720
FONT_PATH = Path('/System/Library/Fonts/Supplemental/Arial Black.ttf')
CAPTION_FONT = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')

# Source order matches the browser compositor: 0 Yope3D cloth, 1 SpinStack,
# 2 MNIST digits, 3 gravity.
SOURCE_CLIPS = [
    ROOT / 'public/media/yope_cloth_b.mp4',
    ROOT / 'public/media/showcase-test/spinstack.mp4',
    ROOT / 'public/media/showcase-test/digits.mp4',
    ROOT / 'public/media/showcase-test/gravity.mp4',
]
SOURCE_NAMES = ['yope', 'spinstack', 'digits', 'gravity']

# Copied from src/scripts/showcase-test.ts so the preview crops match what the
# browser would actually show for copies > 1.
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

WORD_FEATURED = {
    'Yope3D': 0,
    'SpinStack': 1,
    '3D Gravity Simulator': 3,
    'MNIST': 2,
    'Yo': 0,
}


# ---------------------------------------------------------------------------
# lattice (verbatim behaviour from inspect-point-mask.py)
# ---------------------------------------------------------------------------

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
            piece_error = abs(columns * rows - target)
            aspect_error = abs(columns / rows - aspect)
            score = piece_error * 6 + aspect_error
            if score < best[2]:
                best = (columns, rows, score)
    return best[:2]


def build_lattice(piece_columns, piece_rows, columns, rows, fluidity, time_seconds):
    """Return the wobbled vertex positions of a columns x rows lattice.

    The coarse ``piece`` grid supplies the shape; a finer grid is bilinearly
    interpolated inside it exactly as the browser does, then every interior
    vertex wobbles on its own pseudo-random path.
    """
    coarse_positions = []
    for coarse_row in range(piece_rows + 1):
        for coarse_column in range(piece_columns + 1):
            original = (
                ORIGINAL_VERTICES[coarse_row][coarse_column]
                if piece_columns == 4 and piece_rows == 3 else None
            )
            coarse_positions.append((
                original[0] if original else coarse_column * WIDTH / piece_columns,
                original[1] if original else coarse_row * HEIGHT / piece_rows,
            ))

    def coarse_position(column, row):
        return coarse_positions[row * (piece_columns + 1) + column]

    vertices = []
    for row in range(rows + 1):
        for column in range(columns + 1):
            seed = row * 97 + column * 53 + columns * 11 + rows * 7
            coarse_x = column / columns * piece_columns
            coarse_y = row / rows * piece_rows
            left = min(piece_columns - 1, math.floor(coarse_x))
            top = min(piece_rows - 1, math.floor(coarse_y))
            x_amount = coarse_x - left
            y_amount = coarse_y - top
            tl = coarse_position(left, top)
            tr = coarse_position(left + 1, top)
            bl = coarse_position(left, top + 1)
            br = coarse_position(left + 1, top + 1)
            vertices.append({
                'x': (tl[0] * (1 - x_amount) * (1 - y_amount)
                      + tr[0] * x_amount * (1 - y_amount)
                      + bl[0] * (1 - x_amount) * y_amount
                      + br[0] * x_amount * y_amount),
                'y': (tl[1] * (1 - x_amount) * (1 - y_amount)
                      + tr[1] * x_amount * (1 - y_amount)
                      + bl[1] * (1 - x_amount) * y_amount
                      + br[1] * x_amount * y_amount),
                'phase_x': seeded(seed + 1) * math.tau,
                'phase_y': seeded(seed + 2) * math.tau,
                'speed_x': 0.72 + seeded(seed + 3) * 0.56,
                'speed_y': 0.72 + seeded(seed + 4) * 0.56,
            })

    amount = fluidity / 100
    amplitude = min(WIDTH / columns, HEIGHT / rows) * 0.24 * amount
    motion = 0.55 + amount * 0.85
    lattice = []
    for index, vertex in enumerate(vertices):
        column = index % (columns + 1)
        row = index // (columns + 1)
        boundary = column in (0, columns) or row in (0, rows)
        if boundary or amplitude == 0:
            lattice.append((vertex['x'], vertex['y']))
            continue
        phase = time_seconds * motion
        wobble_x = (math.sin(phase * vertex['speed_x'] + vertex['phase_x']) * 0.68
                    + math.sin(phase * 0.61 + vertex['phase_y']) * 0.32)
        wobble_y = (math.sin(phase * vertex['speed_y'] + vertex['phase_y']) * 0.68
                    + math.cos(phase * 0.57 + vertex['phase_x']) * 0.32)
        lattice.append((vertex['x'] + wobble_x * amplitude,
                        vertex['y'] + wobble_y * amplitude))
    return lattice


# ---------------------------------------------------------------------------
# word rasterisation and straight-segment contours
# ---------------------------------------------------------------------------

def rasterise_word(word, max_width=1030, max_height=250):
    """Rasterise ``word`` in Arial Black, auto-fitted to the given box.

    Same technique as generate-showcase-assets.py (size 230 for Yope3D); the
    size is searched so that longer names still fit the frame, which is how a
    long name ends up with thinner strokes.
    """
    size = 240
    while size > 20:
        font = ImageFont.truetype(str(FONT_PATH), size)
        bounds = font.getbbox(word)
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if w <= max_width and h <= max_height:
            break
        size -= 2
    font = ImageFont.truetype(str(FONT_PATH), size)
    bounds = font.getbbox(word)
    w, h = max(1, bounds[2] - bounds[0]), max(1, bounds[3] - bounds[1])
    image = Image.new('L', (w, h), 0)
    ImageDraw.Draw(image).text((-bounds[0], -bounds[1]), word, font=font, fill=255)
    image = image.point(lambda v: 255 if v >= 128 else 0)
    return image, size


def trace_contours(image, sample=8):
    """Marching-squares contours of a binary word bitmap, resampled to
    straight segments (the project asks for no spline boundaries)."""
    w, h = image.size
    pixels = image.load()

    def filled(x, y):
        return 0 <= x < w and 0 <= y < h and pixels[x, y] >= 128

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

    for y in range(-1, h):
        for x in range(-1, w):
            code = ((1 if filled(x, y) else 0)
                    | (2 if filled(x + 1, y) else 0)
                    | (4 if filled(x + 1, y + 1) else 0)
                    | (8 if filled(x, y + 1) else 0))
            for first, second in cases.get(code, []):
                add_segment(edge_points[first](x, y), edge_points[second](x, y))

    unused = set(range(len(segments)))
    contours = []
    while unused:
        first_index = next(iter(unused))
        start = segments[first_index][0]
        current, previous, contour = start, None, []
        while True:
            candidates = [i for i in nodes[current] if i in unused]
            if previous is not None and len(candidates) > 1:
                candidates = [i for i in candidates if i != previous] or candidates
            if not candidates:
                break
            index = candidates[0]
            unused.remove(index)
            a, b = segments[index]
            contour.append((current[0] / 2, current[1] / 2))
            current = b if a == current else a
            previous = index
            if current == start:
                break
        if current != start or len(contour) < 3:
            continue
        total = sum(math.hypot(contour[(i + 1) % len(contour)][0] - p[0],
                               contour[(i + 1) % len(contour)][1] - p[1])
                    for i, p in enumerate(contour))
        count = max(3, math.ceil(total / sample))
        sampled = []
        for step in range(count):
            target = step * total / count
            for i, p in enumerate(contour):
                end = contour[(i + 1) % len(contour)]
                length = math.hypot(end[0] - p[0], end[1] - p[1])
                if target <= length or i == len(contour) - 1:
                    ratio = 0 if length == 0 else target / length
                    sampled.append((p[0] + (end[0] - p[0]) * ratio,
                                    p[1] + (end[1] - p[1]) * ratio))
                    break
                target -= length
        contours.append(sampled)
    return contours


def contour_region(contours, word_x, word_y):
    region = Image.new('1', (WIDTH, HEIGHT), 0)
    for contour in contours:
        loop = Image.new('1', (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(loop).polygon(
            [(word_x + x, word_y + y) for x, y in contour], fill=1)
        region = ImageChops.logical_xor(region, loop)
    return region


def stroke_width(region_array):
    """Median stroke thickness, from the distance transform of the glyphs."""
    from scipy import ndimage
    distance = ndimage.distance_transform_edt(region_array)
    ridge = distance[distance > 0]
    if ridge.size == 0:
        return 0.0
    return float(np.percentile(ridge, 90) * 2)


# ---------------------------------------------------------------------------
# clip frames and per-slot imagery
# ---------------------------------------------------------------------------

_FRAME_CACHE = {}
_SLOT_IMAGE_CACHE = {}


def source_frame(index, frames_dir):
    if index in _FRAME_CACHE:
        return _FRAME_CACHE[index]
    path = frames_dir / f'{SOURCE_NAMES[index]}.png'
    if not path.is_file():
        import subprocess
        frames_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', '2',
                        '-i', str(SOURCE_CLIPS[index]), '-frames:v', '1',
                        str(path)], check=True)
    image = Image.open(path).convert('RGB')
    _FRAME_CACHE[index] = image
    return image


def slot_image(source, copy, frames_dir):
    """The full-screen pixels the compositor would show for one slot.

    Mirrors drawFootageAtlas + the fragment shader: the clip is cover-fitted
    into a 640x360 atlas tile with the copy's zoom/offset, and that tile is
    stretched across the 1280x720 stage.
    """
    key = (source, copy)
    if key in _SLOT_IMAGE_CACHE:
        return _SLOT_IMAGE_CACHE[key]
    frame = source_frame(source, frames_dir)
    zoom, dx, dy = COPY_TRANSFORMS[(copy + source) % len(COPY_TRANSFORMS)]
    tile_w, tile_h = WIDTH, HEIGHT           # atlas tile at 2x for quality
    scale = max(tile_w / frame.width, tile_h / frame.height) * zoom
    w, h = int(round(frame.width * scale)), int(round(frame.height * scale))
    canvas = Image.new('RGB', (tile_w, tile_h), (11, 13, 16))
    canvas.paste(frame.resize((w, h), Image.LANCZOS),
                 (int(round((tile_w - w) / 2 + dx * tile_w)),
                  int(round((tile_h - h) / 2 + dy * tile_h))))
    array = np.asarray(canvas, dtype=np.uint8)
    _SLOT_IMAGE_CACHE[key] = array
    return array


def luma(array):
    a = array.astype(np.float32)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def slot_luma_table(copies, frames_dir):
    return np.array([luma(slot_image(slot // copies, slot % copies, frames_dir)).mean()
                     for slot in range(4 * copies)], dtype=np.float32)


# ---------------------------------------------------------------------------
# ID-space partitions
# ---------------------------------------------------------------------------

def partition(mode, copies, featured, frames_dir):
    """Return (inside_family, outside_family) as disjoint slot lists."""
    total = 4 * copies
    all_slots = list(range(total))
    if mode == 'src-half':
        inside = [s for s in all_slots if s // copies in (2, 3)]
    elif mode == 'featured-in':
        inside = [s for s in all_slots if s // copies == featured]
    elif mode == 'featured-out':
        inside = [s for s in all_slots if s // copies != featured]
    elif mode == 'variant-split':
        half = max(1, copies // 2)
        inside = [s for s in all_slots if s % copies >= half]
    elif mode in ('luma-split', 'luma-split1'):
        # Data-driven split: rank every slot by the mean luma of the pixels it
        # would actually show, then cut where the two groups separate most.
        # ``luma-split`` keeps at least two slots on each side so the letter
        # interior can still be a collage; ``luma-split1`` is the unconstrained
        # control, which degenerates to one flat interior ID at copies = 1.
        table = slot_luma_table(copies, frames_dir)
        order = list(np.argsort(table))
        floor_count = 1 if mode == 'luma-split1' else min(2, total // 2)
        best, best_gap = None, -1.0
        for cut in range(floor_count, total - floor_count + 1):
            dark, bright = order[:cut], order[cut:]
            gap = float(table[bright].mean() - table[dark].mean())
            if gap > best_gap:
                best_gap, best = gap, (dark, bright)
        dark, bright = best
        # Letters take the brighter family: a bright name on a dark collage
        # matches the site's dark opening.
        inside = sorted(int(s) for s in bright)
    elif mode == 'overlap':
        # Control: the failure this experiment is meant to remove.  Inside and
        # outside draw from the same pool, so an inside face can land on the
        # same clip as the outside face it touches and the contour disappears
        # there.  Returned early because the families are deliberately equal.
        return all_slots, all_slots
    else:
        raise SystemExit(f'unknown partition {mode}')
    inside_set = set(inside)
    outside = [s for s in all_slots if s not in inside_set]
    if not inside or not outside:
        raise SystemExit(f'partition {mode} is degenerate at copies={copies}')
    return inside, outside


def ordered_family(family, luma_table):
    """Interleave a family by luma so either parity half spans its range."""
    return [int(s) for s in sorted(family, key=lambda s: luma_table[s])]


def family_slot(family, row, column):
    """Pick a family member for a lattice cell.

    Parity chooses one of two interleaved halves, so two neighbouring cells
    never land on the same slot while len(family) >= 2; a hash chooses within
    the half so the field keeps its scattered look.
    """
    if len(family) == 1:
        return family[0]
    half = family[0::2] if (row + column) % 2 == 0 else family[1::2]
    if not half:
        half = family
    return half[(column * 3 + row * 5 + (column ^ row)) % len(half)]


# ---------------------------------------------------------------------------
# frame construction
# ---------------------------------------------------------------------------

def draw_lattice(columns, rows, lattice, family, copies):
    image = Image.new('L', (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    denominator = max(1, 4 * copies - 1)
    for row in range(rows):
        for column in range(columns):
            slot = family_slot(family, row, column)
            value = js_round(slot * 255 / denominator)
            draw.polygon([
                lattice[row * (columns + 1) + column],
                lattice[row * (columns + 1) + column + 1],
                lattice[(row + 1) * (columns + 1) + column + 1],
                lattice[(row + 1) * (columns + 1) + column],
            ], fill=value)
    return image


def auto_subdivide(config, word_data, piece_columns, piece_rows):
    """How finely the letter interior is cut.

    ``subdivide = 0`` derives the factor from the glyph: the interior lattice
    cell is aimed at roughly 0.85 stroke widths, with a 28px floor so the
    letters stay pieces of footage rather than dither.  This is the part that
    makes the scheme density-independent -- at density 12 one coarse cell would
    otherwise swallow whole letters, and at density 96 no extra cut is needed.
    """
    if config['subdivide'] > 0:
        return config['subdivide']
    cell = math.sqrt((WIDTH / piece_columns) * (HEIGHT / piece_rows))
    target = max(28.0, 0.85 * max(1.0, word_data['stroke']))
    return int(max(1, min(12, round(cell / target))))


def make_frame(config, time_seconds, word_data):
    """Render one selector-mask frame plus the region masks used for scoring."""
    copies = config['copies']
    piece_columns, piece_rows = grid_for_density(config['density'])
    contours, word_w, word_h = word_data['contours'], word_data['w'], word_data['h']

    phase = time_seconds / 12 * math.tau
    amp_x = min(125, max(0, (WIDTH - word_w) / 2 - 20))
    amp_y = min(120, max(0, (HEIGHT - word_h) / 2 - 20))
    word_x = js_round((WIDTH - word_w) / 2 + amp_x * math.sin(phase))
    word_y = js_round((HEIGHT - word_h) / 2 + amp_y * math.sin(phase * 2))

    region = contour_region(contours, word_x, word_y)
    region_mask = region.convert('L').point(lambda v: 255 if v else 0)

    inside, outside = word_data['inside'], word_data['outside']
    luma_table = word_data['luma_table']
    inside = ordered_family(inside, luma_table)
    outside = ordered_family(outside, luma_table)

    coarse = build_lattice(piece_columns, piece_rows, piece_columns, piece_rows,
                           config['fluidity'], time_seconds)
    mask = draw_lattice(piece_columns, piece_rows, coarse, outside, copies)

    subdivide = auto_subdivide(config, word_data, piece_columns, piece_rows)
    fine_columns = piece_columns * subdivide
    fine_rows = piece_rows * subdivide
    fine = build_lattice(piece_columns, piece_rows, fine_columns, fine_rows,
                         config['fluidity'], time_seconds)
    inner = draw_lattice(fine_columns, fine_rows, fine, inside, copies)

    band_mask = None
    if config['band'] > 0:
        radius = int(config['band']) * 2 + 1
        grown = region_mask.filter(ImageFilter.MaxFilter(min(radius, 63)))
        band_mask = ImageChops.subtract(grown, region_mask)
        guard = word_data['guard']
        if config['band_mode'] == 'single':
            value = js_round(guard[0] * 255 / max(1, 4 * copies - 1))
            band_image = Image.new('L', (WIDTH, HEIGHT), value)
        else:
            band_image = draw_lattice(fine_columns, fine_rows, fine, guard, copies)
        mask.paste(band_image, mask=band_mask)

    mask.paste(inner, mask=region_mask)
    return {
        'mask': mask,
        'region': np.asarray(region_mask, dtype=np.uint8) > 0,
        'band': (np.asarray(band_mask, dtype=np.uint8) > 0) if band_mask is not None else None,
        'word_box': (word_x, word_y, word_w, word_h),
    }


def mask_to_slots(mask, copies):
    values = np.asarray(mask, dtype=np.float32)
    return np.clip(np.round(values / 255 * (4 * copies - 1)), 0, 4 * copies - 1).astype(np.int16)


def composite(slots, copies, frames_dir):
    out = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for slot in np.unique(slots):
        selection = slots == slot
        out[selection] = slot_image(int(slot) // copies, int(slot) % copies,
                                    frames_dir)[selection]
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def shifted_pairs(array):
    return [
        (array[1:, :], array[:-1, :]),
        (array[:-1, :], array[1:, :]),
        (array[:, 1:], array[:, :-1]),
        (array[:, :-1], array[:, 1:]),
    ]


def score_frame(frame, slots, rgb, inside_family, copies, luma_table):
    region = frame['region']
    inside_set = np.zeros(4 * copies, dtype=bool)
    inside_set[list(inside_family)] = True

    # (a) contour coverage: glyph-boundary pixel pairs whose ID changes.
    changes = 0
    crossings = 0
    contrast_sum = 0.0
    frame_luma = luma(rgb)
    for (a_region, b_region), (a_slots, b_slots), (a_l, b_l) in zip(
            shifted_pairs(region), shifted_pairs(slots), shifted_pairs(frame_luma)):
        crossing = a_region & ~b_region
        crossings += int(crossing.sum())
        changes += int((crossing & (a_slots != b_slots)).sum())
        if crossing.any():
            contrast_sum += float(np.abs(a_l[crossing] - b_l[crossing]).sum())
    coverage = changes / crossings if crossings else 0.0
    edge_contrast = contrast_sum / crossings if crossings else 0.0

    # (b) interior purity: glyph area actually holding inside-family IDs.
    interior = slots[region]
    purity = float(inside_set[interior].mean()) if interior.size else 0.0
    exterior = slots[~region]
    leak = float(inside_set[exterior].mean()) if exterior.size else 0.0

    # collage: distinct IDs inside, and how many of them are visually distinct.
    ids, counts = np.unique(interior, return_counts=True)
    fractions = counts / max(1, interior.size)
    significant = ids[fractions >= 0.02]
    distinct_ids = int(significant.size)
    lumas = sorted(float(luma_table[i]) for i in significant)
    visual = 1
    last = lumas[0] if lumas else 0.0
    for value in lumas[1:]:
        if value - last >= 12:
            visual += 1
            last = value
    visual = visual if lumas else 0

    # interior seam density: fraction of interior pixels next to another ID.
    seam = np.zeros_like(region)
    seam[1:, :] |= (slots[1:, :] != slots[:-1, :]) & region[1:, :] & region[:-1, :]
    seam[:-1, :] |= (slots[1:, :] != slots[:-1, :]) & region[1:, :] & region[:-1, :]
    seam[:, 1:] |= (slots[:, 1:] != slots[:, :-1]) & region[:, 1:] & region[:, :-1]
    seam[:, :-1] |= (slots[:, 1:] != slots[:, :-1]) & region[:, 1:] & region[:, :-1]
    seam_fraction = float(seam[region].mean()) if region.any() else 0.0

    # Visible seams are measured on the rendered composite, not on slot means:
    # two crops of the same clip still meet in a real discontinuity, while two
    # different but equally black clips do not.
    def visible_seam_map(side):
        visible = np.zeros_like(region)
        vertical = ((slots[1:, :] != slots[:-1, :])
                    & (np.abs(frame_luma[1:, :] - frame_luma[:-1, :]) >= 12)
                    & side[1:, :] & side[:-1, :])
        horizontal = ((slots[:, 1:] != slots[:, :-1])
                      & (np.abs(frame_luma[:, 1:] - frame_luma[:, :-1]) >= 12)
                      & side[:, 1:] & side[:, :-1])
        visible[1:, :] |= vertical
        visible[:-1, :] |= vertical
        visible[:, 1:] |= horizontal
        visible[:, :-1] |= horizontal
        return visible

    interior_visible = visible_seam_map(region)
    exterior_visible = visible_seam_map(~region)
    visible_seam = float(interior_visible[region].mean()) if region.any() else 0.0
    exterior_seam = float(exterior_visible[~region].mean()) if (~region).any() else 0.0
    background_luma = float(frame_luma[~region].mean()) if (~region).any() else 0.0
    interior_luma = float(frame_luma[region].mean()) if region.any() else 0.0

    # (c) squint test: blur + downsample the composite, correlate with the
    # blurred ideal glyph.  This approximates reading the name at a glance.
    blurred = np.asarray(Image.fromarray(frame_luma.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(6)).resize((WIDTH // 8, HEIGHT // 8), Image.BOX),
        dtype=np.float32)
    ideal = np.asarray(Image.fromarray((region * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(6)).resize((WIDTH // 8, HEIGHT // 8), Image.BOX),
        dtype=np.float32)
    if blurred.std() < 1e-3 or ideal.std() < 1e-3:
        squint = 0.0
    else:
        squint = float(np.corrcoef(blurred.ravel(), ideal.ravel())[0, 1])

    # 90 is roughly the luma gap between the SpinStack clip and the three
    # dark clips, i.e. the best separation this footage set can offer.
    contrast_norm = min(1.0, edge_contrast / 90.0)
    legibility = 100 * (0.5 * abs(squint) + 0.3 * contrast_norm + 0.2 * coverage)
    collage = 100 * (0.4 * min(1.0, distinct_ids / 4.0)
                     + 0.6 * min(1.0, visible_seam / 0.08))
    field = 100 * (0.5 * min(1.0, exterior_seam / 0.05)
                   + 0.5 * min(1.0, background_luma / 40.0))
    return {
        'coverage': coverage,
        'purity': purity,
        'leak': leak,
        'edge_contrast': edge_contrast,
        'squint': squint,
        'distinct_ids': distinct_ids,
        'visual_ids': visual,
        'seam': seam_fraction,
        'visible_seam': visible_seam,
        'exterior_seam': exterior_seam,
        'bg_luma': background_luma,
        'fg_luma': interior_luma,
        'LEG': legibility,
        'COL': collage,
        'FIELD': field,
    }


# ---------------------------------------------------------------------------
# one matrix cell
# ---------------------------------------------------------------------------

_WORD_CACHE = {}


def word_data_for(word, copies, partition_mode, frames_dir):
    key = (word, copies, partition_mode)
    if key in _WORD_CACHE:
        return _WORD_CACHE[key]
    bitmap, size = rasterise_word(word)
    contours = trace_contours(bitmap)
    featured = WORD_FEATURED.get(word, 0)
    inside, outside = partition(partition_mode, copies, featured, frames_dir)
    table = slot_luma_table(copies, frames_dir)
    # Guard family: outside slots reserved for the band and never reused
    # elsewhere, chosen for maximum luma distance from the inside family so the
    # band is a real separator rather than a second inside colour.
    inside_luma = float(np.mean([table[s] for s in inside]))
    guard = sorted(outside, key=lambda s: -abs(float(table[s]) - inside_luma))
    guard = sorted(guard[:max(1, len(outside) // 4)])
    data = {
        'contours': contours, 'w': bitmap.width, 'h': bitmap.height,
        'size': size, 'inside': inside, 'outside': outside,
        'guard': guard, 'luma_table': table, 'featured': featured,
        'stroke': stroke_width(np.asarray(bitmap, dtype=np.uint8) > 0),
    }
    _WORD_CACHE[key] = data
    return data


def run_cell(config, frames_dir, times=(0.0, 2.0, 5.0), preview_time=2.0):
    word_data = word_data_for(config['word'], config['copies'],
                              config['partition'], frames_dir)
    scores = []
    preview = None
    for time_seconds in times:
        frame = make_frame(config, time_seconds, word_data)
        slots = mask_to_slots(frame['mask'], config['copies'])
        rgb = composite(slots, config['copies'], frames_dir)
        scores.append(score_frame(frame, slots, rgb, word_data['inside'],
                                  config['copies'], word_data['luma_table']))
        if abs(time_seconds - preview_time) < 1e-6 or preview is None:
            preview = {'rgb': rgb, 'mask': frame['mask'], 'region': frame['region']}
    mean = {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}
    mean['cell_w'] = WIDTH / grid_for_density(config['density'])[0]
    mean['inside_slots'] = len(word_data['inside'])
    mean['outside_slots'] = len(word_data['outside'])
    mean['font_size'] = word_data['size']
    mean['stroke'] = word_data['stroke']
    mean['k'] = auto_subdivide(config, word_data, *grid_for_density(config['density']))
    return mean, preview


# ---------------------------------------------------------------------------
# contact sheets
# ---------------------------------------------------------------------------

def contact_sheet(tiles, columns, tile_width, path, title, caption_height=30):
    font = ImageFont.truetype(str(CAPTION_FONT), 15)
    title_font = ImageFont.truetype(str(CAPTION_FONT), 24)
    tile_height = int(tile_width * HEIGHT / WIDTH)
    rows = math.ceil(len(tiles) / columns)
    header = 44
    sheet = Image.new('RGB', (columns * tile_width,
                              header + rows * (tile_height + caption_height)),
                      (16, 17, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, font=title_font, fill=(245, 245, 245))
    for index, (image, caption, colour) in enumerate(tiles):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = header + row * (tile_height + caption_height)
        sheet.paste(Image.fromarray(image).resize((tile_width, tile_height), Image.LANCZOS)
                    if isinstance(image, np.ndarray)
                    else image.resize((tile_width, tile_height), Image.LANCZOS), (x, y))
        draw.rectangle([x, y + tile_height, x + tile_width - 1,
                        y + tile_height + caption_height - 1], fill=(24, 25, 29))
        for line_index, line in enumerate(caption.split('\n')):
            draw.text((x + 6, y + tile_height + 2 + line_index * 15), line,
                      font=font, fill=colour)
        draw.rectangle([x, y, x + tile_width - 1, y + tile_height - 1],
                       outline=(60, 62, 70))
    sheet.save(path)
    return sheet


def score_colour(leg):
    if leg >= 62:
        return (130, 226, 140)
    if leg >= 48:
        return (232, 214, 120)
    return (240, 130, 120)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def default_config(**overrides):
    config = {
        'density': 48, 'fluidity': 60, 'copies': 12, 'word': 'Yope3D',
        'partition': 'luma-split', 'subdivide': 0, 'band': 0,
        'band_mode': 'family',
    }
    config.update(overrides)
    return config


def cmd_cell(args):
    frames_dir = Path(args.frames)
    config = default_config(density=args.density, fluidity=args.fluidity,
                            copies=args.copies, word=args.word,
                            partition=args.partition, subdivide=args.subdivide,
                            band=args.band, band_mode=args.band_mode)
    mean, preview = run_cell(config, frames_dir, preview_time=args.time,
                             times=(args.time,))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview['rgb']).save(out.with_name(out.name + '-composite.png'))
    preview['mask'].save(out.with_name(out.name + '-mask.png'))
    print(json.dumps({**config, **{k: round(v, 4) for k, v in mean.items()}}, indent=2))


def cmd_matrix(args):
    frames_dir = Path(args.frames)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    words = ['Yope3D', 'SpinStack', '3D Gravity Simulator']
    densities = [12, 24, 48, 96]
    copies_list = [1, 4, 12]
    rows = []
    tiles = []
    mask_tiles = []
    for word in words:
        for copies in copies_list:
            for density in densities:
                config = default_config(density=density, copies=copies,
                                        word=word, fluidity=args.fluidity,
                                        partition=args.partition,
                                        subdivide=args.subdivide, band=args.band,
                                        band_mode=args.band_mode)
                mean, preview = run_cell(config, frames_dir)
                rows.append({**config, **mean})
                caption = (f"{word[:20]}  d{density} c{copies} k{mean['k']:.0f} f{args.fluidity}\n"
                           f"LEG {mean['LEG']:.0f}  COL {mean['COL']:.0f}  FLD {mean['FIELD']:.0f}\n"
                           f"cov {mean['coverage']:.2f} sq {mean['squint']:+.2f} ct {mean['edge_contrast']:.0f} ids {mean['distinct_ids']:.0f}")
                tiles.append((preview['rgb'], caption, score_colour(mean['LEG'])))
                mask_tiles.append((preview['mask'].convert('RGB'), caption,
                                   score_colour(mean['LEG'])))
                print(f"{word[:20]:22s} d{density:3d} c{copies:2d} "
                      f"LEG {mean['LEG']:5.1f} COL {mean['COL']:5.1f} FLD {mean['FIELD']:5.1f} "
                      f"cov {mean['coverage']:.2f} sq {mean['squint']:+.2f} "
                      f"ct {mean['edge_contrast']:5.1f} ids {mean['distinct_ids']:.0f} "
                      f"vseam {mean['visible_seam']:.3f}",
                      flush=True)
    contact_sheet(tiles, len(densities), 480,
                  out_dir / args.name,
                  f'Agent A - ID-space partition ({args.partition}, k={args.subdivide}, '
                  f'band={args.band}{args.band_mode[:3] if args.band else ""}, '
                  f'fluidity {args.fluidity}) - composite preview', 54)
    contact_sheet(mask_tiles, len(densities), 480,
                  out_dir / args.name.replace('.png', '-mask.png'),
                  f'Agent A - raw selector masks ({args.partition}, k={args.subdivide}, '
                  f'fluidity {args.fluidity})', 54)
    with open(out_dir / args.name.replace('.png', '.json'), 'w') as handle:
        json.dump(rows, handle, indent=1)


def cmd_sweep(args):
    """Ablations: partition mode, subdivision, guard band, fluidity."""
    frames_dir = Path(args.frames)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(args.spec).read_text())
    rows, tiles, mask_tiles = [], [], []
    for spec in specs:
        label = spec.pop('label')
        config = default_config(**spec)
        mean, preview = run_cell(config, frames_dir)
        rows.append({'label': label, **config, **mean})
        caption = (f"{label}\nLEG {mean['LEG']:.0f}  COL {mean['COL']:.0f}  FLD {mean['FIELD']:.0f}\n"
                   f"cov {mean['coverage']:.2f} sq {mean['squint']:+.2f} ct {mean['edge_contrast']:.0f} ids {mean['distinct_ids']:.0f}")
        tiles.append((preview['rgb'], caption, score_colour(mean['LEG'])))
        mask_tiles.append((preview['mask'].convert('RGB'), caption,
                           score_colour(mean['LEG'])))
        print(f"{label:32s} LEG {mean['LEG']:5.1f} COL {mean['COL']:5.1f} FLD {mean['FIELD']:5.1f} "
              f"cov {mean['coverage']:.2f} pur {mean['purity']:.2f} leak {mean['leak']:.3f} "
              f"ids {mean['distinct_ids']:.0f} vseam {mean['visible_seam']:.3f} "
              f"sq {mean['squint']:+.2f} ct {mean['edge_contrast']:5.1f}", flush=True)
    contact_sheet(tiles, args.columns, 440, out_dir / args.name,
                  f'Agent A - {args.title}', 54)
    contact_sheet(mask_tiles, args.columns, 440,
                  out_dir / args.name.replace('.png', '-mask.png'),
                  f'Agent A - {args.title} (masks)', 54)
    with open(out_dir / args.name.replace('.png', '.json'), 'w') as handle:
        json.dump(rows, handle, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', default=str(Path('/private/tmp/claude-501/-Users-me-Desktop-dev-yugumishra-github-io/ea9e4c74-aeb0-4767-b13b-b99660eec2a7/scratchpad/agent-a/frames')))
    sub = parser.add_subparsers(dest='command', required=True)

    cell = sub.add_parser('cell')
    cell.add_argument('--density', type=int, default=48)
    cell.add_argument('--fluidity', type=float, default=60)
    cell.add_argument('--copies', type=int, default=12)
    cell.add_argument('--word', default='Yope3D')
    cell.add_argument('--partition', default='luma-split')
    cell.add_argument('--subdivide', type=int, default=0)
    cell.add_argument('--band', type=int, default=0)
    cell.add_argument('--band-mode', default='family', choices=('single', 'family'))
    cell.add_argument('--time', type=float, default=2.0)
    cell.add_argument('--out', required=True)
    cell.set_defaults(func=cmd_cell)

    matrix = sub.add_parser('matrix')
    matrix.add_argument('--out-dir', required=True)
    matrix.add_argument('--name', default='contact-sheet.png')
    matrix.add_argument('--partition', default='luma-split')
    matrix.add_argument('--subdivide', type=int, default=0)
    matrix.add_argument('--band', type=int, default=0)
    matrix.add_argument('--band-mode', default='family', choices=('single', 'family'))
    matrix.add_argument('--fluidity', type=float, default=60)
    matrix.set_defaults(func=cmd_matrix)

    sweep = sub.add_parser('sweep')
    sweep.add_argument('--out-dir', required=True)
    sweep.add_argument('--spec', required=True)
    sweep.add_argument('--name', default='sweep.png')
    sweep.add_argument('--title', default='ablation')
    sweep.add_argument('--columns', type=int, default=4)
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
