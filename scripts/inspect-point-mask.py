"""Render the point-by-point selector as a mask-only video for inspection.

This is an offline, Pillow version of the browser's point-text geometry. It is
deliberately kept as a probe: it writes a grayscale mask video and stills, but
does not change what the showcase page loads. Point mode inserts contour
vertices into a subdivided lattice and renders polygon faces only. The
``contour-faces`` topology is the accepted output; the other topologies are
sibling experiments.

Example:
    python scripts/inspect-point-mask.py --density 96 --fluidity 0 \
        --out /private/tmp/point-mask-96.webm
"""

import argparse
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'public/media/showcase-test/letter-mask.png'
WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 30, 12
WORD_X, WORD_Y, WORD_WIDTH, WORD_HEIGHT = 155, 254, 970, 213

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
            piece_error = abs(columns * rows - target)
            aspect_error = abs(columns / rows - aspect)
            score = piece_error * 6 + aspect_error
            if score < best[2]:
                best = (columns, rows, score)
    return best[:2]


def text_contours():
    """Trace straight-segment contours from the binary word mask."""
    reference = Image.open(REFERENCE).convert('L').crop(
        (WORD_X, WORD_Y, WORD_X + WORD_WIDTH, WORD_Y + WORD_HEIGHT)
    )
    pixels = reference.load()

    def filled(x, y):
        return 0 <= x < WORD_WIDTH and 0 <= y < WORD_HEIGHT and pixels[x, y] >= 128

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

    for y in range(-1, WORD_HEIGHT):
        for x in range(-1, WORD_WIDTH):
            code = (
                (1 if filled(x, y) else 0)
                | (2 if filled(x + 1, y) else 0)
                | (4 if filled(x + 1, y + 1) else 0)
                | (8 if filled(x, y + 1) else 0)
            )
            for first, second in cases.get(code, []):
                add_segment(edge_points[first](x, y), edge_points[second](x, y))

    unused = set(range(len(segments)))
    contours = []
    while unused:
        first_index = next(iter(unused))
        start = segments[first_index][0]
        current = start
        previous = None
        contour = []
        while True:
            candidates = [index for index in nodes[current] if index in unused]
            if previous is not None and len(candidates) > 1:
                candidates = [index for index in candidates if index != previous] or candidates
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

        total = sum(
            math.hypot(
                contour[(index + 1) % len(contour)][0] - point[0],
                contour[(index + 1) % len(contour)][1] - point[1],
            )
            for index, point in enumerate(contour)
        )
        count = max(3, math.ceil(total / 8))
        sampled = []
        for sample in range(count):
            target = sample * total / count
            for index, point in enumerate(contour):
                end = contour[(index + 1) % len(contour)]
                length = math.hypot(end[0] - point[0], end[1] - point[1])
                if target <= length or index == len(contour) - 1:
                    ratio = 0 if length == 0 else target / length
                    sampled.append((point[0] + (end[0] - point[0]) * ratio,
                                    point[1] + (end[1] - point[1]) * ratio))
                    break
                target -= length
        contours.append(sampled)
    return contours


def segment_distance(point, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    raw_projection = 0 if length_squared == 0 else (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared
    projection = max(0, min(1, raw_projection))
    closest = (start[0] + projection * dx, start[1] + projection * dy)
    distance_squared = (point[0] - closest[0]) ** 2 + (point[1] - closest[1]) ** 2
    return distance_squared, raw_projection


def contour_region(contours, word_x, word_y):
    """Return the even/odd-filled, straight-segment word region.

    This deliberately uses the sampled marching-squares paths rather than
    pasting the source bitmap.  It makes the experiment's glyph boundaries
    actual polygon edges and preserves the inner counters through even/odd
    filling.
    """
    region = Image.new('1', (WIDTH, HEIGHT), 0)
    for contour in contours:
        loop = Image.new('1', (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(loop).polygon(
            [(word_x + x, word_y + y) for x, y in contour], fill=1,
        )
        region = ImageChops.logical_xor(region, loop)
    return region


def pure_lattice_mesh(lattice, contours, word_x, word_y, region, moat, columns, rows):
    """Build one two-ID triangulated field, constrained densely at the word.

    The regular, moving lattice vertices remain the structural vertices away
    from the name.  Densely sampled contour vertices are included in the same
    Delaunay mesh, so neighbouring faces turn to follow the glyph boundaries
    instead of a glyph image being painted after the lattice.
    """
    vertices = list(lattice)
    vertices.extend(
        (word_x + x, word_y + y)
        for contour in contours
        for x, y in contour
    )
    mesh = Delaunay(vertices)
    result = Image.new('L', (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(result)
    cell_width = WIDTH / columns
    cell_height = HEIGHT / rows
    for simplex in mesh.simplices:
        triangle = [vertices[index] for index in simplex]
        center_x = sum(point[0] for point in triangle) / 3
        center_y = sum(point[1] for point in triangle) / 3
        sample_x = max(0, min(WIDTH - 1, int(center_x)))
        sample_y = max(0, min(HEIGHT - 1, int(center_y)))
        if region.getpixel((sample_x, sample_y)):
            value = 85
        elif moat.getpixel((sample_x, sample_y)):
            value = 0
        else:
            # This matches the black/gray base-field assignment.  Adjacent
            # triangles in an untouched cell get the same ID, preserving the
            # original quadrilateral look once the name moves away.
            column = max(0, min(columns - 1, int(center_x / cell_width)))
            row = max(0, min(rows - 1, int(center_y / cell_height)))
            value = 85 if (column + row) % 2 else 0
        draw.polygon(triangle, fill=value)
    return result


def make_frame(
    density, fluidity, copies, time_seconds, points, edge_radius=None,
    contours=None, topology='contour-faces',
):
    piece_columns, piece_rows = grid_for_density(density)
    columns = max(piece_columns, 24) if contours else piece_columns
    rows = max(piece_rows, 16) if contours else piece_rows
    coarse_positions = []
    for coarse_row in range(piece_rows + 1):
        for coarse_column in range(piece_columns + 1):
            original = (
                ORIGINAL_VERTICES[coarse_row][coarse_column]
                if piece_columns == 4 and piece_rows == 3 else None
            )
            coarse_positions.append({
                'x': original[0] if original else coarse_column * WIDTH / piece_columns,
                'y': original[1] if original else coarse_row * HEIGHT / piece_rows,
            })

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
            top_left = coarse_position(left, top)
            top_right = coarse_position(left + 1, top)
            bottom_left = coarse_position(left, top + 1)
            bottom_right = coarse_position(left + 1, top + 1)
            vertices.append({
                'x': (
                    top_left['x'] * (1 - x_amount) * (1 - y_amount)
                    + top_right['x'] * x_amount * (1 - y_amount)
                    + bottom_left['x'] * (1 - x_amount) * y_amount
                    + bottom_right['x'] * x_amount * y_amount
                ),
                'y': (
                    top_left['y'] * (1 - x_amount) * (1 - y_amount)
                    + top_right['y'] * x_amount * (1 - y_amount)
                    + bottom_left['y'] * (1 - x_amount) * y_amount
                    + bottom_right['y'] * x_amount * y_amount
                ),
                'phase_x': seeded(seed + 1) * math.tau,
                'phase_y': seeded(seed + 2) * math.tau,
                'speed_x': 0.72 + seeded(seed + 3) * 0.56,
                'speed_y': 0.72 + seeded(seed + 4) * 0.56,
            })

    amount = fluidity / 100
    cell_width = WIDTH / columns
    cell_height = HEIGHT / rows
    amplitude = min(cell_width, cell_height) * 0.24 * amount
    motion = 0.55 + amount * 0.85
    lattice = []
    for index, vertex in enumerate(vertices):
        column = index % (columns + 1)
        row = index // (columns + 1)
        boundary = column == 0 or column == columns or row == 0 or row == rows
        if boundary or amplitude == 0:
            lattice.append((vertex['x'], vertex['y']))
            continue
        phase = time_seconds * motion
        wobble_x = (
            math.sin(phase * vertex['speed_x'] + vertex['phase_x']) * 0.68
            + math.sin(phase * 0.61 + vertex['phase_y']) * 0.32
        )
        wobble_y = (
            math.sin(phase * vertex['speed_y'] + vertex['phase_y']) * 0.68
            + math.cos(phase * 0.57 + vertex['phase_x']) * 0.32
        )
        lattice.append((vertex['x'] + wobble_x * amplitude, vertex['y'] + wobble_y * amplitude))

    phase = time_seconds / 12 * math.tau
    word_x = js_round((WIDTH - WORD_WIDTH) / 2 + 125 * math.sin(phase))
    word_y = js_round((HEIGHT - WORD_HEIGHT) / 2 + 120 * math.sin(phase * 2))

    horizontal = []
    vertical = []
    edges = []

    def slot_for_cell(row, column):
        source = (column + row) % 2
        copy = (column * 3 + row * 5 + (column ^ row)) % copies
        return source * copies + copy

    for row in range(rows + 1):
        row_edges = []
        for column in range(columns):
            edge = {
                'start': lattice[row * (columns + 1) + column],
                'end': lattice[row * (columns + 1) + column + 1],
                'points': [],
                'slots': ([slot_for_cell(row - 1, column)] if row > 0 else [])
                + ([slot_for_cell(row, column)] if row < rows else []),
            }
            row_edges.append(edge)
            edges.append(edge)
        horizontal.append(row_edges)
    for row in range(rows):
        row_edges = []
        for column in range(columns + 1):
            edge = {
                'start': lattice[row * (columns + 1) + column],
                'end': lattice[(row + 1) * (columns + 1) + column],
                'points': [],
                'slots': ([slot_for_cell(row, column - 1)] if column > 0 else [])
                + ([slot_for_cell(row, column)] if column < columns else []),
            }
            row_edges.append(edge)
            edges.append(edge)
        vertical.append(row_edges)

    max_edge_distance = (
        min(22, min(cell_width, cell_height) * 0.5)
        if edge_radius is None else edge_radius
    )
    attachments = []
    for local_point in points:
        text_point = (word_x + local_point[0], word_y + local_point[1])
        closest = None
        closest_distance = float('inf')
        closest_projection = 0
        for edge in edges:
            distance, projection = segment_distance(text_point, edge['start'], edge['end'])
            if projection < 0 or projection > 1:
                continue
            if distance > max_edge_distance * max_edge_distance:
                continue
            if distance < closest_distance:
                closest = edge
                closest_distance = distance
                closest_projection = projection
        if closest:
            closest['points'].append(text_point)
            projection = max(0, min(1, closest_projection))
            attachments.append({
                'point': text_point,
                'foot': (
                    closest['start'][0] + projection * (closest['end'][0] - closest['start'][0]),
                    closest['start'][1] + projection * (closest['end'][1] - closest['start'][1]),
                ),
                'edge': closest,
            })
        else:
            attachments.append(None)

    for edge in edges:
        dx = edge['end'][0] - edge['start'][0]
        dy = edge['end'][1] - edge['start'][1]
        length_squared = dx * dx + dy * dy
        edge['points'].sort(key=lambda point: 0 if length_squared == 0 else (
            (point[0] - edge['start'][0]) * dx + (point[1] - edge['start'][1]) * dy
        ) / length_squared)
        max_points = max(2, min(10, math.ceil(math.sqrt(length_squared) / 24)))
        if len(edge['points']) > max_points:
            edge['points'] = [
                edge['points'][0 if max_points == 1 else js_round(
                    index * (len(edge['points']) - 1) / (max_points - 1)
                )]
                for index in range(max_points)
            ]

    def edge_path(edge):
        dx = edge['end'][0] - edge['start'][0]
        dy = edge['end'][1] - edge['start'][1]
        length_squared = dx * dx + dy * dy
        length = math.sqrt(length_squared)
        if length == 0:
            return [edge['start']]
        result = [edge['start']]
        for text_point in edge['points']:
            if topology == 'lattice-tessellation':
                # Use the contour point as the actual mesh vertex. The
                # accepted face-bridge mode keeps its short before/after
                # detours; the sibling avoids turning dense points into teeth.
                result.append(text_point)
                continue
            projection = (
                (text_point[0] - edge['start'][0]) * dx
                + (text_point[1] - edge['start'][1]) * dy
            ) / length_squared
            projection = max(0, min(1, projection))
            half_span = min(12, max(5, length * 0.06))
            span = half_span / length
            before_projection = max(0, projection - span)
            after_projection = min(1, projection + span)
            before = (
                edge['start'][0] + before_projection * dx,
                edge['start'][1] + before_projection * dy,
            )
            after = (
                edge['start'][0] + after_projection * dx,
                edge['start'][1] + after_projection * dy,
            )
            result.extend([before, text_point, after])
        result.append(edge['end'])
        return result

    mask = Image.new('L', (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    for row in range(rows):
        for column in range(columns):
            top_left = lattice[row * (columns + 1) + column]
            top_right = lattice[row * (columns + 1) + column + 1]
            bottom_right = lattice[(row + 1) * (columns + 1) + column + 1]
            bottom_left = lattice[(row + 1) * (columns + 1) + column]
            slot = slot_for_cell(row, column)
            value = js_round(slot * 255 / (4 * copies - 1))
            polygon = [top_left] + edge_path(horizontal[row][column])
            polygon += edge_path(vertical[row][column + 1])
            polygon += list(reversed(edge_path(horizontal[row + 1][column])))
            polygon += list(reversed(edge_path(vertical[row][column])))
            draw.polygon(polygon, fill=value)

    tessellation_faces = []
    if topology == 'lattice-tessellation':
        offset = 0
        for contour in contours or []:
            contour_attachments = attachments[offset:offset + len(contour)]
            offset += len(contour)
            for index, first in enumerate(contour_attachments):
                second = contour_attachments[(index + 1) % len(contour_attachments)]
                if not first or not second:
                    continue
                if math.hypot(second['foot'][0] - first['foot'][0], second['foot'][1] - first['foot'][1]) > 80:
                    continue
                polygon = [first['foot'], second['foot'], second['point'], first['point']]
                nearby_slot = (first['edge']['slots'][0] if first['edge']['slots']
                               else second['edge']['slots'][0] if second['edge']['slots'] else 0)
                text_copy = nearby_slot % copies
                text_slot = 2 * copies + text_copy
                tessellation_faces.append((polygon, [text_slot, text_slot]))

        # Make the contour faces part of the lattice surface: remove their
        # footprints from the base cells before putting their two triangles
        # back with the discrete text-region IDs carried by the new faces.
        cutout = Image.new('L', (WIDTH, HEIGHT), 0)
        cutout_draw = ImageDraw.Draw(cutout)
        for polygon, _ in tessellation_faces:
            cutout_draw.polygon(polygon, fill=255)
        mask.paste(0, mask=cutout)

    # Every sampled contour segment is a pair of adjoining lattice faces. Its
    # endpoints are the inserted vertices; the other two vertices are the
    # nearest points on the original lattice edges. No point, line, or
    # separate text layer is rendered.
    if topology == 'contour-faces':
        inserted_face_value = js_round((2 * copies) * 255 / (4 * copies - 1))
        offset = 0
        for contour in contours or []:
            contour_attachments = attachments[offset:offset + len(contour)]
            offset += len(contour)
            for index, first in enumerate(contour_attachments):
                second = contour_attachments[(index + 1) % len(contour_attachments)]
                if not first or not second:
                    continue
                point_a = first['point']
                point_b = second['point']
                foot_a = first['foot']
                foot_b = second['foot']
                if math.hypot(foot_b[0] - foot_a[0], foot_b[1] - foot_a[1]) > 80:
                    continue
                draw.polygon([foot_a, foot_b, point_b, point_a], fill=inserted_face_value)
    elif topology == 'lattice-tessellation':
        for polygon, slots in tessellation_faces:
            draw.polygon(polygon[:3], fill=js_round(slots[0] * 255 / (4 * copies - 1)))
            draw.polygon([polygon[0], polygon[2], polygon[3]], fill=js_round(slots[1] * 255 / (4 * copies - 1)))

    elif topology == 'pure-lattice':
        # A single two-ID mesh replaces the base field for this frame.  The
        # word contour vertices and ordinary moving lattice vertices share
        # one triangulation; this is not an alpha/pixel paste over a grid.
        region = contour_region(contours or [], word_x, word_y).convert('L')
        moat = region.filter(ImageFilter.MaxFilter(21))
        mask = pure_lattice_mesh(
            lattice, contours or [], word_x, word_y, region, moat, columns, rows,
        )

    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density', type=int, default=96)
    parser.add_argument('--fluidity', type=float, default=0)
    parser.add_argument('--copies', type=int, default=1)
    parser.add_argument('--topology', choices=('contour-faces', 'edge-splits', 'lattice-tessellation', 'pure-lattice'),
                        default='contour-faces', help='point-face topology to probe')
    parser.add_argument('--seconds', type=float, default=SECONDS,
                        help='duration of the probe video (shorten this for geometry experiments)')
    parser.add_argument('--edge-radius', type=float, default=None,
                        help='maximum distance from a lattice edge for a text point (0 keeps only exact edges)')
    parser.add_argument('--out', type=Path, default=Path('/private/tmp/point-mask-probe.webm'))
    args = parser.parse_args()
    if args.topology == 'pure-lattice' and args.copies != 1:
        raise SystemExit('pure-lattice is a two-ID probe; use --copies 1.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    contours = text_contours()
    points = [point for contour in contours for point in contour]
    selected = {0, 3, 6, 9}
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libvpx-vp9', '-lossless', '1', '-b:v', '0', '-cpu-used', '4',
        # The grayscale values are clip IDs. Lossless luma preserves their
        # hard boundaries while yuv420 stays broadly browser-decodable.
        '-pix_fmt', 'yuv420p', str(args.out),
    ], stdin=subprocess.PIPE)
    try:
        for frame in range(max(1, math.ceil(FPS * args.seconds))):
            time_seconds = frame / FPS
            mask = make_frame(
                args.density, args.fluidity, args.copies, time_seconds, points, args.edge_radius,
                contours, args.topology,
            )
            if frame // FPS in selected and frame % FPS == 0:
                mask.save(args.out.with_name(f'{args.out.stem}-{frame // FPS}s.png'))
            encoder.stdin.write(mask.tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit('Point mask encoding failed.')
    print(f'{args.out}: {args.out.stat().st_size / 1024:.0f} KB')
    print(f'text points sampled: {len(points)}')
    print(f'topology: {args.topology}')


if __name__ == '__main__':
    main()
