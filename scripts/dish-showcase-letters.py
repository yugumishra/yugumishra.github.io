"""The rings of a dished letter top, for scripts/bake-showcase-cloth.py --dish.

Reads {"loops": [[[x, y], ...], ...], "levels": [a, b, ...], "simplify": P} on
stdin, the letters' outlines in canvas px (even-odd, so counters are holes),
and writes {"levels": [[loop, ...], ...]} to stdout: for each level, the
closed curves at that distance inside the letters, thinned to within P px.
Contours of one distance field are nested and never cross, so rings built
between them cannot fold. The bake runs it with the system's python3, which
has numpy and contourpy; Blender's own Python has neither.
"""

import json
import sys

import contourpy
import numpy as np

PAD = 4
INF = 1e20


def fill(loops, x0, y0, w, h):
    """Even-odd fill at pixel centres."""
    edges = np.concatenate([np.concatenate([np.asarray(l, float), np.roll(np.asarray(l, float), -1, axis=0)], axis=1) for l in loops])
    ex0, ey0, ex1, ey1 = edges.T
    xs = np.arange(w) + x0 + 0.5
    mask = np.zeros((h, w), bool)
    for r in range(h):
        y = y0 + r + 0.5
        hit = (ey0 > y) != (ey1 > y)
        if hit.any():
            cross = np.sort(ex0[hit] + (y - ey0[hit]) / (ey1[hit] - ey0[hit]) * (ex1[hit] - ex0[hit]))
            mask[r] = (np.searchsorted(cross, xs) % 2) == 1
    return mask


def edt_1d(f):
    """Felzenszwalb and Huttenlocher: squared distance transform of a row."""
    n = len(f)
    d = np.empty(n)
    v = np.zeros(n, int)
    z = np.empty(n + 1)
    k = 0
    z[0], z[1] = -INF, INF
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def inside_distance(mask):
    """Euclidean distance, in px, from each inside pixel to the nearest outside."""
    f = np.where(mask, INF, 0.0)
    for r in range(f.shape[0]):
        f[r] = edt_1d(f[r])
    for c in range(f.shape[1]):
        f[:, c] = edt_1d(f[:, c])
    return np.sqrt(f)


def simplify(loop, tolerance):
    pts = np.asarray(loop, float)
    if tolerance <= 0 or len(pts) < 8:
        return pts

    def chain(c):
        keep = np.zeros(len(c), bool)
        keep[0] = keep[-1] = True
        stack = [(0, len(c) - 1)]
        while stack:
            a, b = stack.pop()
            if b <= a + 1:
                continue
            seg = c[b] - c[a]
            length = np.hypot(*seg)
            rel = c[a + 1:b] - c[a]
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length if length > 0 else np.hypot(rel[:, 0], rel[:, 1])
            i = int(np.argmax(dist))
            if dist[i] > tolerance:
                keep[a + 1 + i] = True
                stack += [(a, a + 1 + i), (a + 1 + i, b)]
        return c[keep]

    far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    return np.vstack([chain(pts[:far + 1])[:-1], chain(np.vstack([pts[far:], pts[:1]]))[:-1]])


def main():
    job = json.load(sys.stdin)
    loops, levels, tolerance = job["loops"], job["levels"], float(job.get("simplify", 1.0))
    pts = np.concatenate([np.asarray(l, float) for l in loops])
    x0 = int(np.floor(pts[:, 0].min())) - PAD
    y0 = int(np.floor(pts[:, 1].min())) - PAD
    w = int(np.ceil(pts[:, 0].max())) + PAD - x0
    h = int(np.ceil(pts[:, 1].max())) + PAD - y0
    # Distance from the pixel centres to the outside, less half a pixel so
    # the edge itself is at 0.
    d = inside_distance(fill(loops, x0, y0, w, h)) - 0.5
    gen = contourpy.contour_generator(z=d, line_type=contourpy.LineType.Separate)
    out = []
    for level in levels:
        rings = []
        for line in gen.lines(level):
            if len(line) < 6:
                continue
            ring = simplify(np.asarray(line)[:-1] if np.allclose(line[0], line[-1]) else np.asarray(line), tolerance)
            if len(ring) >= 3:
                rings.append([[round(float(i) + x0 + 0.5, 2), round(float(j) + y0 + 0.5, 2)] for i, j in ring])
        out.append(rings)
    json.dump({"levels": out}, sys.stdout)


if __name__ == "__main__":
    main()
