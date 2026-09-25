"""Round and join the floor's letter holes for scripts/bake-showcase-cloth.py.

Reads {"loops": [[[x, y], ...], ...], "join": R, "smooth": S, "union": U} on
stdin, in canvas px, and writes {"loops": [...]} to stdout. The loops are
filled at 1 px, even-odd, or with "union" each on its own and merged, then
closed by a disk of radius R (holes less than 2R apart run together, and
concave corners fill in to radius R), blurred by a Gaussian of S px and
traced again at the half level (every corner rounds off by about S).
The bake runs it with the system's python3, which has numpy and contourpy
(the glyph export needs them too); Blender's own Python has neither.
"""

import json
import sys

import contourpy
import numpy as np

PAD = 64


def fill(loops, x0, y0, w, h):
    """Even-odd fill at pixel centres."""
    mask = np.zeros((h, w), bool)
    edges = []
    for loop in loops:
        p = np.asarray(loop, float)
        q = np.roll(p, -1, axis=0)
        edges.append(np.concatenate([p, q], axis=1))
    edges = np.concatenate(edges)
    ex0, ey0, ex1, ey1 = edges.T
    xs = np.arange(w) + x0 + 0.5
    for r in range(h):
        y = y0 + r + 0.5
        hit = (ey0 > y) != (ey1 > y)
        if not hit.any():
            continue
        cross = np.sort(ex0[hit] + (y - ey0[hit]) / (ey1[hit] - ey0[hit]) * (ex1[hit] - ex0[hit]))
        inside = (np.searchsorted(cross, xs) % 2) == 1
        mask[r] = inside
    return mask


def convolve(field, kernel):
    """Same-size convolution by FFT, the kernel centred."""
    h, w = field.shape
    kh, kw = kernel.shape
    shape = (h + kh, w + kw)
    out = np.fft.irfft2(np.fft.rfft2(field, shape) * np.fft.rfft2(kernel, shape), shape)
    return out[kh // 2:kh // 2 + h, kw // 2:kw // 2 + w]


def disk(r):
    n = int(np.ceil(r))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    return (x * x + y * y <= r * r).astype(float)


def main():
    job = json.load(sys.stdin)
    loops, join, smooth = job["loops"], float(job.get("join", 0)), float(job.get("smooth", 0))
    union = bool(job.get("union", False))
    pts = np.concatenate([np.asarray(loop, float) for loop in loops])
    margin = PAD + join + 3 * smooth
    x0 = int(np.floor(pts[:, 0].min() - margin))
    y0 = int(np.floor(pts[:, 1].min() - margin))
    w = int(np.ceil(pts[:, 0].max() + margin)) - x0
    h = int(np.ceil(pts[:, 1].max() + margin)) - y0
    if union:
        mask = np.zeros((h, w), bool)
        for loop in loops:
            mask |= fill([loop], x0, y0, w, h)
        field = mask.astype(float)
    else:
        field = fill(loops, x0, y0, w, h).astype(float)
    if join > 0:
        k = disk(join)
        grown = convolve(field, k) > 0.5
        field = (convolve(grown.astype(float), k) > k.sum() - 0.5).astype(float)
    if smooth > 0:
        n = int(np.ceil(3 * smooth))
        g = np.exp(-(np.arange(-n, n + 1) ** 2) / (2 * smooth * smooth))
        g /= g.sum()
        field = convolve(field, np.outer(g, g))
    lines = contourpy.contour_generator(z=field, line_type=contourpy.LineType.Separate).lines(0.5)
    out = [[[round(float(i) + x0 + 0.5, 2), round(float(j) + y0 + 0.5, 2)] for i, j in line] for line in lines if len(line) >= 8]
    json.dump({"loops": out}, sys.stdout)


if __name__ == "__main__":
    main()
