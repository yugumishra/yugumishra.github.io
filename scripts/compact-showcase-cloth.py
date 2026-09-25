"""Compact a cloth bake's mesh export (bake-showcase-cloth.py --mesh / --lift)
for the page.

  python3 scripts/compact-showcase-cloth.py IN.json IN.bin OUT.json OUT.bin --glyphs scripts/showcase-glyphs/WORD.json
      [--fine 10 --coarse 30 --tol 1.0]

The bake's sheet is dense for the simulation's sake; the page needs far less.
1. Resample: the sheet at rest is a plane, so every point of a new rest grid
   lies in one of the bake's triangles and moves with it, by its barycentric
   weights. The grid is --fine px (10) over the word and --coarse px (30)
   elsewhere, its rows and columns placed independently.
2. Frames: frames the page can interpolate from their neighbours to within
   --tol px (1.0), at the 99th percentile of the points in the frame, are
   dropped, greedily, keeping the lift's phase frames.
3. Quantise: positions to 1/2 px, heights to 1/200 unit.
4. Encode: each point's track predicted from its last two kept frames, the
   residuals planar (every x, then y, then z, a frame at a time), int16, and
   the whole file zlib-compressed. The page decompresses it with the
   browser's DecompressionStream('deflate').
"""

import json
import sys
import zlib

import numpy as np

ARGS = sys.argv[1:]
IN_META, IN_DATA, OUT_META, OUT_DATA = ARGS[:4]


def arg(name, default):
    return float(ARGS[ARGS.index(name) + 1]) if name in ARGS else default


FINE, COARSE, TOL = arg("--fine", 10.0), arg("--coarse", 30.0), arg("--tol", 1.0)
XY_Q, Z_Q = arg("--xyq", 2.0), arg("--zq", 200.0)  # steps a px (2: half px), a unit


def axis(lo, hi, band_lo, band_hi):
    """Stops from lo to hi: COARSE outside [band_lo, band_hi], FINE inside."""
    stops = [lo]
    while stops[-1] < hi - 1e-6:
        x = stops[-1]
        step = FINE if band_lo - 1e-6 <= x < band_hi - 1e-6 else COARSE
        nxt = x + step
        # land exactly on the band's edges
        if x < band_lo - 1e-6 < nxt:
            nxt = band_lo
        elif x < band_hi - 1e-6 < nxt:
            nxt = band_hi
        stops.append(min(nxt, hi))
    return np.array(stops)


def main():
    meta = json.load(open(IN_META))
    buf = open(IN_DATA, "rb").read()
    P, T, frames = meta["points"], meta["triangles"], meta["frames"]
    F = len(frames)
    xy, zs = meta["offsetScale"], meta["heightScale"]
    rest = np.frombuffer(buf, "<i2", P * 2, 0).reshape(-1, 2) * xy
    tris = np.frombuffer(buf, "<u2", T * 3, P * 4).reshape(-1, 3).astype(np.int64)
    pos = np.frombuffer(buf, "<i2", F * P * 3, P * 4 + T * 6).reshape(F, P, 3).astype(np.float64)
    pos[:, :, :2] *= xy
    pos[:, :, 2] *= zs

    # 1. the new rest grid, fine over the word
    glyph = json.load(open(ARGS[ARGS.index("--glyphs") + 1]))
    lx = [x for loop in glyph["letters"] for x, _ in loop]
    ly = [y for loop in glyph["letters"] for _, y in loop]
    pad = 60
    xs = axis(rest[:, 0].min(), rest[:, 0].max(), min(lx) - pad, max(lx) + pad)
    ys = axis(rest[:, 1].min(), rest[:, 1].max(), min(ly) - pad, max(ly) + pad)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    n = len(grid)
    owner = np.full(n, -1, np.int64)
    weights = np.zeros((n, 3))
    # every grid point in each rest triangle's bounding box, then barycentrics
    a, b, c = rest[tris[:, 0]], rest[tris[:, 1]], rest[tris[:, 2]]
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    i0 = np.searchsorted(xs, lo[:, 0] - 1e-6)
    i1 = np.searchsorted(xs, hi[:, 0] + 1e-6)
    j0 = np.searchsorted(ys, lo[:, 1] - 1e-6)
    j1 = np.searchsorted(ys, hi[:, 1] + 1e-6)
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    for di in range(int((i1 - i0).max()) + 1):
        for dj in range(int((j1 - j0).max()) + 1):
            sel = (i0 + di < i1) & (j0 + dj < j1) & (np.abs(det) > 1e-12)
            if not sel.any():
                continue
            t = np.nonzero(sel)[0]
            ii, jj = i0[t] + di, j0[t] + dj
            px, py = xs[ii], ys[jj]
            w0 = ((b[t, 1] - c[t, 1]) * (px - c[t, 0]) + (c[t, 0] - b[t, 0]) * (py - c[t, 1])) / det[t]
            w1 = ((c[t, 1] - a[t, 1]) * (px - c[t, 0]) + (a[t, 0] - c[t, 0]) * (py - c[t, 1])) / det[t]
            w2 = 1 - w0 - w1
            ok = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            k = jj[ok] * len(xs) + ii[ok]
            fresh = owner[k] < 0
            owner[k[fresh]] = t[ok][fresh]
            weights[k[fresh]] = np.stack([w0[ok], w1[ok], w2[ok]], axis=1)[fresh]
    missing = int((owner < 0).sum())
    if missing:
        # a grid point off the bake's sheet by a rounding hair: its nearest
        # rest point stands in
        for k in np.nonzero(owner < 0)[0]:
            nearest = int(np.argmin(((rest - grid[k]) ** 2).sum(axis=1)))
            t = int(np.nonzero((tris == nearest).any(axis=1))[0][0])
            owner[k] = t
            weights[k] = [1.0 if v == nearest else 0.0 for v in tris[t]]
    corners = tris[owner]
    new = np.einsum("nk,fnkc->fnc", weights, pos[:, corners])
    cols, rows = len(xs), len(ys)
    quads = [(j * cols + i, j * cols + i + 1, (j + 1) * cols + i + 1, (j + 1) * cols + i) for j in range(rows - 1) for i in range(cols - 1)]
    new_tris = np.array([(q[0], q[1], q[2]) for q in quads] + [(q[0], q[2], q[3]) for q in quads], np.int64)
    if n > 65535:
        raise SystemExit("%d points will not index as uint16" % n)

    # 2. drop frames the page can interpolate, keeping the phase frames
    keep_frames = {0, F - 1}
    lift = meta.get("lift") or {}
    for key in ("growEnd", "slideStart"):
        if key in lift:
            keep_frames.add(int(np.argmin([abs(f - lift[key]) for f in frames])))
    kept = list(range(F))

    def error(i):
        # the error if kept[i] is dropped: interpolate from its neighbours
        p, q, r = kept[i - 1], kept[i], kept[i + 1]
        u = (frames[q] - frames[p]) / (frames[r] - frames[p])
        guess = new[p] * (1 - u) + new[r] * u
        d = guess - new[q]
        # only what is on screen then counts, and a stray flap does not
        # decide it: the 99th percentile of the points in the frame
        on = (new[q][:, 0] > -40) & (new[q][:, 0] < 1320) & (new[q][:, 1] > -40) & (new[q][:, 1] < 760)
        if not on.any():
            return 0.0
        return float(np.percentile(np.sqrt(d[on, 0] ** 2 + d[on, 1] ** 2 + (d[on, 2] * 100) ** 2), 99))

    while True:
        best, at = None, None
        for i in range(1, len(kept) - 1):
            if kept[i] in keep_frames:
                continue
            e = error(i)
            if best is None or e < best:
                best, at = e, i
        if best is None or best > TOL:
            break
        del kept[at]
    track = new[kept]

    # 3. quantise, 4. predict, planar, deflate
    q = np.empty(track.shape, np.int64)
    q[:, :, :2] = np.round(track[:, :, :2] * XY_Q)
    q[:, :, 2] = np.round(track[:, :, 2] * Z_Q)
    res = q.copy()
    if len(q) > 1:
        res[1] = q[1] - q[0]
    if len(q) > 2:
        res[2:] = q[2:] - 2 * q[1:-1] + q[:-2]
    if np.abs(res).max() > 32767 or np.abs(np.round(grid * XY_Q)).max() > 32767:
        raise SystemExit("a value will not fit int16")
    body = [np.round(grid * XY_Q).astype("<i2").tobytes(), new_tris.astype("<u2").tobytes(),
            np.transpose(res, (0, 2, 1)).astype("<i2").tobytes()]
    packed = zlib.compress(b"".join(body), 9)
    open(OUT_DATA, "wb").write(packed)
    out = dict(meta)
    out.update({
        "version": 2,
        "points": n,
        "triangles": len(new_tris),
        "frames": [frames[i] for i in kept],
        "offsetScale": 1 / XY_Q,
        "heightScale": 1 / Z_Q,
        "encoding": "deflate-planar-accel",
        "grid": {"columns": cols, "rows": rows, "fine": FINE, "coarse": COARSE},
        "layout": "zlib of: int16 rest x,y per point; uint16 triangles; per kept frame, planar int16 x then y then z per point, as residuals from the last two frames' linear prediction",
    })
    json.dump(out, open(OUT_META, "w"), ensure_ascii=False, separators=(",", ":"))
    print("%s: %d -> %d points, %d -> %d frames, %.1f MB -> %.2f MB" % (meta["word"].replace("\n", " "), P, n, F, len(kept), len(buf) / 1e6, len(packed) / 1e6))


if __name__ == "__main__":
    main()
