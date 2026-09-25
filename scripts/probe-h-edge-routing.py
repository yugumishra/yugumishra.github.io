#!/usr/bin/env python3
"""
probe-h-edge-routing.py  --  Agent H

Route the glyph outline THROUGH the lattice edge graph.

The only geometry that exists in the output is a single fine quad lattice.
The letter outline is a closed cycle of ordinary lattice edges, chosen by
min-cost routing on the lattice graph (edge cost = distance from the true
outline + alignment penalty + reuse penalty).  Only vertices on the routed
cycles are snapped; everything else is relaxed so the distortion bleeds out
into the surrounding field.

Nothing is drawn, pasted, stamped or overlaid.  No stroke.  No second colour.
"""
import os, sys, math, json, time, argparse, subprocess, shutil
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1280, 720
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
OUT = os.environ.get("AGENT_H_OUT",
      "/private/tmp/claude-501/-Users-me-Desktop-dev-yugumishra-github-io/"
      "ea9e4c74-aeb0-4767-b13b-b99660eec2a7/scratchpad/agent-h")
FRAMES = os.path.join(OUT, "frames")
os.makedirs(OUT, exist_ok=True); os.makedirs(FRAMES, exist_ok=True)

INK = (34, 36, 40)
PAPER = (247, 246, 243)


# ----------------------------------------------------------------------------
# marching squares -> closed polylines  (vectorised)
# ----------------------------------------------------------------------------
# code = TL*8 + TR*4 + BR*2 + BL*1 ; edges T=0 R=1 B=2 L=3
_MS = {
    1: [(3, 2)], 2: [(2, 1)], 3: [(3, 1)], 4: [(1, 0)],
    5: [(3, 0), (1, 2)], 6: [(2, 0)], 7: [(3, 0)], 8: [(0, 3)],
    9: [(0, 2)], 10: [(0, 3), (2, 1)], 11: [(0, 1)], 12: [(1, 3)],
    13: [(1, 2)], 14: [(2, 3)], 15: [], 0: [],
}


def marching_squares(F, level=0.5):
    F = F.astype(np.float64)
    h, w = F.shape
    tl = F[:-1, :-1]; tr = F[:-1, 1:]; br = F[1:, 1:]; bl = F[1:, :-1]
    code = ((tl >= level).astype(np.uint8) << 3 | (tr >= level).astype(np.uint8) << 2
            | (br >= level).astype(np.uint8) << 1 | (bl >= level).astype(np.uint8))
    rr, cc = np.mgrid[0:h - 1, 0:w - 1]

    def interp(a, b):
        d = b - a
        d = np.where(np.abs(d) < 1e-12, 1e-12, d)
        return np.clip((level - a) / d, 0.0, 1.0)

    # edge points, per-cell
    tx = cc + interp(tl, tr); ty = rr.astype(np.float64)
    rx = (cc + 1).astype(np.float64); ry = rr + interp(tr, br)
    bx = cc + interp(bl, br); by = (rr + 1).astype(np.float64)
    lx = cc.astype(np.float64); ly = rr + interp(tl, bl)
    EX = np.stack([tx, rx, bx, lx]); EY = np.stack([ty, ry, by, ly])

    segs = []
    for c, pairs in _MS.items():
        if not pairs:
            continue
        sel = code == c
        if not sel.any():
            continue
        for (e0, e1) in pairs:
            x0 = EX[e0][sel]; y0 = EY[e0][sel]
            x1 = EX[e1][sel]; y1 = EY[e1][sel]
            segs.append(np.stack([x0, y0, x1, y1], axis=1))
    if not segs:
        return []
    S = np.concatenate(segs, axis=0)

    # link by exact endpoint hashing (shared-edge interpolants are bit-identical)
    def key(x, y):
        return (round(float(x), 9), round(float(y), 9))
    start = {}
    for i in range(S.shape[0]):
        start.setdefault(key(S[i, 0], S[i, 1]), []).append(i)
    used = np.zeros(S.shape[0], bool)
    polys = []
    for i0 in range(S.shape[0]):
        if used[i0]:
            continue
        chain = [(S[i0, 0], S[i0, 1])]
        i = i0
        while True:
            used[i] = True
            chain.append((S[i, 2], S[i, 3]))
            k = key(S[i, 2], S[i, 3])
            nxt = None
            for j in start.get(k, ()):
                if not used[j]:
                    nxt = j; break
            if nxt is None:
                break
            i = nxt
            if i == i0:
                break
        P = np.array(chain, float)
        if len(P) > 8:
            polys.append(P)
    return polys


def poly_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def resample(P, step):
    """uniform arc-length resample of a closed polyline"""
    Q = np.vstack([P, P[:1]]) if not np.allclose(P[0], P[-1]) else P
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    L = d[-1]
    n = max(12, int(round(L / step)))
    t = np.linspace(0, L, n, endpoint=False)
    x = np.interp(t, d, Q[:, 0]); y = np.interp(t, d, Q[:, 1])
    return np.stack([x, y], 1), L


def smooth_closed(P, k=2, iters=1):
    for _ in range(iters):
        acc = np.zeros_like(P); wsum = 0.0
        for o in range(-k, k + 1):
            w = 1.0 / (1 + abs(o))
            acc += w * np.roll(P, o, axis=0); wsum += w
        P = acc / wsum
    return P


def point_in_polys(pts, polys):
    """even-odd ray cast; pts (N,2)"""
    inside = np.zeros(len(pts), bool)
    px, py = pts[:, 0], pts[:, 1]
    for P in polys:
        x1 = P[:, 0]; y1 = P[:, 1]
        x2 = np.roll(x1, -1); y2 = np.roll(y1, -1)
        for i in range(len(x1)):
            a = (y1[i] > py) != (y2[i] > py)
            if not a.any():
                continue
            xin = (x2[i] - x1[i]) * (py - y1[i]) / (y2[i] - y1[i] + 1e-12) + x1[i]
            inside ^= (a & (px < xin))
    return inside


# ----------------------------------------------------------------------------
# glyph contours
# ----------------------------------------------------------------------------
def glyph_contours(word, cap_h=213.0, ss=3, ctr=(W / 2, H / 2), resample_px=3.0):
    # find pixel size that yields the requested cap height
    size = int(cap_h * 1.35)
    for _ in range(24):
        f = ImageFont.truetype(FONT_PATH, size)
        bb = f.getbbox(word)
        hgt = bb[3] - bb[1]
        if abs(hgt - cap_h) < 1.0:
            break
        size = max(8, int(round(size * cap_h / max(hgt, 1))))
    f = ImageFont.truetype(FONT_PATH, size)
    bb = f.getbbox(word)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    # if too wide for canvas, shrink
    maxw = W * 0.90
    if tw > maxw:
        size = int(size * maxw / tw)
        f = ImageFont.truetype(FONT_PATH, size)
        bb = f.getbbox(word); tw, th = bb[2] - bb[0], bb[3] - bb[1]

    img = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(img)
    ox = ctr[0] * ss - tw * ss / 2 - bb[0] * ss
    oy = ctr[1] * ss - th * ss / 2 - bb[1] * ss
    fs = ImageFont.truetype(FONT_PATH, size * ss)
    bbs = fs.getbbox(word)
    ox = ctr[0] * ss - (bbs[2] - bbs[0]) / 2 - bbs[0]
    oy = ctr[1] * ss - (bbs[3] - bbs[1]) / 2 - bbs[1]
    d.text((ox, oy), word, fill=255, font=fs)
    img = img.filter(ImageFilter.GaussianBlur(ss * 0.9))
    F = np.asarray(img, np.float32) / 255.0

    polys = marching_squares(F, 0.5)
    out = []
    for P in polys:
        P = P[:, ::-1] if False else P          # already (x,y)
        P = P / ss
        A = poly_area(P)
        if abs(A) < 60:
            continue
        Pr, L = resample(P, resample_px)
        Pr = smooth_closed(Pr, k=2, iters=2)
        out.append({"pts": Pr, "area": A, "len": L})
    # orientation: outer = larger |area| and not contained in another
    for c in out:
        c["is_hole"] = False
    for i, c in enumerate(out):
        cen = c["pts"].mean(0, keepdims=True)
        for j, o in enumerate(out):
            if i == j:
                continue
            if abs(o["area"]) > abs(c["area"]):
                if point_in_polys(cen, [o["pts"]])[0]:
                    c["is_hole"] = not c["is_hole"]
    out.sort(key=lambda c: -abs(c["area"]))
    # stroke width estimate = 2 * max distance-transform inside
    mask = (F >= 0.5)
    dt = ndimage.distance_transform_edt(mask) / ss
    return out, float(np.percentile(dt[mask], 99.0) * 2.0), size, mask


# ----------------------------------------------------------------------------
# fields
# ----------------------------------------------------------------------------
def contour_field(pts):
    """distance to this contour + index of nearest contour sample, on the canvas grid"""
    idximg = np.full((H, W), -1, np.int32)
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    # densify so no gaps
    P2 = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(P2, axis=0), axis=1)
    reps = np.maximum(1, np.ceil(seg / 0.5).astype(int))
    xs_l = []; ys_l = []; id_l = []
    for i in range(len(pts)):
        n = reps[i]
        t = np.arange(n) / n
        xs_l.append(P2[i, 0] + t * (P2[i + 1, 0] - P2[i, 0]))
        ys_l.append(P2[i, 1] + t * (P2[i + 1, 1] - P2[i, 1]))
        id_l.append(np.full(n, i))
    X = np.clip(np.round(np.concatenate(xs_l)).astype(int), 0, W - 1)
    Y = np.clip(np.round(np.concatenate(ys_l)).astype(int), 0, H - 1)
    I = np.concatenate(id_l)
    idximg[Y, X] = I
    bg = idximg < 0
    D, (iy, ix) = ndimage.distance_transform_edt(bg, return_indices=True)
    NEAR = idximg[iy, ix]
    return D.astype(np.float32), NEAR.astype(np.int32)


def union_field(contours):
    """distance + exact nearest point on the union of all contours"""
    ALL = np.vstack([c["pts"] for c in contours])
    idximg = np.full((H, W), -1, np.int32)
    P2 = []
    gid = []
    base = 0
    for c in contours:
        pts = c["pts"]; n = len(pts)
        Q = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        reps = np.maximum(1, np.ceil(seg / 0.5).astype(int))
        for i in range(n):
            t = np.arange(reps[i]) / reps[i]
            P2.append(np.stack([Q[i, 0] + t * (Q[i + 1, 0] - Q[i, 0]),
                                Q[i, 1] + t * (Q[i + 1, 1] - Q[i, 1])], 1))
            gid.append(np.full(reps[i], base + i))
        base += n
    P2 = np.concatenate(P2); gid = np.concatenate(gid)
    X = np.clip(np.round(P2[:, 0]).astype(int), 0, W - 1)
    Y = np.clip(np.round(P2[:, 1]).astype(int), 0, H - 1)
    idximg[Y, X] = gid
    D, (iy, ix) = ndimage.distance_transform_edt(idximg < 0, return_indices=True)
    NEAR = idximg[iy, ix]
    return D.astype(np.float32), NEAR.astype(np.int32), ALL


def ridge(P, free, D, NEAR, ALL, cell, band_cells=2.2, gamma=1.7, floor=0.18):
    """
    Compression band on the routed outline.

    Re-map every free vertex's distance d to the outline through a monotone
    f(d) <= d, moving it along its own outline normal.  Lattice lines bunch
    onto the letter, thin out just beyond it, and the effect decays over
    `band_cells`.  Monotone in d, so it cannot invert a cell.
    """
    if band_cells <= 0:
        return P
    band = band_cells * cell
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)
    d = D[yi, xi].astype(np.float64)
    np_ = ALL[np.clip(NEAR[yi, xi], 0, len(ALL) - 1)]
    v = P - np_
    L = np.linalg.norm(v, axis=1)
    use = free & (L > 1e-6) & (d < band * 1.6)
    if not use.any():
        return P
    dd = L[use]
    f = band * np.power(np.clip(dd / band, 0, 1), gamma)
    f = np.where(dd >= band, dd, np.maximum(f, floor * dd))
    P[use] = np_[use] + v[use] * (f / dd)[:, None]
    return P


def tangents(pts):
    T = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    n = np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    return T / n


# ----------------------------------------------------------------------------
# lattice
# ----------------------------------------------------------------------------
class Lattice:
    def __init__(self, nx, ny, margin=2):
        self.nx, self.ny, self.m = nx, ny, margin
        self.cw = W / nx; self.ch = H / ny
        self.NX = nx + 1 + 2 * margin       # vertices across
        self.NY = ny + 1 + 2 * margin
        gx = (np.arange(self.NX) - margin) * self.cw
        gy = (np.arange(self.NY) - margin) * self.ch
        GX, GY = np.meshgrid(gx, gy)
        self.P0 = np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float64)
        self.NV = self.P0.shape[0]
        idx = np.arange(self.NV).reshape(self.NY, self.NX)
        self.idx = idx
        eh = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)
        ev = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1)
        self.E = np.vstack([eh, ev]).astype(np.int32)
        # quads
        self.Q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
                           idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
        self.cell = 0.5 * (self.cw + self.ch)

    def border_mask(self):
        b = np.zeros((self.NY, self.NX), bool)
        b[0, :] = b[-1, :] = True; b[:, 0] = b[:, -1] = True
        return b.ravel()


MODES = [(1.0, 0.7, 1, 0.13), (1.7, 1.3, -1, 0.61), (2.3, 2.9, 2, 0.29),
         (3.7, 2.1, -2, 0.87), (5.1, 4.3, 1, 0.45)]


def wobbled(lat, t, T, amp):
    if amp <= 0:
        return lat.P0.copy()
    x = lat.P0[:, 0] / W; y = lat.P0[:, 1] / H
    dx = np.zeros(lat.NV); dy = np.zeros(lat.NV)
    for (fx, fy, ft, ph) in MODES:
        dx += np.sin(2 * np.pi * (fx * x + fy * y + ft * t / T + ph))
        dy += np.cos(2 * np.pi * (fy * x + fx * y + ft * t / T + ph * 1.7))
    s = amp * lat.cell / len(MODES)
    P = lat.P0.copy()
    P[:, 0] += s * dx; P[:, 1] += s * dy
    return P


def sample(field, P):
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)
    return field[yi, xi]


# ----------------------------------------------------------------------------
# ROUTING  --  the core of agent H
# ----------------------------------------------------------------------------
def edge_costs(lat, P, D, NEAR, TAN, lam=3.0, pw=1.6, mu=1.4):
    """cost of every lattice edge for one contour"""
    a = P[lat.E[:, 0]]; b = P[lat.E[:, 1]]
    mid = 0.5 * (a + b)
    d = b - a
    L = np.linalg.norm(d, axis=1) + 1e-9
    dh = d / L[:, None]
    Dm = sample(D, mid)
    ni = sample(NEAR, mid)
    tg = TAN[np.clip(ni, 0, len(TAN) - 1)]
    align = 1.0 - np.abs(np.sum(dh * tg, axis=1))          # 0 aligned, 1 perpendicular
    prox = np.exp(-Dm / (0.9 * lat.cell))                   # alignment only matters near
    dn = Dm / lat.cell
    c = L / lat.cell * (1.0 + lam * dn ** pw) * (1.0 + mu * align * prox)
    return c


def route_contour(lat, P, contour, D, NEAR, TAN, anchor_mult=1.15,
                  reuse_pen=14.0, lam=3.0, mu=1.4, claim_radius=1.6):
    """
    Min-cost closed cycle of lattice edges that follows `contour`.
    Anchors sampled along the true contour in order pin the homotopy class
    (so counters can never bridge to the outer contour); Dijkstra on the
    lattice graph picks the actual edges between them.
    """
    pts = contour["pts"]
    base = edge_costs(lat, P, D, NEAR, TAN, lam=lam, mu=mu)
    step = max(anchor_mult * lat.cell, 4.0)
    A, L = resample(pts, step)
    if len(A) < 4:
        A, L = resample(pts, max(L / 8.0, 2.0))
    corners = corner_points(pts, lat.cell)
    # merge corners into the uniform anchor ring, in contour order
    A, iscorner = merge_anchors(A, corners, pts)
    # anchor -> nearest lattice vertex, with EXCLUSIVE claims.
    # Two anchors on opposite walls of a narrow aperture would otherwise grab the
    # same vertex and the aperture pinches shut; forcing distinct vertices opens
    # it to exactly one cell instead, which is what keeps e's throat and 3's
    # bowls alive at coarse densities.
    tree = cKDTree(P)
    kq = min(16, len(P))
    dists, cand = tree.query(A, k=kq)
    if kq == 1:
        dists = dists[:, None]; cand = cand[:, None]
    claimed = set()
    av = []
    limit = claim_radius * lat.cell
    for i in range(len(A)):
        pick = int(cand[i, 0])
        prev = av[-1] if av else -1
        for j in range(kq):
            v = int(cand[i, j])
            if v == prev:
                pick = v; break                    # consecutive repeat is fine
            if v not in claimed and dists[i, j] <= limit:
                pick = v; break
        av.append(pick); claimed.add(pick)
    # dedupe consecutive repeats; a corner wins the shared vertex
    seq = []; seqpt = []
    for k, v in enumerate(av):
        if seq and v == seq[-1]:
            if iscorner[k] and seqpt[-1] is None:
                seqpt[-1] = A[k]
            continue
        seq.append(v); seqpt.append(A[k] if iscorner[k] else None)
    while len(seq) > 2 and seq[0] == seq[-1]:
        seq.pop(); seqpt.pop()
    if len(seq) < 3:
        return None, 0.0, {}
    forced = {seq[i]: seqpt[i] for i in range(len(seq)) if seqpt[i] is not None}

    R, C = lat.E[:, 0], lat.E[:, 1]
    rows = np.concatenate([R, C]); cols = np.concatenate([C, R])
    used = np.zeros(lat.NV, np.float64)
    walk = []
    total = 0.0
    for k in range(len(seq)):
        s = seq[k]; g = seq[(k + 1) % len(seq)]
        pen = 1.0 + reuse_pen * np.maximum(used[R], used[C])
        w = base * pen
        # never penalise the two endpoints of this leg
        M = csr_matrix((np.concatenate([w, w]), (rows, cols)), shape=(lat.NV, lat.NV))
        dist, pred = dijkstra(M, directed=False, indices=s, return_predecessors=True)
        if not np.isfinite(dist[g]):
            return None, 0.0, {}
        path = [g]
        while path[-1] != s:
            p = pred[path[-1]]
            if p < 0:
                return None, 0.0, {}
            path.append(int(p))
        path.reverse()
        total += dist[g]
        for v in path[:-1]:
            walk.append(v)
        for v in path:
            used[v] = 1.0
    # walk is a closed vertex walk; excise short backtracks to make it simple
    cyc = excise_loops(walk, max_excise=4)
    live = set(cyc)
    forced = {v: q for v, q in forced.items() if v in live}
    return cyc, total, forced


def corner_points(pts, cell, ang_thresh=0.55):
    """
    High-turning points of the contour.  These are what a coarse lattice loses
    first, so they get their own anchors and an exact snap target.
    """
    n = len(pts)
    k = max(2, int(round(0.45 * cell / max(1e-6, np.linalg.norm(pts[1] - pts[0])))))
    k = min(k, max(2, n // 8))
    a = np.roll(pts, k, axis=0) - pts
    b = np.roll(pts, -k, axis=0) - pts
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    turn = np.pi - np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1))
    out = []
    for i in range(n):
        if turn[i] < ang_thresh:
            continue
        w = [(i + o) % n for o in range(-k, k + 1)]
        if turn[i] >= turn[w].max() - 1e-9:
            out.append(i)
    # thin out near-duplicates
    keep = []
    for i in out:
        if all(min(abs(i - j), n - abs(i - j)) > k for j in keep):
            keep.append(i)
    if len(keep) > 1 and min(abs(keep[0] - keep[-1]), n - abs(keep[0] - keep[-1])) <= k:
        keep.pop()
    return pts[keep] if keep else np.zeros((0, 2))


def merge_anchors(A, corners, pts):
    """insert corner points into the uniform anchor ring, keeping contour order"""
    if len(corners) == 0:
        return A, np.zeros(len(A), bool)
    def param(q):
        d = (pts[:, 0] - q[0]) ** 2 + (pts[:, 1] - q[1]) ** 2
        return int(np.argmin(d))
    items = [(param(a), a, False) for a in A] + [(param(c), c, True) for c in corners]
    items.sort(key=lambda z: z[0])
    # a corner absorbs any uniform anchor sitting on the same contour sample
    out = []
    for t, q, isc in items:
        if out and out[-1][0] == t:
            if isc:
                out[-1] = (t, q, True)
            continue
        out.append((t, q, isc))
    return np.array([o[1] for o in out]), np.array([o[2] for o in out], bool)


def excise_loops(walk, max_excise=8):
    """remove short repeated-vertex loops (backtracks); keeps global winding"""
    out = []
    pos = {}
    for v in walk:
        if v in pos and len(out) - pos[v] <= max_excise:
            k = pos[v]
            for u in out[k:]:
                pos.pop(u, None)
            out = out[:k]
        pos[v] = len(out)
        out.append(v)
    # close-up backtrack at the seam
    while len(out) > 4 and out[0] == out[-1]:
        out.pop()
    return out


# ----------------------------------------------------------------------------
# snapping + relaxation
# ----------------------------------------------------------------------------
def snap_cycle(P, cyc, contour, cap_cells, cell, monotone=True, forced=None):
    """
    Pull the routed cycle's vertices onto the true contour.

    Plain nearest-point projection: the routed cycle already hugs the contour
    (max pull is under one cell), so the projection is locally order-preserving.
    `monotone` then only repairs the rare inversion, by nudging an out-of-order
    vertex back between its neighbours' targets instead of forcing a modular
    march (which used to send a vertex racing round the whole contour).
    """
    pts = contour["pts"]; n = len(pts)
    V = np.array(cyc, int)
    p = P[V]
    dd = ((p[:, None, 0] - pts[None, :, 0]) ** 2 +
          (p[:, None, 1] - pts[None, :, 1]) ** 2)
    ni = np.argmin(dd, axis=1)
    tgt = pts[ni].copy()
    if monotone and len(V) > 5:
        # an inversion = the step to the next target reverses the local march
        for _ in range(2):
            prv = np.roll(tgt, 1, axis=0); nxt = np.roll(tgt, -1, axis=0)
            step_in = np.linalg.norm(tgt - prv, axis=1)
            step_out = np.linalg.norm(nxt - tgt, axis=1)
            span = np.linalg.norm(nxt - prv, axis=1)
            bad = (step_in + step_out) > (span + 3.0 * cell)
            if not bad.any():
                break
            tgt[bad] = 0.5 * (prv[bad] + nxt[bad])
    if forced:
        for i, v in enumerate(V):
            if v in forced:
                tgt[i] = forced[v]
    d = tgt - p
    L = np.linalg.norm(d, axis=1)
    cap = cap_cells * cell
    sc = np.where(L > cap, cap / np.maximum(L, 1e-9), 1.0)
    P[V] = p + d * sc[:, None]
    return float(np.mean(L)), float(np.max(L))


def relax(lat, P, pinned, P0, Dall, iters=70, omega=0.85,
          spread_cells=7.0, anchor_far=0.22):
    NY, NX = lat.NY, lat.NX
    free = ~pinned
    # anchor weight: ~0 near the letters (distortion spreads), stronger far away
    dn = sample(Dall, P0) / (spread_cells * lat.cell)
    aw = anchor_far * np.clip(dn, 0.0, 1.0) ** 1.3
    aw = aw[:, None]
    ew = np.ones(lat.E.shape[0])
    Wv = np.zeros(lat.NV)
    np.add.at(Wv, lat.E[:, 0], ew); np.add.at(Wv, lat.E[:, 1], ew)
    Wv = np.maximum(Wv, 1e-9)
    e0 = lat.E[:, 0]; e1 = lat.E[:, 1]
    for _ in range(iters):
        acc = np.zeros_like(P)
        np.add.at(acc, e0, ew[:, None] * P[e1])
        np.add.at(acc, e1, ew[:, None] * P[e0])
        lap = acc / Wv[:, None]
        tgt = lap * (1 - aw) + P0 * aw
        Pn = P + omega * (tgt - P)
        P[free] = Pn[free]
    return P


def fold_stats(lat, P):
    q = P[lat.Q]
    x = q[:, :, 0]; y = q[:, :, 1]
    A = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    # triangle check catches bowties a quad area can hide
    t1 = tri_area(q[:, 0], q[:, 1], q[:, 2]); t2 = tri_area(q[:, 0], q[:, 2], q[:, 3])
    t3 = tri_area(q[:, 1], q[:, 2], q[:, 3]); t4 = tri_area(q[:, 1], q[:, 3], q[:, 0])
    bad = (A <= 0) | (t1 <= 0) | (t2 <= 0) | (t3 <= 0) | (t4 <= 0)
    return float(bad.mean()), A


def tri_area(a, b, c):
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                  (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def untangle(lat, P, pinned, iters=40):
    free = ~pinned
    e0 = lat.E[:, 0]; e1 = lat.E[:, 1]
    deg = np.zeros(lat.NV); np.add.at(deg, e0, 1); np.add.at(deg, e1, 1)
    deg = np.maximum(deg, 1)
    for _ in range(iters):
        bad, A = fold_stats(lat, P)
        if bad <= 0:
            break
        q = lat.Q[A <= 0]
        if len(q) == 0:
            break
        touch = np.zeros(lat.NV, bool); touch[q.ravel()] = True
        acc = np.zeros_like(P)
        np.add.at(acc, e0, P[e1]); np.add.at(acc, e1, P[e0])
        lap = acc / deg[:, None]
        m = touch & free
        P[m] = 0.35 * P[m] + 0.65 * lap[m]
    return P


# ----------------------------------------------------------------------------
# build one frame
# ----------------------------------------------------------------------------
def build(word, nx, ny, t=0.0, T=12.0, amp=0.30, cap_cells=99.0, lam=3.0, mu=1.4,
          anchor_mult=1.15, relax_iters=70, spread=7.0,
          band=2.2, gamma=1.7, floor=0.18, claim=1.6, cached=None, route_from=None, verbose=False):
    ck = cached if cached is not None else {}
    if "contours" not in ck:
        cs, sw, fsize, gm = glyph_contours(word)
        ck["contours"] = cs; ck["stroke"] = sw; ck["fsize"] = fsize
        ck["fields"] = [contour_field(c["pts"]) for c in cs]
        ck["tan"] = [tangents(c["pts"]) for c in cs]
        Du, Nu, ALLu = union_field(cs)
        ck["Dall"] = Du; ck["Nall"] = Nu; ck["ALL"] = ALLu
    cs = ck["contours"]
    lat = Lattice(nx, ny)
    P = wobbled(lat, t, T, amp)
    P0 = lat.P0.copy()

    t0 = time.time()
    cycles, forced_all = (route_from if route_from is not None else (None, None))
    routecost = 0.0
    if cycles is None:
        cycles = []; forced_all = []
        for ci, c in enumerate(cs):
            D, NEAR = ck["fields"][ci]
            cyc, cost, forced = route_contour(lat, P, c, D, NEAR, ck["tan"][ci],
                                              anchor_mult=anchor_mult, lam=lam, mu=mu,
                                              claim_radius=claim)
            cycles.append(cyc); forced_all.append(forced); routecost += cost
    t_route = time.time() - t0

    t0 = time.time()
    pinned = lat.border_mask().copy()
    snapd = []
    for ci, cyc in enumerate(cycles):
        if cyc is None or len(cyc) < 4:
            snapd.append(None); continue
        m, mx = snap_cycle(P, cyc, cs[ci], cap_cells, lat.cell,
                           forced=forced_all[ci] if forced_all else None)
        pinned[np.array(cyc, int)] = True
        snapd.append((m, mx))
    polys = [P[np.array(c, int)] for c in cycles if c is not None and len(c) >= 4]

    P = relax(lat, P, pinned, P0, ck["Dall"], iters=relax_iters, spread_cells=spread)
    if band > 0:
        P = ridge(P, ~pinned, ck["Dall"], ck["Nall"], ck["ALL"], lat.cell,
                  band_cells=band, gamma=gamma, floor=floor)
        P = relax(lat, P, pinned, P0, ck["Dall"], iters=max(8, relax_iters // 6),
                  spread_cells=spread, omega=0.35)
    P = untangle(lat, P, pinned)
    t_relax = time.time() - t0

    foldr, _ = fold_stats(lat, P)
    info = dict(nx=nx, ny=ny, cell=lat.cell, t_route=t_route, t_relax=t_relax,
                fold=foldr, ncyc=sum(1 for c in cycles if c is not None),
                ncontour=len(cs), stroke=ck["stroke"],
                cyclen=[0 if c is None else len(c) for c in cycles],
                snap=snapd, routecost=routecost)
    return lat, P, (cycles, forced_all), cs, info, ck


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
def render_wireframe(lat, P, path=None, ss=3, lw=1.0, size=(W, H),
                     ink=INK, paper=PAPER):
    """THE ACCEPTANCE TEST: every lattice edge, thin, uniform, one colour."""
    im = Image.new("RGB", (size[0] * ss, size[1] * ss), paper)
    d = ImageDraw.Draw(im)
    a = P[lat.E[:, 0]] * ss; b = P[lat.E[:, 1]] * ss
    wpx = max(1, int(round(lw * ss)))
    d.fontmode = 'L'
    for i in range(lat.E.shape[0]):
        d.line([a[i, 0], a[i, 1], b[i, 0], b[i, 1]], fill=ink, width=wpx)
    im = im.resize(size, Image.LANCZOS)
    if path:
        im.save(path)
    return im


def cell_ids(lat, P, polys, rng, id_in, id_out, size=(W, H)):
    cen = P[lat.Q].mean(axis=1)
    ins = point_in_polys(cen, polys) if polys else np.zeros(len(cen), bool)
    ids = np.where(ins, rng.choice(id_in, len(cen)), rng.choice(id_out, len(cen)))
    m = Image.new("L", size, 255)
    d = ImageDraw.Draw(m)
    q = P[lat.Q]
    for i in range(lat.Q.shape[0]):
        d.polygon([tuple(p) for p in q[i]], fill=int(ids[i]))
    return np.asarray(m), ins


def composite(lat, P, polys, clips, seed=7, id_in=(0, 1), id_out=(2, 3)):
    rng = np.random.default_rng(seed)
    M, ins = cell_ids(lat, P, polys, rng, list(id_in), list(id_out))
    out = np.zeros((H, W, 3), np.uint8)
    for i, c in enumerate(clips):
        sel = M == i
        out[sel] = c[sel]
    fill = M > 3
    if fill.any():
        out[fill] = clips[id_out[0]][fill]
    return Image.fromarray(out), M


def load_clips():
    cl = []
    for i in range(4):
        p = os.path.join(FRAMES, "clip%d.png" % i)
        im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
        cl.append(np.asarray(im))
    return cl


def label(im, text, h=26):
    out = Image.new("RGB", (im.width, im.height + h), (255, 255, 255))
    out.paste(im, (0, h))
    d = ImageDraw.Draw(out)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except Exception:
        f = ImageFont.load_default()
    d.text((6, 5), text, fill=(10, 10, 10), font=f)
    return out


def grid_sheet(tiles, cols, path, tw=620):
    ims = []
    for im, cap in tiles:
        s = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
        ims.append(label(s, cap))
    rows = math.ceil(len(ims) / cols)
    cw = max(i.width for i in ims); chh = max(i.height for i in ims)
    sheet = Image.new("RGB", (cols * cw + (cols + 1) * 8, rows * chh + (rows + 1) * 8),
                      (228, 228, 230))
    for k, i in enumerate(ims):
        r, c = divmod(k, cols)
        sheet.paste(i, (8 + c * (cw + 8), 8 + r * (chh + 8)))
    sheet.save(path)
    return sheet


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------
DENSITIES = [(96, 54), (72, 40), (64, 36), (48, 27), (40, 22), (32, 18),
             (24, 14), (20, 11), (16, 9), (12, 7)]


def cmd_wire(args):
    ck = {}
    tiles = []
    log = []
    for (nx, ny) in DENSITIES:
        lat, P, cyc, cs, info, ck = build(args.word, nx, ny, t=args.t, amp=args.amp,
                                          cap_cells=args.cap,
                                          spread=args.spread, relax_iters=args.iters,
                                          lam=args.lam, mu=args.mu,
                                          band=args.band, gamma=args.gamma, floor=args.floor, claim=args.claim, cached=ck)
        im = render_wireframe(lat, P, lw=args.lw)
        cap = ("%dx%d cells  cell=%.0fpx  stroke=%.0fpx (%.2f cell)  cycles=%d/%d  fold=%.2f%%"
               % (nx, ny, info["cell"], info["stroke"], info["stroke"] / info["cell"],
                  info["ncyc"], info["ncontour"], 100 * info["fold"]))
        tiles.append((im, cap))
        log.append(dict(nx=nx, ny=ny, **{k: info[k] for k in
                   ("cell", "fold", "ncyc", "ncontour", "t_route", "t_relax", "stroke")}))
        print(cap, flush=True)
        im.save(os.path.join(OUT, "work", "wire-%s-%dx%d.png" % (args.word.replace(" ", ""), nx, ny)))
    grid_sheet(tiles, 2, os.path.join(OUT, "wireframe-sheet.png"), tw=620)
    json.dump(log, open(os.path.join(OUT, "work", "wire-sweep-%s.json" % args.word.replace(" ", "")), "w"), indent=1)


def cmd_one(args):
    lat, P, cyc, cs, info, ck = build(args.word, args.nx, args.ny, t=args.t, amp=args.amp,
                                      cap_cells=args.cap,
                                      spread=args.spread, relax_iters=args.iters,
                                      lam=args.lam, mu=args.mu,
                                      band=args.band, gamma=args.gamma, floor=args.floor, claim=args.claim)
    im = render_wireframe(lat, P, lw=args.lw)
    im.save(args.out or os.path.join(OUT, "wireframe.png"))
    print(json.dumps({k: v for k, v in info.items() if k != "snap"}, default=str))
    if args.comp:
        polys = [P[np.array(c, int)] for c in cyc[0] if c is not None and len(c) >= 4]
        clips = load_clips()
        c, M = composite(lat, P, polys, clips, id_in=tuple(args.idin), id_out=tuple(args.idout))
        c.save(args.comp)


def cmd_matrix(args):
    clips = load_clips()
    lum = [float(np.asarray(c).mean()) for c in clips]
    order = np.argsort(lum)
    id_out = (int(order[0]), int(order[1]))   # dark family outside
    id_in = (int(order[2]), int(order[3]))    # bright family inside
    print("clip luminance", [round(l, 1) for l in lum], "-> in", id_in, "out", id_out)

    tiles = []; wtiles = []; log = []
    for word in args.words:
        ck = {}
        for (nx, ny) in args.dens:
            for amp in args.amps:
                lat, P, cyc, cs, info, ck = build(word, nx, ny, t=args.t, amp=amp,
                                                  cap_cells=args.cap,
                                                  spread=args.spread, relax_iters=args.iters,
                                                  lam=args.lam, mu=args.mu,
                                                  band=args.band, gamma=args.gamma, floor=args.floor, claim=args.claim, cached=ck)
                polys = [P[np.array(c, int)] for c in cyc[0] if c is not None and len(c) >= 4]
                cim, M = composite(lat, P, polys, clips, id_in=id_in, id_out=id_out)
                wim = render_wireframe(lat, P, lw=args.lw)
                cap = ("%s  %dx%d (cell %.0fpx, stroke %.2f cell)  wobble %.2f  cyc %d/%d  fold %.2f%%"
                       % (word, nx, ny, info["cell"], info["stroke"] / info["cell"], amp,
                          info["ncyc"], info["ncontour"], 100 * info["fold"]))
                tiles.append((cim, cap)); wtiles.append((wim, cap))
                log.append(dict(word=word, nx=nx, ny=ny, amp=amp,
                                cell=info["cell"], fold=info["fold"], ncyc=info["ncyc"],
                                ncontour=info["ncontour"], t_route=info["t_route"],
                                t_relax=info["t_relax"],
                                stroke_cells=info["stroke"] / info["cell"]))
                print(cap, flush=True)
    grid_sheet(tiles, args.cols, os.path.join(OUT, "contact-sheet.png"), tw=args.tw)
    grid_sheet(wtiles, args.cols, os.path.join(OUT, "contact-sheet-wire.png"), tw=args.tw)
    json.dump(log, open(os.path.join(OUT, "work", "matrix.json"), "w"), indent=1)


def cmd_temporal(args):
    """route-once vs route-per-frame: measure edge-set churn between frames"""
    ck = {}
    lat = Lattice(args.nx, args.ny)
    prev = None; churn = []; disp = []
    prevP = None
    ref = None
    for k in range(args.n):
        t = args.T * k / args.n
        lat, P, cyc, cs, info, ck = build(args.word, args.nx, args.ny, t=t, amp=args.amp,
                                          cap_cells=args.cap,
                                          spread=args.spread, relax_iters=args.iters,
                                          band=args.band, gamma=args.gamma, floor=args.floor, claim=args.claim, cached=ck)
        eset = set()
        for c in cyc[0]:
            if c is None: continue
            for i in range(len(c)):
                a, b = c[i], c[(i + 1) % len(c)]
                eset.add((min(a, b), max(a, b)))
        if prev is not None:
            churn.append(len(eset ^ prev) / max(1, len(eset | prev)))
        prev = eset
        if prevP is not None:
            disp.append(float(np.percentile(np.linalg.norm(P - prevP, axis=1), 99)))
        prevP = P.copy()
        if args.save and k % max(1, args.n // 8) == 0:
            render_wireframe(lat, P, lw=args.lw).save(
                os.path.join(OUT, "work", "temporal-%s-%02d.png" % (args.mode, k)))
    print(json.dumps(dict(mode=args.mode, nx=args.nx, amp=args.amp,
                          churn_mean=float(np.mean(churn)) if churn else 0.0,
                          churn_max=float(np.max(churn)) if churn else 0.0,
                          disp_p99_mean=float(np.mean(disp)) if disp else 0.0,
                          disp_p99_max=float(np.max(disp)) if disp else 0.0)))


def cmd_temporal_fixed(args):
    """route once on the rest lattice, re-snap per frame"""
    ck = {}
    lat = Lattice(args.nx, args.ny)
    lat0, P0r, cyc0, cs, info0, ck = build(args.word, args.nx, args.ny, t=0.0, amp=0.0,
                                           cap_cells=args.cap,
                                           spread=args.spread, relax_iters=args.iters,
                                           band=args.band, gamma=args.gamma, floor=args.floor, claim=args.claim, cached=ck)
    prevP = None; disp = []; folds = []
    for k in range(args.n):
        t = args.T * k / args.n
        lat, P, cyc, cs, info, ck = build(args.word, args.nx, args.ny, t=t, amp=args.amp,
                                          cap_cells=args.cap,
                                          spread=args.spread, relax_iters=args.iters,
                                          cached=ck, route_from=cyc0)
        folds.append(info["fold"])
        if prevP is not None:
            disp.append(float(np.percentile(np.linalg.norm(P - prevP, axis=1), 99)))
        prevP = P.copy()
        if args.save and k % max(1, args.n // 8) == 0:
            render_wireframe(lat, P, lw=args.lw).save(
                os.path.join(OUT, "work", "temporal-fixed-%02d.png" % k))
    print(json.dumps(dict(mode="fixed-route", nx=args.nx, amp=args.amp, churn_mean=0.0,
                          churn_max=0.0,
                          disp_p99_mean=float(np.mean(disp)), disp_p99_max=float(np.max(disp)),
                          fold_mean=float(np.mean(folds)), fold_max=float(np.max(folds)))))


def cmd_caps(args):
    ck = {}
    tiles = []
    for cap in args.caps:
        lat, P, cyc, cs, info, ck = build(args.word, args.nx, args.ny, t=args.t, amp=args.amp,
                                          cap_cells=cap,
                                          spread=args.spread, relax_iters=args.iters,
                                          band=args.band, gamma=args.gamma, floor=args.floor, claim=args.claim, cached=ck)
        im = render_wireframe(lat, P, lw=args.lw)
        mean = np.mean([s[0] for s in info["snap"] if s]) if info["snap"] else 0
        cp = "cap %.2f cell (%.0fpx)  mean pull %.1fpx  fold %.2f%%" % (
            cap, cap * info["cell"], mean, 100 * info["fold"])
        tiles.append((im, cp)); print(cp, flush=True)
    grid_sheet(tiles, 2, os.path.join(OUT, "snap-cap-sheet.png"), tw=620)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["wire", "one", "matrix", "temporal",
                                    "temporalfixed", "caps", "extract"])
    ap.add_argument("--word", default="Yope3D")
    ap.add_argument("--words", nargs="*", default=["Yope3D", "SpinStack", "3D Gravity Simulator"])
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=36)
    ap.add_argument("--dens", nargs="*", type=int, default=None)
    ap.add_argument("--amp", type=float, default=0.30)
    ap.add_argument("--amps", nargs="*", type=float, default=[0.0, 0.30])
    ap.add_argument("--cap", type=float, default=99.0)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--mu", type=float, default=1.4)
    ap.add_argument("--band", type=float, default=2.2)
    ap.add_argument("--gamma", type=float, default=1.7)
    ap.add_argument("--floor", type=float, default=0.18)
    ap.add_argument("--claim", type=float, default=1.6)
    ap.add_argument("--spread", type=float, default=7.0)
    ap.add_argument("--iters", type=int, default=70)
    ap.add_argument("--lw", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=12.0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--mode", default="perframe")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--comp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--tw", type=int, default=620)
    ap.add_argument("--caps", nargs="*", type=float, default=[0.25, 0.5, 1.0, 99.0])
    ap.add_argument("--idin", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--idout", nargs="*", type=int, default=[2, 3])
    a = ap.parse_args()
    if a.cmd == "wire":
        cmd_wire(a)
    elif a.cmd == "one":
        cmd_one(a)
    elif a.cmd == "matrix":
        d = a.dens or [64, 36, 40, 22, 24, 14]
        a.dens = [(d[i], d[i + 1]) for i in range(0, len(d), 2)]
        cmd_matrix(a)
    elif a.cmd == "temporal":
        cmd_temporal(a)
    elif a.cmd == "temporalfixed":
        cmd_temporal_fixed(a)
    elif a.cmd == "caps":
        cmd_caps(a)
    elif a.cmd == "extract":
        extract_clips()


def extract_clips():
    base = "/Users/me/Desktop/dev/yugumishra.github.io/public/media"
    srcs = [os.path.join(base, "yope_cloth_b.mp4"),
            os.path.join(base, "showcase-test", "spinstack.mp4"),
            os.path.join(base, "showcase-test", "digits.mp4"),
            os.path.join(base, "showcase-test", "gravity.mp4")]
    for i, s in enumerate(srcs):
        dst = os.path.join(FRAMES, "clip%d.png" % i)
        if os.path.exists(dst):
            print("have", dst); continue
        if not os.path.exists(s):
            print("MISSING", s); continue
        subprocess.run(["ffmpeg", "-y", "-ss", "2", "-i", s, "-frames:v", "1",
                        "-vf", "scale=%d:%d" % (W, H), dst],
                       check=True, capture_output=True)
        print("wrote", dst)


if __name__ == "__main__":
    main()
