#!/usr/bin/env python3
"""
probe-g-global-warp.py  --  Agent G probe: ONE SMOOTH GLOBAL WARP.

Goal (art direction):
  A single fine quad lattice.  Nothing drawn on top.  The letter outline must BE
  a closed chain of ordinary lattice edges.  One scale of piece.  Not square.
  Deformity + motion survive.

Algorithm family owned by agent G:
  Do not snap vertices individually.  Compute ONE continuous deformation field
  over the whole canvas and push the entire regular lattice through it, so that
  chosen grid lines land on the glyph outline.

Machinery actually used here:
  1. Rasterise the word, extract sub-pixel contours (marching squares).
  2. Corner-aware resample of each contour  ->  target polygon P.
  3. Miter-offset P away from the glyph solid by `dilate` px -> SOURCE polygon Q.
     (The source region is a fattened glyph; it maps onto the true glyph, so the
      letter interior COMPRESSES -> denser cells inside -> the word reads as a
      change of cell density, not as a drawn line.)
  4. Correspondence: nearest undeformed lattice node of each Q sample, walked in
     contour order with gap-filling, gives a closed CHAIN of grid-adjacent nodes.
     Each chain node inherits the matching point of P as its target.
     (This is the control-pair construction; no routing, no per-vertex snapping
      of anything else, no inside/outside labelling of cells.)
  5. ONE global solve for the displacement field u over every lattice node:
        minimise  || L u ||^2 + alpha ||u||^2      (L = graph Laplacian)
        subject to  u = target - source  at chain nodes.
     Biharmonic => C1 across the constraint chain => neighbours travel WITH the
     contour instead of piling into it, and the distortion bleeds far out.
     A thin-plate-spline variant is included for comparison (--solver tps).
  6. Motion: wobble the undeformed lattice, then push the wobbled lattice
     through the same static field.  12 s exact loop from integer harmonics.

Outputs into the agent-g scratchpad.  Self-contained.  Reads nothing under src/.
"""

import argparse, json, math, os, random, sys, time
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

# ----------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 1280, 720
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
OUT = ("/private/tmp/claude-501/-Users-me-Desktop-dev-yugumishra-github-io/"
       "ea9e4c74-aeb0-4767-b13b-b99660eec2a7/scratchpad/agent-g")
FRAMES = os.path.join(OUT, "frames")
LOOP_SECONDS = 12.0
FPS = 30

INK = (26, 30, 38)
PAPER = (250, 249, 246)


# ============================================================ glyph -> contours
def render_word(word, canvas=(CANVAS_W, CANVAS_H), width_frac=0.80, height_frac=0.32,
                ss=4):
    """Supersampled alpha coverage of `word`, centred, fitted to the canvas."""
    W, H = canvas
    lo, hi = 8, 900
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(FONT_PATH, mid)
        bb = f.getbbox(word)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w <= W * width_frac and h <= H * height_frac:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    font = ImageFont.truetype(FONT_PATH, best)
    bb = font.getbbox(word)
    gw, gh = bb[2] - bb[0], bb[3] - bb[1]

    bigfont = ImageFont.truetype(FONT_PATH, best * ss)
    bbb = bigfont.getbbox(word)
    big = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(big)
    ox = (W * ss - (bbb[2] - bbb[0])) / 2.0 - bbb[0]
    oy = (H * ss - (bbb[3] - bbb[1])) / 2.0 - bbb[1]
    d.text((ox, oy), word, fill=255, font=bigfont)
    small = big.resize((W, H), Image.LANCZOS)
    alpha = np.asarray(small).astype(np.float64) / 255.0
    return alpha, dict(font_px=best, cap_h=gh, ink_w=gw)


def stroke_width_estimate(alpha):
    """Median run length of solid pixels along scanlines that cut the letters."""
    m = alpha > 0.5
    runs = []
    for row in m:
        idx = np.flatnonzero(np.diff(row.astype(np.int8)))
        if len(idx) < 2:
            continue
        for a, b in zip(idx[0::2], idx[1::2]):
            runs.append(b - a)
    return float(np.median(runs)) if runs else 0.0


def marching_squares(F, level=0.5):
    """Sub-pixel closed contours of F at `level`.  Returns list of (N,2) arrays."""
    H, W = F.shape
    f = F - level
    a = f[:-1, :-1]; b = f[:-1, 1:]; c = f[1:, 1:]; d = f[1:, :-1]
    code = ((a > 0).astype(np.uint8) | ((b > 0).astype(np.uint8) << 1) |
            ((c > 0).astype(np.uint8) << 2) | ((d > 0).astype(np.uint8) << 3))
    ys, xs = np.nonzero((code > 0) & (code < 15))
    if len(xs) == 0:
        return []

    TABLE = {1: [(3, 0)], 14: [(3, 0)], 2: [(0, 1)], 13: [(0, 1)],
             3: [(3, 1)], 12: [(3, 1)], 4: [(1, 2)], 11: [(1, 2)],
             6: [(0, 2)], 9: [(0, 2)], 7: [(3, 2)], 8: [(3, 2)],
             5: [(3, 0), (1, 2)], 10: [(0, 1), (2, 3)]}

    pos = {}
    adj = defaultdict(list)

    def crossing(x, y, e):
        """Unique id + sub-pixel position for edge `e` of cell (x,y)."""
        if e == 0:   key = (0, x, y);     p0 = (x, y);         p1 = (x + 1, y)
        elif e == 1: key = (1, x + 1, y); p0 = (x + 1, y);     p1 = (x + 1, y + 1)
        elif e == 2: key = (0, x, y + 1); p0 = (x, y + 1);     p1 = (x + 1, y + 1)
        else:        key = (1, x, y);     p0 = (x, y);         p1 = (x, y + 1)
        if key not in pos:
            v0 = f[p0[1], p0[0]]; v1 = f[p1[1], p1[0]]
            t = 0.5 if v1 == v0 else v0 / (v0 - v1)
            t = min(1.0, max(0.0, t))
            pos[key] = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
        return key

    for x, y in zip(xs, ys):
        for e0, e1 in TABLE[int(code[y, x])]:
            k0 = crossing(x, y, e0); k1 = crossing(x, y, e1)
            adj[k0].append(k1); adj[k1].append(k0)

    seen = set(); polys = []
    for start in list(adj.keys()):
        if start in seen:
            continue
        chain = [start]; seen.add(start); prev = None; cur = start
        while True:
            nxt = None
            for cand in adj[cur]:
                if cand != prev and cand not in seen:
                    nxt = cand; break
            if nxt is None:
                break
            chain.append(nxt); seen.add(nxt); prev, cur = cur, nxt
        if len(chain) >= 8:
            polys.append(np.array([pos[k] for k in chain], dtype=np.float64))
    return polys


# ============================================================ polygon utilities
def signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def rdp(P, eps):
    """Douglas-Peucker on a closed polyline (iterative)."""
    n = len(P)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        A, B = P[i], P[j]
        ab = B - A
        L = math.hypot(*ab)
        seg = P[i + 1:j]
        if L < 1e-9:
            dist = np.hypot(seg[:, 0] - A[0], seg[:, 1] - A[1])
        else:
            dist = np.abs((seg[:, 0] - A[0]) * ab[1] - (seg[:, 1] - A[1]) * ab[0]) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k)); stack.append((i + 1 + k, j))
    return P[keep]


def corner_flags(P, thresh_deg=32.0):
    """True where the closed polygon P turns by more than thresh_deg."""
    prev = np.roll(P, 1, axis=0); nxt = np.roll(P, -1, axis=0)
    a = P - prev; b = nxt - P
    na = np.hypot(a[:, 0], a[:, 1]) + 1e-12
    nb = np.hypot(b[:, 0], b[:, 1]) + 1e-12
    cosang = np.clip((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / (na * nb), -1, 1)
    return np.degrees(np.arccos(cosang)) > thresh_deg


def detect_corners(P, win_px=7.0, deg=38.0):
    """Corners of a DENSE closed contour: windowed turn angle + non-max
    suppression.  Far more reliable than reading angles off an RDP polyline."""
    n = len(P)
    step = np.hypot(*(np.roll(P, -1, 0) - P).T[::-1])
    mean_sp = max(float(np.mean(step)), 1e-6)
    k = int(max(2, round(win_px / mean_sp)))
    if 2 * k + 1 >= n:
        return np.zeros(n, bool), np.zeros(n)
    a = P - np.roll(P, k, axis=0)
    b = np.roll(P, -k, axis=0) - P
    na = np.hypot(a[:, 0], a[:, 1]) + 1e-12
    nb = np.hypot(b[:, 0], b[:, 1]) + 1e-12
    cosang = np.clip((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / (na * nb), -1, 1)
    turn = np.degrees(np.arccos(cosang))
    flag = turn > deg
    # non-maximum suppression inside each flagged run (circular)
    out = np.zeros(n, bool)
    if not flag.any():
        return out, turn
    idx = np.flatnonzero(flag)
    runs, cur = [], [idx[0]]
    for p, q in zip(idx, idx[1:]):
        if q == p + 1:
            cur.append(q)
        else:
            runs.append(cur); cur = [q]
    runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
        runs[0] = runs[-1] + runs[0]; runs.pop()
    for r in runs:
        out[r[int(np.argmax(turn[np.array(r)]))]] = True
    return out, turn


def resample_corner_aware(P, step, corner_deg=38.0, corner_win=7.0):
    """Walk the dense contour by arc length emitting a sample every ~`step`,
    but ALWAYS emit at a detected corner so the corner survives verbatim.
    Returns (points, is_corner)."""
    cor, _ = detect_corners(P, corner_win, corner_deg)
    n = len(P)
    seg = np.hypot(*(np.roll(P, -1, 0) - P).T[::-1])
    out, oc, acc = [], [], 1e9
    for i in range(n):
        if cor[i] or acc >= step:
            out.append(P[i]); oc.append(bool(cor[i])); acc = 0.0
        acc += seg[i]
    if len(out) < 4:
        return P, cor
    return np.array(out), np.array(oc, dtype=bool)


def outward_normals(P, alpha_map, probe=2.5):
    """Unit normal at each vertex pointing AWAY from the glyph solid.
    Orientation is decided by SAMPLING THE MASK, so the marching-squares walk
    direction cannot get it wrong."""
    prev = np.roll(P, 1, axis=0); nxt = np.roll(P, -1, axis=0)
    t = nxt - prev
    n = np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-12
    t = t / n
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)          # y-down rotate
    H, W = alpha_map.shape

    def samp(pts):
        xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, H - 1)
        return alpha_map[yi, xi]
    plus = samp(P + nrm * probe); minus = samp(P - nrm * probe)
    if np.mean(plus) > np.mean(minus):       # +n points INTO the solid -> flip
        nrm = -nrm
    return nrm


def miter_offset(P, dist, out_n, miter_limit=2.6):
    """Offset each vertex along its angle bisector so corners STAY corners."""
    prev = np.roll(P, 1, axis=0); nxt = np.roll(P, -1, axis=0)
    e0 = P - prev; e1 = nxt - P
    l0 = np.hypot(e0[:, 0], e0[:, 1])[:, None] + 1e-12
    l1 = np.hypot(e1[:, 0], e1[:, 1])[:, None] + 1e-12
    e0 = e0 / l0; e1 = e1 / l1
    n0 = np.stack([e0[:, 1], -e0[:, 0]], axis=1)
    n1 = np.stack([e1[:, 1], -e1[:, 0]], axis=1)
    s = np.sign(np.sum(n0 * out_n, axis=1))[:, None]
    s[s == 0] = 1.0
    n0 = n0 * s; n1 = n1 * s
    bis = n0 + n1
    bl = np.hypot(bis[:, 0], bis[:, 1])[:, None]
    safe = bl > 1e-6
    bis = np.where(safe, bis / np.where(safe, bl, 1.0), out_n)
    scale = np.where(safe, 2.0 / np.maximum(bl ** 2, 1e-6), 1.0)
    scale = np.minimum(scale, miter_limit)
    dist = np.asarray(dist, dtype=float).reshape(-1, 1)
    return P + bis * (dist * scale)


def point_in_poly(pt, P):
    x, y = pt
    inside = False
    n = len(P)
    j = n - 1
    for i in range(n):
        xi, yi = P[i]; xj, yj = P[j]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


# ============================================================== control chain
def build_chain(Q, Ptar, is_corner, nx, ny, cw, ch, ox, oy):
    """Nearest undeformed lattice node for every SOURCE sample, walked in order,
    gaps filled, so the result is a CLOSED CHAIN of grid-adjacent nodes.
    Returns  {node_index: target_xy}  plus diagnostics."""
    gi = np.clip(np.rint((Q[:, 0] - ox) / cw).astype(int), 0, nx)
    gj = np.clip(np.rint((Q[:, 1] - oy) / ch).astype(int), 0, ny)

    walk = []          # (i, j, target_xy, corner?, source_xy)
    n = len(Q)
    for k in range(n):
        i, j = int(gi[k]), int(gj[k])
        tgt = Ptar[k]; srcp = Q[k]
        if walk and walk[-1][0] == i and walk[-1][1] == j:
            # same node claimed twice by neighbouring samples; corner wins
            if is_corner[k] and not walk[-1][3]:
                walk[-1] = (i, j, tgt, True, srcp)
            continue
        if walk:
            pi, pj = walk[-1][0], walk[-1][1]
            di, dj = i - pi, j - pj
            steps = max(abs(di), abs(dj))
            if steps > 1 or (abs(di) == 1 and abs(dj) == 1):
                # fill with a 4-connected staircase, targets interpolated
                ptgt = walk[-1][2]; psrc = walk[-1][4]
                ci, cj = pi, pj
                total = abs(di) + abs(dj)
                done = 0
                while (ci, cj) != (i, j):
                    if abs(i - ci) >= abs(j - cj) and ci != i:
                        ci += 1 if i > ci else -1
                    elif cj != j:
                        cj += 1 if j > cj else -1
                    else:
                        ci += 1 if i > ci else -1
                    done += 1
                    if (ci, cj) == (i, j):
                        break
                    t = done / float(total)
                    walk.append((ci, cj, ptgt + (tgt - ptgt) * t, False,
                                 psrc + (srcp - psrc) * t))
        walk.append((i, j, tgt, bool(is_corner[k]), srcp))

    # close the loop
    if walk and (walk[0][0], walk[0][1]) != (walk[-1][0], walk[-1][1]):
        pi, pj = walk[-1][0], walk[-1][1]
        i, j = walk[0][0], walk[0][1]
        ptgt, tgt = walk[-1][2], walk[0][2]
        psrc, srcp = walk[-1][4], walk[0][4]
        total = abs(i - pi) + abs(j - pj)
        ci, cj = pi, pj; done = 0
        while (ci, cj) != (i, j):
            if abs(i - ci) >= abs(j - cj) and ci != i:
                ci += 1 if i > ci else -1
            elif cj != j:
                cj += 1 if j > cj else -1
            else:
                ci += 1 if i > ci else -1
            done += 1
            if (ci, cj) == (i, j):
                break
            fr = done / max(total, 1)
            walk.append((ci, cj, ptgt + (tgt - ptgt) * fr, False,
                         psrc + (srcp - psrc) * fr))

    # strip immediate backtracks a,b,a
    cleaned = []
    for w in walk:
        if len(cleaned) >= 2 and (cleaned[-2][0], cleaned[-2][1]) == (w[0], w[1]):
            cleaned.pop()
            continue
        cleaned.append(w)
    return cleaned


def collect_constraints(chains, nx, ny, cw, ch, ox, oy):
    """Merge per-contour chains into one constraint dict.

    When two distant pieces of outline want the SAME lattice node (the counter
    of an `e` against its crossbar, two letters touching) averaging the two
    targets tears a diagonal gash across the letter.  Instead the claim whose
    SOURCE point sits closest to the node wins outright and the loser is simply
    dropped -- the field interpolates through the gap smoothly.  The number of
    such events is the honest density criterion."""
    claims = defaultdict(list)
    for ch_ in chains:
        for (i, j, tgt, corner, srcp) in ch_:
            claims[(i, j)].append((tgt, corner, srcp))
    cons = {}
    collisions = 0
    max_spread = 0.0
    for (i, j), lst in claims.items():
        node = np.array([ox + i * cw, oy + j * ch])
        pts = np.array([t for t, _c, _s in lst])
        if len(pts) > 1:
            spread = float(np.max(np.hypot(pts[:, 0] - pts[:, 0].mean(),
                                           pts[:, 1] - pts[:, 1].mean())))
            max_spread = max(max_spread, spread)
            if spread > 0.75 * max(cw, ch):
                collisions += 1
        best, bestd = None, 1e18
        for (t, c, srcp) in lst:
            d = float(np.hypot(*(srcp - node))) - (0.45 * min(cw, ch) if c else 0.0)
            if d < bestd:
                bestd, best = d, t
        cons[j * (nx + 1) + i] = best
    return cons, collisions, max_spread


# ================================================================ global solve
def grid_laplacian(nx, ny):
    N = (nx + 1) * (ny + 1)
    rows, cols, vals = [], [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            p = j * (nx + 1) + i
            deg = 0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a <= nx and 0 <= b <= ny:
                    rows.append(p); cols.append(b * (nx + 1) + a); vals.append(1.0)
                    deg += 1
            rows.append(p); cols.append(p); vals.append(-float(deg))
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def solve_biharmonic(nx, ny, cons, alpha, order=2):
    """min ||L^order u||^2 + alpha||u||^2  s.t.  u = g on constrained nodes."""
    N = (nx + 1) * (ny + 1)
    L = grid_laplacian(nx, ny)
    K = L
    for _ in range(order - 1):
        K = K @ L
    S = (K.T @ K).tocsr() + alpha * sp.identity(N, format="csr")

    cidx = np.array(sorted(cons.keys()), dtype=int)
    mask = np.zeros(N, bool); mask[cidx] = True
    fidx = np.flatnonzero(~mask)
    g = np.zeros((N, 2))
    for p in cidx:
        g[p] = cons[p]

    Sff = S[fidx][:, fidx].tocsc()
    Sfc = S[fidx][:, cidx]
    rhs = -(Sfc @ g[cidx])
    t0 = time.time()
    lu = spla.splu(Sff)
    uf = np.column_stack([lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])])
    solve_t = time.time() - t0

    u = np.zeros((N, 2))
    u[cidx] = g[cidx]
    u[fidx] = uf
    return u, solve_t


def solve_tps(nx, ny, cons, base, lam=0.0):
    """Thin-plate spline over the same control pairs, for comparison."""
    cidx = np.array(sorted(cons.keys()), dtype=int)
    src = base[cidx]
    dst = np.array([cons[p] for p in cidx])
    n = len(cidx)
    d2 = (np.sum(src ** 2, 1)[:, None] + np.sum(src ** 2, 1)[None, :]
          - 2 * src @ src.T)
    d2 = np.maximum(d2, 0)
    Kk = np.where(d2 > 1e-12, 0.5 * d2 * np.log(np.maximum(d2, 1e-12)), 0.0)
    Pm = np.column_stack([np.ones(n), src])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = Kk + lam * np.eye(n)
    A[:n, n:] = Pm; A[n:, :n] = Pm.T
    rhs = np.zeros((n + 3, 2)); rhs[:n] = dst
    t0 = time.time()
    W = np.linalg.lstsq(A, rhs, rcond=None)[0]
    solve_t = time.time() - t0
    w, aff = W[:n], W[n:]

    out = np.empty_like(base)
    CH = 4000
    for s in range(0, len(base), CH):
        blk = base[s:s + CH]
        dd = (np.sum(blk ** 2, 1)[:, None] + np.sum(src ** 2, 1)[None, :]
              - 2 * blk @ src.T)
        dd = np.maximum(dd, 0)
        kk = np.where(dd > 1e-12, 0.5 * dd * np.log(np.maximum(dd, 1e-12)), 0.0)
        out[s:s + CH] = kk @ w + np.column_stack([np.ones(len(blk)), blk]) @ aff
    return out - base, solve_t


# ==================================================================== dynamics
def wobble_field(nx, ny, cw, ch, amp, t, seed=7):
    """Time-periodic wobble, exact 12 s loop (integer harmonics)."""
    if amp <= 0:
        return np.zeros(((nx + 1) * (ny + 1), 2))
    rng = np.random.default_rng(seed)
    N = (nx + 1) * (ny + 1)
    acc = np.zeros((N, 2))
    for h, w in ((1, 1.0), (2, 0.55), (3, 0.3)):
        ph = rng.uniform(0, 2 * np.pi, (N, 2))
        acc += w * np.sin(2 * np.pi * h * t / LOOP_SECONDS + ph)
    acc /= 1.85
    return acc * amp * np.array([cw, ch])[None, :]


def sample_field(u, nx, ny, cw, ch, ox, oy, pts):
    """Bilinear sample of the node-defined displacement field at arbitrary pts."""
    U = u.reshape(ny + 1, nx + 1, 2)
    gx = np.clip((pts[:, 0] - ox) / cw, 0, nx - 1e-6)
    gy = np.clip((pts[:, 1] - oy) / ch, 0, ny - 1e-6)
    i0 = gx.astype(int); j0 = gy.astype(int)
    fx = (gx - i0)[:, None]; fy = (gy - j0)[:, None]
    i1 = np.minimum(i0 + 1, nx); j1 = np.minimum(j0 + 1, ny)
    return ((U[j0, i0] * (1 - fx) + U[j0, i1] * fx) * (1 - fy) +
            (U[j1, i0] * (1 - fx) + U[j1, i1] * fx) * fy)


# ==================================================================== metrics
def quad_metrics(pos, nx, ny):
    P = pos.reshape(ny + 1, nx + 1, 2)
    a = P[:-1, :-1]; b = P[:-1, 1:]; c = P[1:, 1:]; d = P[1:, :-1]
    def cross(u, v): return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    area = 0.5 * (cross(a, b) + cross(b, c) + cross(c, d) + cross(d, a))
    def elen(u, v):
        w = v - u
        return np.hypot(w[..., 0], w[..., 1])
    e = np.stack([elen(a, b), elen(b, c), elen(c, d), elen(d, a)], axis=-1)
    emax = e.max(-1); emin = np.maximum(e.min(-1), 1e-9)
    ar = emax / emin
    # scaled jacobian at each corner (worst of the four)
    sj = []
    for (p, q, r) in ((a, b, d), (b, c, a), (c, d, b), (d, a, c)):
        u1 = q - p; u2 = r - p
        n1 = np.hypot(u1[..., 0], u1[..., 1]) + 1e-9
        n2 = np.hypot(u2[..., 0], u2[..., 1]) + 1e-9
        sj.append(cross(u1, u2) / (n1 * n2))
    sj = np.min(np.stack(sj, -1), -1)
    sgn = np.sign(np.median(area))
    folded = int(np.sum(area * sgn <= 0))
    return dict(
        fold_pct=100.0 * folded / area.size,
        ar_median=float(np.median(ar)), ar_p95=float(np.percentile(ar, 95)),
        ar_max=float(ar.max()),
        sj_min=float((sj * sgn).min()), sj_p05=float(np.percentile(sj * sgn, 5)),
        area_ratio=float(np.percentile(np.abs(area), 95) /
                         max(np.percentile(np.abs(area), 5), 1e-9)),
    )


def density_contrast(B, pos):
    """Median cell area OUTSIDE the glyph / median cell area INSIDE it.
    This is the number that decides whether the word reads in a plain
    single-colour wireframe: the letter has to be a denser patch of lattice."""
    nx, ny = B["nx"], B["ny"]
    P = pos.reshape(ny + 1, nx + 1, 2)
    a = P[:-1, :-1]; b = P[:-1, 1:]; c = P[1:, 1:]; d = P[1:, :-1]
    def cr(u, v): return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    area = np.abs(0.5 * (cr(a, b) + cr(b, c) + cr(c, d) + cr(d, a)))
    ctr = 0.25 * (a + b + c + d)
    A = B["alpha_map"]
    xi = np.clip(np.rint(ctr[..., 0]).astype(int), 0, CANVAS_W - 1)
    yi = np.clip(np.rint(ctr[..., 1]).astype(int), 0, CANVAS_H - 1)
    ins = A[yi, xi] > 0.5
    if ins.sum() < 8 or (~ins).sum() < 8:
        return 1.0
    return float(np.median(area[~ins]) / max(np.median(area[ins]), 1e-9))


# ==================================================================== drawing
def draw_wireframe(pos, nx, ny, size=(CANVAS_W, CANVAS_H), ss=2,
                   lw=2, ink=INK, paper=PAPER, chain=None, chain_ink=None):
    W, H = size
    im = Image.new("RGB", (W * ss, H * ss), paper)
    d = ImageDraw.Draw(im)
    P = (pos.reshape(ny + 1, nx + 1, 2) * ss)
    for j in range(ny + 1):
        d.line([tuple(p) for p in P[j]], fill=ink, width=lw, joint="curve")
    for i in range(nx + 1):
        d.line([tuple(p) for p in P[:, i]], fill=ink, width=lw, joint="curve")
    if chain is not None and chain_ink is not None:
        for ch_ in chain:
            pts = [tuple(P[j, i]) for (i, j, *_r) in ch_]
            if len(pts) > 2:
                d.line(pts + [pts[0]], fill=chain_ink, width=lw * 2, joint="curve")
    return im.resize((W, H), Image.LANCZOS)


def cell_clip_ids(nx, ny, inside_cell, inside_family, outside_family, seed=3,
                  blob=0.55):
    """Shuffle clip ids among fine cells; inside/outside families disjoint."""
    rng = np.random.default_rng(seed)
    ids = np.zeros((ny, nx), dtype=np.uint8)
    for j in range(ny):
        for i in range(nx):
            fam = inside_family if inside_cell[j, i] else outside_family
            # mild spatial coherence so cells merge into organic blobs
            if i > 0 and rng.random() < blob and \
               (inside_cell[j, i - 1] == inside_cell[j, i]):
                ids[j, i] = ids[j, i - 1]
            elif j > 0 and rng.random() < blob and \
                    (inside_cell[j - 1, i] == inside_cell[j, i]):
                ids[j, i] = ids[j - 1, i]
            else:
                ids[j, i] = fam[int(rng.integers(len(fam)))]
    return ids


def draw_selector(pos, nx, ny, ids, size=(CANVAS_W, CANVAS_H), ss=2):
    W, H = size
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    P = pos.reshape(ny + 1, nx + 1, 2) * ss
    for j in range(ny):
        for i in range(nx):
            poly = [tuple(P[j, i]), tuple(P[j, i + 1]),
                    tuple(P[j + 1, i + 1]), tuple(P[j + 1, i])]
            v = int(ids[j, i]) + 1
            d.polygon(poly, fill=v, outline=v)
    return np.asarray(im.resize((W, H), Image.NEAREST)).astype(np.int16) - 1


def composite(sel, clips):
    H, W = sel.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for k, c in enumerate(clips):
        m = sel == k
        if m.any():
            out[m] = c[m]
    out[sel < 0] = 0
    return Image.fromarray(out)


def caption(im, text, h=30, bg=(18, 18, 22), fg=(245, 245, 245)):
    W, H = im.size
    out = Image.new("RGB", (W, H + h), bg)
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    except Exception:
        f = ImageFont.load_default()
    d.text((8, H + 6), text, fill=fg, font=f)
    return out


def grid_sheet(images, cols, path, pad=10, bg=(12, 12, 14)):
    if not images:
        return
    w = max(i.size[0] for i in images); h = max(i.size[1] for i in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad,
                              rows * h + (rows + 1) * pad), bg)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        sheet.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    sheet.save(path)
    return sheet


# ================================================================ the pipeline
def build(word, nx, ny, dilate_stroke=0.45, hole_cells=0.35, alpha=3e-3, order=1,
          solver="harmonic", overscan=0.06, sample_per_cell=0.5,
          corner_deg=38.0, miter_limit=2.0, dilate_cells=None,
          source_mode="offset", scale_k=0.7, rot_deg=0.0, verbose=True):
    W, H = CANVAS_W, CANVAS_H
    alpha_map, info = render_word(word)
    ox = -overscan * W / 2.0
    oy = -overscan * H / 2.0
    cw = W * (1 + overscan) / nx
    chh = H * (1 + overscan) / ny

    polys = marching_squares(alpha_map, 0.5)
    polys = [p for p in polys if len(p) > 16 and abs(signed_area(p)) > 30]
    if not polys:
        raise RuntimeError("no contours")
    # outer vs hole: a polygon is a hole if it sits inside another
    is_hole = []
    for k, p in enumerate(polys):
        c = p.mean(axis=0)
        hole = any(m != k and abs(signed_area(polys[m])) > abs(signed_area(p))
                   and point_in_poly(c, polys[m]) for m in range(len(polys)))
        is_hole.append(hole)

    step = sample_per_cell * min(cw, chh)
    stroke = stroke_width_estimate(alpha_map)
    # the gain that makes the word READ is how much grid material is funnelled
    # into the letter, so tie it to stroke width, not to cell size.
    d_px = (dilate_cells * min(cw, chh) if dilate_cells is not None
            else max(dilate_stroke, 0.0) * stroke)
    h_px = hole_cells * min(cw, chh)

    # Local feature size.  A constant offset is what destroys `e` and `3`: in a
    # narrow aperture the two walls' offsets collide and their chains fight over
    # the same lattice node.  Cap the offset per-sample by how much clear space
    # there actually is, measured on the mask's Euclidean distance transform.
    edt_out = ndimage.distance_transform_edt(alpha_map <= 0.5)
    edt_in = ndimage.distance_transform_edt(alpha_map > 0.5)

    def clear_offset(Pp, nrm, dmax, field, floor=0.25, keep=0.80, steps=10):
        if dmax <= 1e-6:
            return np.zeros(len(Pp))
        ts = np.linspace(dmax, dmax * floor, steps)
        ok = np.zeros((steps, len(Pp)), bool)
        for s, t in enumerate(ts):
            q = Pp + nrm * t
            xi = np.clip(np.rint(q[:, 0]).astype(int), 0, CANVAS_W - 1)
            yi = np.clip(np.rint(q[:, 1]).astype(int), 0, CANVAS_H - 1)
            ok[s] = field[yi, xi] >= keep * t
        first = np.argmax(ok, axis=0)
        first[~ok.any(axis=0)] = steps - 1
        d = ts[first]
        # smooth along the contour so the source polygon keeps a clean shape
        k = np.ones(5) / 5.0
        return np.convolve(np.r_[d[-4:], d, d[:4]], k, "same")[4:-4]

    # group each counter with the outer contour that owns it, so a whole
    # letter shares one centroid when source_mode == "scale"
    owner = []
    for k, p in enumerate(polys):
        c = p.mean(axis=0); best, ba = k, 1e18
        if is_hole[k]:
            for m in range(len(polys)):
                if m != k and not is_hole[m] and point_in_poly(c, polys[m]):
                    a = abs(signed_area(polys[m]))
                    if a < ba:
                        ba, best = a, m
        owner.append(best)
    cent = [polys[owner[k]].mean(axis=0) for k in range(len(polys))]

    chains, n_ctrl, n_corners = [], 0, 0
    dd_stats = []
    for ki, (p, hole) in enumerate(zip(polys, is_hole)):
        Pp, cf = resample_corner_aware(p, step, corner_deg=corner_deg)
        sa = signed_area(Pp)
        nrm = outward_normals(Pp, alpha_map)
        # counters get their OWN, much smaller offset: shrinking a counter in the
        # source blows it up in the target and the letterform turns to a blob.
        r_in = math.sqrt(abs(sa) / math.pi)
        if hole:
            dd = clear_offset(Pp, nrm, min(h_px, 0.18 * r_in), edt_in)
        else:
            dd = clear_offset(Pp, nrm, d_px, edt_out)
        dd_stats.append(dd)
        if source_mode == "scale":
            # LENS correspondence: the source is a similarity-scaled copy of the
            # letter about its own centroid.  k<1 -> the content inside the
            # letter is MAGNIFIED by 1/k, which is a far stronger cue in a pure
            # image warp than the thin crease an offset source produces.
            th = math.radians(rot_deg)
            R = np.array([[math.cos(th), -math.sin(th)],
                          [math.sin(th), math.cos(th)]])
            Q = cent[ki] + ((Pp - cent[ki]) * scale_k) @ R.T
        else:
            Q = miter_offset(Pp, dd, nrm, miter_limit)
        ch_ = build_chain(Q, Pp, cf, nx, ny, cw, chh, ox, oy)
        chains.append(ch_)
        n_ctrl += len(ch_); n_corners += int(cf.sum())

    cons, collisions, spread = collect_constraints(chains, nx, ny, cw, chh, ox, oy)

    ii, jj = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    base = np.column_stack([(ox + ii.ravel() * cw), (oy + jj.ravel() * chh)])
    cons_disp = {p: cons[p] - base[p] for p in cons}

    if solver == "tps":
        u, solve_t = solve_tps(nx, ny, cons, base, lam=alpha)
    else:
        u, solve_t = solve_biharmonic(nx, ny, cons_disp, alpha, order=order)

    return dict(word=word, nx=nx, ny=ny, cw=cw, ch=chh, ox=ox, oy=oy,
                base=base, u=u, chains=chains, cons=cons, alpha_map=alpha_map,
                collisions=collisions, spread=spread, n_ctrl=len(cons),
                n_corners=n_corners, solve_t=solve_t, info=info,
                d_px=d_px, h_px=h_px, polys=polys, is_hole=is_hole,
                d_applied=float(np.mean(np.concatenate(dd_stats))),
                stroke=stroke)


def posed(B, t=0.0, wob=0.0, mode="wobble-then-field"):
    nx, ny, cw, chh, ox, oy = B["nx"], B["ny"], B["cw"], B["ch"], B["ox"], B["oy"]
    base, u = B["base"], B["u"]
    if wob <= 0:
        return base + u
    w = wobble_field(nx, ny, cw, chh, wob, t)
    if mode == "field-then-wobble":
        return base + u + w
    p = base + w
    return p + sample_field(u, nx, ny, cw, chh, ox, oy, p)


def inside_cells(B):
    """Cell-centre-in-source-region test, used ONLY to pick clip families for
    the composite preview.  It plays no part in the deformation."""
    nx, ny = B["nx"], B["ny"]
    cw, chh, ox, oy = B["cw"], B["ch"], B["ox"], B["oy"]
    A = B["alpha_map"]
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    cx = ox + (ii + 0.5) * cw; cy = oy + (jj + 0.5) * chh
    # push the cell centre through the field, then test the TRUE glyph mask
    pts = np.column_stack([cx.ravel(), cy.ravel()])
    pts = pts + sample_field(B["u"], nx, ny, cw, chh, ox, oy, pts)
    xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, CANVAS_W - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, CANVAS_H - 1)
    return (A[yi, xi] > 0.5).reshape(ny, nx)


# ======================================================== IMAGE WARP (round 2)
# No selector mask, no clip ids, no compositing.  ONE video is pushed through
# exactly the same displacement field: the undeformed lattice is texture space,
# the warped lattice is screen space.  The letters must read purely as
# distortion of the footage.  Nothing is drawn on top of the image.  Ever.
import subprocess
try:
    import cv2
except Exception:
    cv2 = None


def wobble_smooth(base, t, amp, cw, ch, seed=11, nwaves=5):
    """Low-frequency, spatially SMOOTH breathing.  The per-node random-phase
    wobble used for the wireframe is white noise in space; pushed through an
    image it reads as chatter/grain.  A few plane waves read as cloth."""
    if amp <= 0:
        return np.zeros_like(base)
    rng = np.random.default_rng(seed)
    x = base[:, 0] / CANVAS_W; y = base[:, 1] / CANVAS_H
    acc = np.zeros_like(base)
    for _ in range(nwaves):
        fx, fy = rng.integers(1, 4), rng.integers(1, 4)
        h = int(rng.integers(1, 3))
        ph = rng.uniform(0, 2 * np.pi, 2)
        dirn = rng.uniform(0, 2 * np.pi)
        s = np.sin(2 * np.pi * (fx * x + fy * y + h * t / LOOP_SECONDS) + ph[0])
        acc[:, 0] += s * math.cos(dirn)
        acc[:, 1] += s * math.sin(dirn)
    acc /= math.sqrt(nwaves)
    return acc * amp * np.array([cw, ch])[None, :]


def attenuation(t, T=LOOP_SECONDS):
    """0 -> 1 -> hold -> 0, C1 at every join, exact loop.
    idle 0-1s, settle 1-4s, HOLD 4-7.5s, release 7.5-10.5s, idle 10.5-12s."""
    t = t % T
    if t < 1.0:
        return 0.0
    if t < 4.0:
        return 0.5 - 0.5 * math.cos(math.pi * (t - 1.0) / 3.0)
    if t < 7.5:
        return 1.0
    if t < 10.5:
        return 0.5 + 0.5 * math.cos(math.pi * (t - 7.5) / 3.0)
    return 0.0


def node_disp(B, t, amp=1.0, wob=0.35):
    """Total displacement at the UNDEFORMED node positions, at time t.
    The wobble breathes always; only the letter field is attenuated."""
    base, u = B["base"], B["u"]
    nx, ny, cw, chh, ox, oy = B["nx"], B["ny"], B["cw"], B["ch"], B["ox"], B["oy"]
    w = wobble_smooth(base, t, wob, cw, chh)
    if amp <= 0:
        return w
    p = base + w
    return w + amp * sample_field(u, nx, ny, cw, chh, ox, oy, p)


def full_field(B, D):
    """Bilinear upsample of the node field D to one float32 (H,W,2) canvas
    field, in pixel units.  Done with a fixed remap, once per frame."""
    nx, ny, cw, chh, ox, oy = B["nx"], B["ny"], B["cw"], B["ch"], B["ox"], B["oy"]
    key = ("gm", nx, ny)
    if key not in _CACHE:
        xs = np.arange(CANVAS_W, dtype=np.float32)
        ys = np.arange(CANVAS_H, dtype=np.float32)
        gx = ((xs - ox) / cw).astype(np.float32)
        gy = ((ys - oy) / chh).astype(np.float32)
        _CACHE[key] = (np.tile(gx, (CANVAS_H, 1)),
                       np.repeat(gy[:, None], CANVAS_W, axis=1))
    gx, gy = _CACHE[key]
    Dn = D.reshape(ny + 1, nx + 1, 2).astype(np.float32)
    return cv2.remap(Dn, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_CACHE = {}


def inverse_map(Dfull, iters=12):
    """Destination pixel -> source (texture) pixel.

    Forward map is  F(x) = x + D(x)  with x in TEXTURE space.  We need F^-1.
    Fixed point:  s <- d - D(s).  D is a smooth contraction almost everywhere
    (|grad D| < 1 wherever the mesh is unfolded), so this converges in a few
    iterations and costs one bilinear resample each.  Returns (sx, sy, resid)."""
    H, W = CANVAS_H, CANVAS_W
    key = ("id", W, H)
    if key not in _CACHE:
        _CACHE[key] = (np.tile(np.arange(W, dtype=np.float32), (H, 1)),
                       np.repeat(np.arange(H, dtype=np.float32)[:, None], W, 1))
    dx, dy = _CACHE[key]
    sx, sy = dx.copy(), dy.copy()
    for _ in range(iters):
        Ds = cv2.remap(Dfull, sx, sy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        sx = dx - Ds[..., 0]
        sy = dy - Ds[..., 1]
    Ds = cv2.remap(Dfull, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    resid = np.hypot(sx + Ds[..., 0] - dx, sy + Ds[..., 1] - dy)
    return sx, sy, resid


def warp_frame(img, B, t, amp, wob=0.35, iters=12):
    D = node_disp(B, t, amp, wob)
    Dfull = full_field(B, D)
    sx, sy, resid = inverse_map(Dfull, iters)
    out = cv2.remap(img, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, float(resid.max()), float(resid.mean())


def ffmpeg_frames(path, seconds=LOOP_SECONDS, fps=FPS):
    """Yield RGB uint8 frames, looping the clip to fill `seconds`."""
    cmd = ["ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", path,
           "-t", str(seconds), "-r", str(fps),
           "-vf", "scale=%d:%d" % (CANVAS_W, CANVAS_H),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = CANVAS_W * CANVAS_H * 3
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 4)
    while True:
        buf = pr.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(CANVAS_H, CANVAS_W, 3)
    pr.stdout.close(); pr.wait()


def ffmpeg_writer(path, fps=FPS):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (CANVAS_W, CANVAS_H), "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-preset", "medium", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ====================================================================== main
def load_clips():
    cl = []
    for k in range(4):
        pth = os.path.join(FRAMES, "clip%d.png" % k)
        cl.append(np.asarray(Image.open(pth).convert("RGB").resize(
            (CANVAS_W, CANVAS_H))).astype(np.uint8))
    return cl


# Tuned defaults, found by the sweeps below.
BEST = dict(dilate_stroke=0.30, hole_cells=0.35, alpha=3e-4, order=1,
            sample_per_cell=0.5, corner_deg=38.0, miter_limit=2.0)
DENS = [(40, 23), (56, 32), (72, 41), (88, 50), (112, 63), (144, 81), (176, 99)]
WORDS = ["Yope3D", "SpinStack", "3D Gravity Simulator"]

LOG = []


def P(*s):
    line = " ".join(str(x) for x in s)
    print(line, flush=True); LOG.append(line)


def measure(B, pos):
    m = quad_metrics(pos, B["nx"], B["ny"])
    m["contrast"] = density_contrast(B, pos)
    return m


def stage_density():
    P("=== density sweep : Yope3D, tuned gain ===")
    ims, rows = [], []
    for nx, ny in DENS:
        t0 = time.time(); B = build("Yope3D", nx, ny, **BEST)
        pos = posed(B); m = measure(B, pos); dt = time.time() - t0
        tag = ("%dx%d cell=%.1fpx stroke/cell=%.1f ctrl=%d coll=%d contrast=%.2f "
               "fold=%.2f%% AR50=%.2f AR95=%.2f" %
               (nx, ny, B["cw"], B["stroke"] / B["cw"], B["n_ctrl"],
                B["collisions"], m["contrast"], m["fold_pct"], m["ar_median"],
                m["ar_p95"]))
        P(tag, "build=%.2fs" % dt)
        rows.append(dict(nx=nx, ny=ny, cell=B["cw"], stroke=B["stroke"],
                         ctrl=B["n_ctrl"], coll=B["collisions"],
                         build_s=dt, **m))
        ims.append(caption(draw_wireframe(pos, nx, ny).resize((860, 484)), tag))
    grid_sheet(ims, 2, os.path.join(OUT, "wireframe-sheet.png"))
    json.dump(rows, open(os.path.join(OUT, "density.json"), "w"), indent=1)
    P("wrote wireframe-sheet.png")


def stage_wireframe(nx=112, ny=63):
    """THE ACCEPTANCE TEST: one colour, uniform thin lines, nothing else."""
    B = build("Yope3D", nx, ny, **BEST)
    draw_wireframe(posed(B), nx, ny).save(os.path.join(OUT, "wireframe.png"))
    draw_wireframe(posed(B, t=3.1, wob=0.30), nx, ny).save(
        os.path.join(OUT, "wireframe-wobble.png"))
    P("wrote wireframe.png  (%dx%d, cell %.1fpx)" % (nx, ny, B["cw"]))
    return B


def stage_words():
    P("=== words x density ===")
    ims, rows = [], []
    for w in WORDS:
        for nx, ny in [(72, 41), (112, 63), (176, 99), (224, 126)]:
            t0 = time.time(); B = build(w, nx, ny, **BEST)
            pos = posed(B); m = measure(B, pos); dt = time.time() - t0
            tag = ("%-20s %dx%d cell=%.1f cap=%d stroke=%.0f s/c=%.1f coll=%d "
                   "fold=%.2f%% AR95=%.2f" %
                   (w, nx, ny, B["cw"], B["info"]["cap_h"], B["stroke"],
                    B["stroke"] / B["cw"], B["collisions"], m["fold_pct"],
                    m["ar_p95"]))
            P(tag, "%.2fs" % dt)
            rows.append(dict(word=w, nx=nx, ny=ny, cell=B["cw"],
                             cap=B["info"]["cap_h"], stroke=B["stroke"],
                             coll=B["collisions"], build_s=dt, **m))
            ims.append(caption(draw_wireframe(pos, nx, ny).resize((640, 360)), tag))
    grid_sheet(ims, 4, os.path.join(OUT, "words-sheet.png"))
    json.dump(rows, open(os.path.join(OUT, "words.json"), "w"), indent=1)
    P("wrote words-sheet.png")


def stage_wobble(nx=112, ny=63):
    P("=== wobble ===")
    B = build("Yope3D", nx, ny, **BEST)
    ims = []
    for wob in (0.0, 0.15, 0.30, 0.50):
        for t in (0.0, 4.0):
            pos = posed(B, t=t, wob=wob)
            m = measure(B, pos)
            tag = ("wobble=%.2f cell  t=%.1fs  fold=%.2f%% AR95=%.2f contrast=%.2f"
                   % (wob, t, m["fold_pct"], m["ar_p95"], m["contrast"]))
            P(tag)
            ims.append(caption(draw_wireframe(pos, nx, ny).resize((640, 360)), tag))
    grid_sheet(ims, 2, os.path.join(OUT, "wobble-sheet.png"))
    P("wrote wobble-sheet.png")


def stage_loop(nx=112, ny=63, wob=0.30):
    """12 s loop: per-frame cost, and pop detection (max jerk between frames)."""
    P("=== 12 s loop @ %d fps ===" % FPS)
    B = build("Yope3D", nx, ny, **BEST)
    n = int(LOOP_SECONDS * FPS)
    prev = prev_v = None
    vmax, jmax = 0.0, 0.0
    t0 = time.time()
    for k in range(n + 1):
        t = k / FPS
        pos = posed(B, t=t, wob=wob)
        if prev is not None:
            v = pos - prev
            vmax = max(vmax, float(np.hypot(v[:, 0], v[:, 1]).max()))
            if prev_v is not None:
                j = v - prev_v
                jmax = max(jmax, float(np.hypot(j[:, 0], j[:, 1]).max()))
            prev_v = v
        prev = pos
    per = (time.time() - t0) / (n + 1)
    wrap = np.hypot(*(posed(B, t=LOOP_SECONDS, wob=wob) -
                      posed(B, t=0.0, wob=wob)).T[::-1]).max()
    P("field cost/frame = %.2f ms (numpy, %d nodes)  max step %.2f px  "
      "max jerk %.3f px  loop wrap error %.2e px" %
      (per * 1e3, (nx + 1) * (ny + 1), vmax, jmax, wrap))
    return dict(ms_per_frame=per * 1e3, max_step_px=vmax, max_jerk_px=jmax,
                wrap_px=float(wrap))


def stage_tps(nx=88, ny=50):
    P("=== thin-plate spline vs harmonic solve, same control pairs ===")
    ims = []
    out = {}
    for solver in ("harmonic", "tps"):
        t0 = time.time()
        B = build("Yope3D", nx, ny, solver=solver,
                  **{k: v for k, v in BEST.items() if k != "order" or
                     solver != "tps"})
        pos = posed(B); m = measure(B, pos); dt = time.time() - t0
        tag = ("%s  ctrl=%d  solve=%.2fs  fold=%.2f%%  AR95=%.2f  contrast=%.2f"
               % (solver, B["n_ctrl"], B["solve_t"], m["fold_pct"], m["ar_p95"],
                  m["contrast"]))
        P(tag, "total %.2fs" % dt)
        out[solver] = dict(solve_s=B["solve_t"], total_s=dt, **m)
        ims.append(caption(draw_wireframe(pos, nx, ny).resize((860, 484)), tag))
    grid_sheet(ims, 2, os.path.join(OUT, "solver-sheet.png"))
    json.dump(out, open(os.path.join(OUT, "solver.json"), "w"), indent=1)
    P("wrote solver-sheet.png")


def stage_contact():
    P("=== composite contact sheet ===")
    clips = load_clips()
    # clip0 mean-luma 19, clip1 127, clip2 29, clip3 0 (gravity.mp4 is a black
    # placeholder).  Inside gets the two brightest, outside the two darkest, so
    # the seam survives against real footage.
    INSIDE, OUTSIDE = (1, 2), (0, 3)
    ims = []
    for w in WORDS:
        for nx, ny in ((88, 50), (144, 81)):
            B = build(w, nx, ny, **BEST)
            for wob in (0.0, 0.30):
                pos = posed(B, t=2.7, wob=wob)
                ids = cell_clip_ids(nx, ny, inside_cells(B), INSIDE, OUTSIDE)
                sel = draw_selector(pos, nx, ny, ids)
                im = composite(sel, clips)
                tag = "%s  %dx%d cell=%.1fpx  wobble=%.2f" % (w, nx, ny, B["cw"], wob)
                P(tag)
                ims.append(caption(im.resize((640, 360)), tag))
    grid_sheet(ims, 4, os.path.join(OUT, "contact-sheet.png"))
    P("wrote contact-sheet.png")


# Peak configuration for the IMAGE WARP.  Different from the wireframe optimum:
# in a warp the readable cue is an ORIENTATION discontinuity at the letter edge,
# not the crease and not density, so the source is a rotated+scaled copy of the
# letter rather than an outward offset of its outline.
WARP = dict(alpha=3e-4, order=1, sample_per_cell=0.5, corner_deg=38.0,
            hole_cells=0.35, source_mode="scale", scale_k=0.82, rot_deg=25.0)
WOB = 0.35


def _src(name):
    return np.asarray(Image.open(os.path.join(OUT, "src", name + ".png"))
                      .convert("RGB"))


def stage_warpgain(nx=112, ny=63):
    """Peak frame across the gain sweep.  Both families, because for a warp the
    right gain AND the right kind of gain are both open questions."""
    P("=== warp gain sweep ===")
    ims = []
    cfgs = ([("offset d=%.1f*stroke" % d,
              dict(source_mode="offset", dilate_stroke=d)) for d in (0.3, 0.8, 1.5)] +
            [("lens k=%.2f rot=%d deg" % (k, r),
              dict(source_mode="scale", scale_k=k, rot_deg=r))
             for k, r in ((0.55, 0), (0.82, 25), (0.70, 25))])
    for label, kw in cfgs:
        base = {k: v for k, v in WARP.items()
                if k not in ("source_mode", "scale_k", "rot_deg")}
        B = build("Yope3D", nx, ny, **{**base, **kw})
        m = quad_metrics(posed(B), nx, ny)
        for n in ("cloth", "ctrl"):
            out, rmax, rmean = warp_frame(_src(n), B, 5.0, 1.0, wob=0.0)
            tag = ("%s | %s | fold=%.2f%% AR95=%.2f resid max/mean=%.1f/%.2fpx"
                   % (n, label, m["fold_pct"], m["ar_p95"], rmax, rmean))
            P(tag)
            ims.append(caption(Image.fromarray(out).resize((640, 360)), tag))
    grid_sheet(ims, 2, os.path.join(OUT, "warp-gain.png"))
    P("wrote warp-gain.png")


def stage_warpcycle(nx=112, ny=63):
    P("=== attenuation cycle ===")
    B = build("Yope3D", nx, ny, **WARP)
    m = quad_metrics(posed(B), nx, ny)
    P("peak mesh: fold=%.2f%% AR50=%.2f AR95=%.2f" %
      (m["fold_pct"], m["ar_median"], m["ar_p95"]))

    # full-res hold frame
    cloth = _src("cloth")
    out, rmax, rmean = warp_frame(cloth, B, 5.5, 1.0, wob=WOB)
    Image.fromarray(out).save(os.path.join(OUT, "warp-hold.png"))
    P("wrote warp-hold.png  (inverse residual max %.1f mean %.3f px)" % (rmax, rmean))

    # labelled strip across the cycle
    ims = []
    for t in (0.5, 1.8, 2.5, 3.2, 4.5, 6.0, 8.2, 9.2, 11.0):
        a = attenuation(t)
        out, _, _ = warp_frame(cloth, B, t, a, wob=WOB)
        tag = "t=%.1fs   amplitude a=%.2f" % (t, a)
        P(tag)
        ims.append(caption(Image.fromarray(out).resize((640, 360)), tag))
    grid_sheet(ims, 3, os.path.join(OUT, "warp-strip.png"))
    P("wrote warp-strip.png")

    # same strip on the high-frequency control texture, to show the ceiling
    ims = []
    ctrl = _src("ctrl")
    for t in (0.5, 2.5, 3.2, 4.5, 6.0, 9.2):
        a = attenuation(t)
        out, _, _ = warp_frame(ctrl, B, t, a, wob=WOB)
        ims.append(caption(Image.fromarray(out).resize((640, 360)),
                           "control texture  t=%.1fs  a=%.2f" % (t, a)))
    grid_sheet(ims, 3, os.path.join(OUT, "warp-strip-ctrl.png"))
    P("wrote warp-strip-ctrl.png")
    return B


def stage_warpvideo(B=None, nx=112, ny=63):
    P("=== cycle videos ===")
    if B is None:
        B = build("Yope3D", nx, ny, **WARP)
    jobs = [("public/media/yope_cloth_b.mp4", "warp-cycle.mp4"),
            ("public/media/showcase-test/digits.mp4", "warp-cycle-digits.mp4")]
    root = "/Users/me/Desktop/dev/yugumishra.github.io"
    for src, name in jobs:
        path = os.path.join(root, src)
        if not os.path.exists(path):
            P("missing " + path); continue
        w = ffmpeg_writer(os.path.join(OUT, name))
        t0 = time.time(); k = 0
        for frame in ffmpeg_frames(path):
            t = k / FPS
            out, _, _ = warp_frame(frame, B, t, attenuation(t), wob=WOB)
            w.stdin.write(np.ascontiguousarray(out).tobytes())
            k += 1
        w.stdin.close(); w.wait()
        P("wrote %s  (%d frames, %.1f ms/frame end to end)" %
          (name, k, 1e3 * (time.time() - t0) / max(k, 1)))


# ================================================= JACOBIAN BRIGHTNESS VARIANT
# Brightness modulated by the local area change of the warp itself.  No glyph
# mask is consulted: `det J` is a property of the deformation field, so the term
# is identically 1 wherever the warp is 1:1 and vanishes when amplitude is 0.
# Physical reading: squeeze the material and you get more radiance per unit
# area; stretch it and you get less.

def atten2(t, t0, t1, t2, t3, T=LOOP_SECONDS):
    """0 -> 1 over [t0,t1], hold to t2, back to 0 by t3.  C1, exact loop."""
    t = t % T
    if t <= t0 or t >= t3:
        return 0.0
    if t < t1:
        return 0.5 - 0.5 * math.cos(math.pi * (t - t0) / (t1 - t0))
    if t < t2:
        return 1.0
    return 0.5 + 0.5 * math.cos(math.pi * (t - t2) / (t3 - t2))


RAMP_FAST = (1.0, 4.0, 7.5, 10.5)      # 3.0 s settle
RAMP_SLOW = (0.4, 6.4, 8.6, 11.6)      # 6.0 s settle


def jacobian_shade(sx, sy, strength=0.75, blur=3, lo=0.40, hi=2.50):
    """Multiplier from det J of the inverse map.  det>1 = compressed = brighter.
    `strength` is the exponent, so strength=0 is exactly no effect."""
    dsxdy, dsxdx = np.gradient(sx)
    dsydy, dsydx = np.gradient(sy)
    det = dsxdx * dsydy - dsxdy * dsydx
    det = np.clip(det, lo, hi)
    if blur:
        # kills the pixel speckle that np.gradient produces on folded quads.
        # Purely a denoise of the term; it does not move the band.
        det = cv2.GaussianBlur(det, (0, 0), blur)
    r = det / max(float(np.median(det)), 1e-6)
    return np.power(r, strength).astype(np.float32)


def warp_frame_j(img, B, t, amp, wob=0.35, strength=0.75, blur=3, iters=6):
    D = node_disp(B, t, amp, wob)
    Dfull = full_field(B, D)
    sx, sy, _ = inverse_map(Dfull, iters)
    out = cv2.remap(img, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if strength <= 0:
        return out, np.ones((CANVAS_H, CANVAS_W), np.float32)
    sh = jacobian_shade(sx, sy, strength, blur)
    out = np.clip(out.astype(np.float32) * sh[..., None], 0, 255).astype(np.uint8)
    return out, sh


def stage_jacobian(nx=112, ny=63):
    P("=== JACOBIAN BRIGHTNESS: full cycles on MOVING footage ===")
    B = build("Yope3D", nx, ny, **WARP)
    root = "/Users/me/Desktop/dev/yugumishra.github.io"
    STR = 0.55

    # ---- strength sweep at peak amplitude, on a moving-video frame ----------
    P("--- strength sweep (peak amplitude) ---")
    ims = []
    gen = ffmpeg_frames(os.path.join(root, "public/media/yope_cloth_b.mp4"))
    peak = None
    for k, fr in enumerate(gen):
        if k == 165:
            peak = fr.copy(); break
    for st in (0.0, 0.35, 0.55, 0.75, 1.10, 1.60):
        out, sh = warp_frame_j(peak, B, 5.5, 1.0, strength=st)
        tag = ("strength=%.2f   shade p1=%.2f p99=%.2f  (1.00 = untouched)"
               % (st, np.percentile(sh, 1), np.percentile(sh, 99)))
        P(tag)
        ims.append(caption(Image.fromarray(out).resize((640, 360)), tag))
    grid_sheet(ims, 2, os.path.join(OUT, "jacobian-strength.png"))
    P("wrote jacobian-strength.png")

    # ---- the cycles, decoding the moving clip frame by frame ----------------
    stats = {}
    jobs = [("public/media/yope_cloth_b.mp4", "jacobian-cycle.mp4", RAMP_FAST, True),
            ("public/media/yope_cloth_b.mp4", "jacobian-cycle-slow.mp4", RAMP_SLOW, True),
            ("public/media/showcase-test/digits.mp4", "jacobian-cycle-digits.mp4",
             RAMP_SLOW, False)]
    for src, name, ramp, want_strip in jobs:
        path = os.path.join(root, src)
        if not os.path.exists(path):
            P("missing " + path); continue
        w = ffmpeg_writer(os.path.join(OUT, name))
        strip, prev_sh, dsh, shmin, shmax = [], None, [], 9e9, -9e9
        # 12 frames evenly spaced across the RAMP-UP only, for the formation strip
        t0, t1 = ramp[0], ramp[1]
        want = [t0 + (t1 - t0) * i / 11.0 for i in range(12)]
        t_start = time.time(); k = 0
        for frame in ffmpeg_frames(path):
            t = k / FPS
            a = atten2(t, *ramp)
            out, sh = warp_frame_j(frame, B, t, a, strength=STR)
            w.stdin.write(np.ascontiguousarray(out).tobytes())
            if prev_sh is not None:
                dsh.append(float(np.percentile(np.abs(sh - prev_sh), 99.9)))
            prev_sh = sh
            shmin = min(shmin, float(sh.min())); shmax = max(shmax, float(sh.max()))
            if want_strip and want and t >= want[0] - 1e-9:
                want.pop(0)
                strip.append(caption(Image.fromarray(out).resize((640, 360)),
                                     "t=%.2fs   a=%.3f" % (t, a)))
            k += 1
        w.stdin.close(); w.wait()
        ms = 1e3 * (time.time() - t_start) / max(k, 1)
        P("wrote %s  (%d frames, %.1f ms/frame end to end, ramp %s)"
          % (name, k, ms, str(ramp)))
        P("   shade range over the whole cycle: %.3f .. %.3f" % (shmin, shmax))
        P("   frame-to-frame |d shade| p99.9: median %.4f  max %.4f"
          % (float(np.median(dsh)), float(np.max(dsh))))
        stats[name] = dict(ms_per_frame=ms, shade_min=shmin, shade_max=shmax,
                           dshade_med=float(np.median(dsh)),
                           dshade_max=float(np.max(dsh)))
        if want_strip and name == "jacobian-cycle-slow.mp4" and strip:
            grid_sheet(strip, 4, os.path.join(OUT, "jacobian-formation-strip.png"))
            P("wrote jacobian-formation-strip.png (from the SLOW ramp)")
    json.dump(stats, open(os.path.join(OUT, "jacobian.json"), "w"), indent=1)


# ================================== ROUND 4: STRENGTH AS ITS OWN ANIMATED CHANNEL
# Salience and legibility are different jobs.  High strength is salient but the
# interiors crush; low strength is legible but quiet.  So strength gets its own
# curve s(t), decoupled from the geometry amplitude a(t).

RAMP_SHIP = (0.8, 5.3, 8.3, 11.3)      # 4.5 s settle, 3.0 s hold, 3.0 s release
S_BASE, S_PEAK = 0.55, 1.70


def _bump(x, c=0.45, w=0.28):
    """Flare centred on the amplitude at which the letters are still forming."""
    return math.exp(-((x - c) / w) ** 2)


def strength_sched(name, t, ramp=RAMP_SHIP):
    a = atten2(t, *ramp)
    t0, t1, t2, t3 = ramp
    k = S_PEAK - S_BASE
    if name == "const":
        return S_BASE
    if name == "overshoot":
        return S_BASE + k * _bump(a) if t <= t1 else S_BASE
    if name == "inverse":
        return 0.30 + 0.70 * a ** 3
    if name == "lead":
        return (S_BASE + k * _bump(atten2(t + 0.55, *ramp))) if t <= t1 else S_BASE
    if name == "lag":
        if t <= t1:
            return S_BASE + k * _bump(a)
        if t < t2:
            return S_BASE
        return S_BASE + 0.75 * max(0.0, atten2(t - 0.7, *ramp) - a) * 2.0
    if name == "ship":
        # RECOMMENDED: overshoot flare while the letters form (salience), relax
        # to the legible 0.55 for the hold, then collapse strength AHEAD of the
        # geometry on the way out so the word disperses instead of rewinding.
        if t <= t1:
            return S_BASE + k * _bump(a)
        if t < t2:
            return S_BASE
        f = (t - t2) / (t3 - t2)
        return S_BASE * max(0.0, 1.0 - 2.4 * f)
    raise ValueError(name)


SCHEDULES = ["const", "overshoot", "inverse", "lead", "lag"]
SHIP = "ship"


def shade_base(sx, sy, blur=3, lo=0.40, hi=2.50):
    """Everything in the Jacobian term that does NOT depend on strength."""
    dsxdy, dsxdx = np.gradient(sx)
    dsydy, dsydx = np.gradient(sy)
    det = np.clip(dsxdx * dsydy - dsxdy * dsydx, lo, hi)
    if blur:
        det = cv2.GaussianBlur(det, (0, 0), blur)
    return (det / max(float(np.median(det)), 1e-6)).astype(np.float32)


def apply_shade(warped, r, strength):
    if strength <= 0:
        return warped
    sh = np.power(r, strength)
    return np.clip(warped.astype(np.float32) * sh[..., None], 0, 255).astype(np.uint8)


def ffmpeg_writer_sz(path, w, h, fps=FPS):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-preset", "medium", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def stage_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.linspace(0, LOOP_SECONDS, 1200)
    a = np.array([atten2(t, *RAMP_SHIP) for t in ts])
    names = SCHEDULES + ["ship"]
    fig, ax = plt.subplots(len(names), 1, figsize=(9, 13), sharex=True)
    for i, nm in enumerate(names):
        sv = np.array([strength_sched(nm, t) for t in ts])
        ax[i].plot(ts, a, lw=2, color="#3b78c3", label="a(t)  geometry amplitude")
        ax[i].plot(ts, sv, lw=2, color="#d1603d", label="s(t)  Jacobian strength")
        ax[i].axhline(0.55, ls=":", lw=1, color="#888")
        ax[i].axhline(0.75, ls="--", lw=1, color="#b33",
                      label="0.75 = top of the 'not a sticker' range")
        ax[i].axhspan(0.35, 0.75, color="#4a4", alpha=0.08)
        ax[i].set_ylim(0, 2.0)
        ax[i].set_ylabel(nm + ("  <-- SHIP" if nm == "ship" else ""), fontsize=10)
        ax[i].grid(alpha=0.25)
        if i == 0:
            ax[i].legend(fontsize=8, loc="upper right")
    ax[-1].set_xlabel("t (s), 12 s loop,  ramp = %s" % (RAMP_SHIP,))
    fig.suptitle("Geometry amplitude vs Jacobian strength, per schedule", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "strength-curves.png"), dpi=110)
    P("wrote strength-curves.png")


def stage_strength(nx=112, ny=63):
    P("=== ROUND 4: animated strength channel ===")
    B = build("Yope3D", nx, ny, **WARP)
    root = "/Users/me/Desktop/dev/yugumishra.github.io"
    clip = os.path.join(root, "public/media/yope_cloth_b.mp4")
    stage_curves()

    # One warp per frame, five shadings -> all schedules share the inverse map.
    writers = {nm: ffmpeg_writer(os.path.join(OUT, "strength-%s.mp4" % nm))
               for nm in SCHEDULES}
    mont = ffmpeg_writer_sz(os.path.join(OUT, "strength-schedules.mp4"), 1920, 1080)
    strip_t = [1.6, 2.3, 3.0, 3.7, 4.4, 6.8, 9.4, 10.4]
    strips = {nm: [] for nm in SCHEDULES}
    want = list(strip_t)
    k = 0; t_start = time.time()
    for frame in ffmpeg_frames(clip):
        t = k / FPS
        a = atten2(t, *RAMP_SHIP)
        D = node_disp(B, t, a, WOB)
        sx, sy, _ = inverse_map(full_field(B, D), 6)
        warped = cv2.remap(frame, sx, sy, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        r = shade_base(sx, sy)
        panels = []
        for nm in SCHEDULES:
            sv = strength_sched(nm, t)
            out = apply_shade(warped, r, sv)
            writers[nm].stdin.write(np.ascontiguousarray(out).tobytes())
            small = cv2.resize(out, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(small, "%s  a=%.2f s=%.2f" % (nm, a, sv), (10, 348),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(small)
            if want and t >= want[0] - 1e-9:
                strips[nm].append(caption(Image.fromarray(out).resize((520, 293)),
                                          "%s  t=%.1f  a=%.2f  s=%.2f"
                                          % (nm, t, a, sv), h=24))
        if want and t >= want[0] - 1e-9:
            want.pop(0)
        blank = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(blank, "t=%.2fs   a=%.3f" % (t, a), (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        grid = np.vstack([np.hstack(panels[0:3]), np.hstack(panels[3:5] + [blank])])
        mont.stdin.write(np.ascontiguousarray(grid).tobytes())
        k += 1
    for w in writers.values():
        w.stdin.close(); w.wait()
    mont.stdin.close(); mont.wait()
    P("rendered %d frames x %d schedules, %.1f ms/frame for all five"
      % (k, len(SCHEDULES), 1e3 * (time.time() - t_start) / max(k, 1)))
    rows = []
    for nm in SCHEDULES:
        rows.extend(strips[nm])
    grid_sheet(rows, len(strip_t), os.path.join(OUT, "strength-schedule-strip.png"))
    P("wrote strength-schedule-strip.png (rows = schedules, cols = matched t)")


# ------------------------------------------------------------------ TASK 2
def stage_ship(nx=112, ny=63):
    """The single recommended render: schedule `ship` on the moving cloth."""
    P("=== SHIP: overshoot in, 0.55 hold, strength-first out ===")
    B = build("Yope3D", nx, ny, **WARP)
    clip = ("/Users/me/Desktop/dev/yugumishra.github.io/"
            "public/media/yope_cloth_b.mp4")
    w = ffmpeg_writer(os.path.join(OUT, "strength-ship.mp4"))
    k = 0; t0 = time.time()
    for frame in ffmpeg_frames(clip):
        t = k / FPS
        a = atten2(t, *RAMP_SHIP)
        D = node_disp(B, t, a, WOB)
        sx, sy, _ = inverse_map(full_field(B, D), 6)
        wf = cv2.remap(frame, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out = apply_shade(wf, shade_base(sx, sy), strength_sched("ship", t))
        w.stdin.write(np.ascontiguousarray(out).tobytes())
        k += 1
    w.stdin.close(); w.wait()
    P("wrote strength-ship.mp4 (%d frames, %.1f ms/frame)"
      % (k, 1e3 * (time.time() - t0) / max(k, 1)))


def stage_release(nx=112, ny=63):
    P("=== release comparison ===")
    B = build("Yope3D", nx, ny, **WARP)
    root = "/Users/me/Desktop/dev/yugumishra.github.io"
    clip = os.path.join(root, "public/media/yope_cloth_b.mp4")
    A_SYM = RAMP_SHIP
    A_ASYM = (0.8, 5.3, 9.8, 11.3)          # same settle, 1.5 s release
    mont = ffmpeg_writer_sz(os.path.join(OUT, "release-compare.mp4"), 1920, 360)
    strip = []
    want = [8.3, 8.9, 9.5, 10.1, 10.7, 11.3]
    k = 0
    for frame in ffmpeg_frames(clip):
        t = k / FPS
        panels = []
        for label, ramp, smode in (
                ("A REWIND: symmetric a, s held at 0.55 out", A_SYM, "const"),
                ("B STRENGTH FIRST: s collapses ahead of a", A_SYM, "sfast"),
                ("C ASYMMETRIC a: 1.5s release, s held", A_ASYM, "const")):
            a = atten2(t, *ramp)
            if smode == "sfast":
                # on the way out, strength collapses well ahead of the geometry:
                # the word loses its edge while the mesh is still deformed
                if t < ramp[2]:
                    sv = strength_sched("overshoot", t, ramp)
                else:
                    f = (t - ramp[2]) / (ramp[3] - ramp[2])
                    sv = S_BASE * max(0.0, 1.0 - 2.4 * f)
            else:
                sv = S_BASE if t <= ramp[2] else S_BASE
                if t <= ramp[1]:
                    sv = strength_sched("overshoot", t, ramp)
            D = node_disp(B, t, a, WOB)
            sx, sy, _ = inverse_map(full_field(B, D), 6)
            w = cv2.remap(frame, sx, sy, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
            out = apply_shade(w, shade_base(sx, sy), sv)
            sm = cv2.resize(out, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(sm, "%s  a=%.2f s=%.2f" % (label, a, sv), (8, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(sm)
        mont.stdin.write(np.ascontiguousarray(np.hstack(panels)).tobytes())
        if want and t >= want[0] - 1e-9:
            want.pop(0)
            strip.append(caption(Image.fromarray(np.hstack(panels)), "t=%.1fs" % t))
        k += 1
    mont.stdin.close(); mont.wait()
    grid_sheet(strip, 1, os.path.join(OUT, "release-compare-strip.png"))
    P("wrote release-compare.mp4 + release-compare-strip.png (%d frames)" % k)


# ------------------------------------------------------------------ TASK 3
def luma(img):
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] +
            0.0722 * img[..., 2]).astype(np.float32)


def stage_adaptive(nx=112, ny=63):
    P("=== TASK 3: spatially adaptive strength in dark passages ===")
    B = build("Yope3D", nx, ny, **WARP)
    root = "/Users/me/Desktop/dev/yugumishra.github.io"
    ims = []
    for clipname, path in (("digits", "public/media/showcase-test/digits.mp4"),
                           ("cloth", "public/media/yope_cloth_b.mp4")):
        frame = None
        for i, f in enumerate(ffmpeg_frames(os.path.join(root, path))):
            if i == 165:
                frame = f.copy(); break
        L = luma(frame)
        P("%s: frac luma==0 %.3f  frac<8 %.3f  mean %.1f"
          % (clipname, float((L == 0).mean()), float((L < 8).mean()), float(L.mean())))
        D = node_disp(B, 5.5, 1.0, WOB)
        sx, sy, _ = inverse_map(full_field(B, D), 6)
        w = cv2.remap(frame, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        Lw = luma(w) / 255.0
        r = shade_base(sx, sy)
        # (a) flat
        ims.append(caption(Image.fromarray(apply_shade(w, r, 0.55)).resize((640, 360)),
                           "%s | flat strength 0.55" % clipname))
        # (b) adaptive: exponent raised where local luma is low
        smap = (0.55 * (1.0 + 2.2 * (1.0 - Lw) ** 2)).astype(np.float32)
        sh = np.power(r, smap)
        adp = np.clip(w.astype(np.float32) * sh[..., None], 0, 255).astype(np.uint8)
        ims.append(caption(Image.fromarray(adp).resize((640, 360)),
                           "%s | ADAPTIVE strength 0.55..2.3 by local luma" % clipname))
        # (c) additive fallback, clearly a different thing
        add = np.clip(w.astype(np.float32) * np.power(r, 0.55)[..., None]
                      + 70.0 * (np.power(r, 1.2) - 1.0)[..., None]
                      * ((1.0 - Lw) ** 2)[..., None], 0, 255).astype(np.uint8)
        ims.append(caption(Image.fromarray(add).resize((640, 360)),
                           "%s | ADDITIVE light in the blacks (NOT multiplicative)"
                           % clipname))
    grid_sheet(ims, 3, os.path.join(OUT, "adaptive-strength.png"))
    P("wrote adaptive-strength.png")

    # short cycle on digits with adaptive strength
    w = ffmpeg_writer(os.path.join(OUT, "adaptive-digits.mp4"))
    k = 0
    for frame in ffmpeg_frames(os.path.join(root,
                                            "public/media/showcase-test/digits.mp4")):
        t = k / FPS
        a = atten2(t, *RAMP_SHIP)
        D = node_disp(B, t, a, WOB)
        sx, sy, _ = inverse_map(full_field(B, D), 6)
        wf = cv2.remap(frame, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        Lw = luma(wf) / 255.0
        smap = (strength_sched("overshoot", t) *
                (1.0 + 2.2 * (1.0 - Lw) ** 2)).astype(np.float32)
        sh = np.power(shade_base(sx, sy), smap)
        out = np.clip(wf.astype(np.float32) * sh[..., None], 0, 255).astype(np.uint8)
        w.stdin.write(np.ascontiguousarray(out).tobytes())
        k += 1
    w.stdin.close(); w.wait()
    P("wrote adaptive-digits.mp4 (%d frames)" % k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    st = a.stage
    if st in ("all", "wireframe"): stage_wireframe()
    if st in ("all", "density"):   stage_density()
    if st in ("all", "words"):     stage_words()
    if st in ("all", "wobble"):    stage_wobble()
    if st in ("all", "loop"):      stage_loop()
    if st in ("all", "tps"):       stage_tps()
    if st in ("all", "contact"):   stage_contact()
    if st in ("all", "warp", "warpgain"):  stage_warpgain()
    Bw = stage_warpcycle() if st in ("all", "warp", "warpcycle") else None
    if st in ("all", "warp", "warpvideo"): stage_warpvideo(Bw)
    if st in ("all", "jacobian"):  stage_jacobian()
    if st in ("all", "strength"):  stage_strength()
    if st in ("all", "release"):   stage_release()
    if st in ("all", "adaptive"):  stage_adaptive()
    if st in ("all", "ship"):      stage_ship()
    if st == "curves":             stage_curves()
    open(os.path.join(OUT, "run.log"), "a").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
