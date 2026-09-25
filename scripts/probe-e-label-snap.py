#!/usr/bin/env python3
"""
probe-e-label-snap.py  --  Agent E probe: "label and snap".

One single fine quad lattice.  Nothing is drawn on top of it.  The word is made
readable purely by how the lattice edges arrange themselves:

  1. wobble a regular quad lattice (time periodic, 12 s loop)
  2. label every CELL inside/outside by sampling the word bitmap at its centroid
  3. the inside/outside frontier is then automatically a closed chain of
     ordinary lattice edges -- but a staircase one
  4. project the chain vertices onto the true glyph outline with a cKDTree,
     alternating tangential smoothing and re-projection so they spread evenly
  5. Laplacian-relax the remaining (free) vertices, chain + image border pinned,
     so the snap distortion bleeds several cells out into the surrounding field
  6. repair folds: any cell that inverts has its free vertices blended back
     toward their wobbled base positions until every cell is positively oriented

Deliverables are written under the agent-e scratchpad.  Self contained; does not
import or modify any other probe.

Usage:
    probe-e-label-snap.py single      # one frame + wireframe + composite
    probe-e-label-snap.py density     # density sweep -> wireframe-sheet.png
    probe-e-label-snap.py matrix      # full matrix -> contact-sheet.png
    probe-e-label-snap.py motion      # 12 s loop pop / stability check
    probe-e-label-snap.py all
"""

import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

W, H = 1280, 720
FPS = 30
LOOP = 12.0

OUT = os.environ.get(
    "AGENT_E_OUT",
    "/private/tmp/claude-501/-Users-me-Desktop-dev-yugumishra-github-io/"
    "ea9e4c74-aeb0-4767-b13b-b99660eec2a7/scratchpad/agent-e",
)
FRAMES = os.path.join(OUT, "frames")
WORK = os.path.join(OUT, "work")

FONT_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FONT_UI = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_UI_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

WORDS = {
    "yope3d": "Yope3D",
    "spinstack": "SpinStack",
    "gravity": "3D Gravity Simulator",
}

WOBBLE_K = float(os.environ.get("AGENT_E_WK", "0.30"))

INK = (26, 32, 44)          # single wireframe colour
PAPER = (250, 250, 248)

os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


# --------------------------------------------------------------------------
# glyph bitmap
# --------------------------------------------------------------------------

def glyph_assets(word, target_h=213, margin=64, cache={}):
    """Binary word bitmap + KD-tree over its outline pixels."""
    key = (word, target_h, margin)
    if key in cache:
        return cache[key]

    # find the font size whose ink bbox is target_h tall, then shrink to fit W
    size = target_h
    for _ in range(40):
        f = ImageFont.truetype(FONT_BLACK, size)
        bb = f.getbbox(word)
        h = bb[3] - bb[1]
        if h == 0:
            break
        size = int(round(size * target_h / h))
        f = ImageFont.truetype(FONT_BLACK, size)
        bb = f.getbbox(word)
        if abs((bb[3] - bb[1]) - target_h) <= 1:
            break
    f = ImageFont.truetype(FONT_BLACK, size)
    bb = f.getbbox(word)
    w_ink = bb[2] - bb[0]
    if w_ink > W - 2 * margin:                      # long words: fit to width
        size = int(size * (W - 2 * margin) / w_ink)
        f = ImageFont.truetype(FONT_BLACK, size)
        bb = f.getbbox(word)
        w_ink, target_h = bb[2] - bb[0], bb[3] - bb[1]

    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    d.text(((W - (bb[2] - bb[0])) / 2 - bb[0],
            (H - (bb[3] - bb[1])) / 2 - bb[1]), word, font=f, fill=255)
    inside = np.asarray(img) > 127

    # outline = 1px ring on the inside of the shape
    ring = inside & ~ndimage.binary_erosion(inside)
    ys, xs = np.nonzero(ring)
    pts = np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float64)
    tree = cKDTree(pts)

    # stroke width: 2x the max distance-to-outside inside the shape,
    # measured on the thinnest limb (median of the skeleton ridge)
    dt = ndimage.distance_transform_edt(inside)
    ridge = dt[dt > 0]
    stroke = 2.0 * np.percentile(ridge, 90) if ridge.size else 0.0

    # signed distance, positive outside.  Labelling may use an inflated or
    # eroded threshold on this while the SNAP TARGET stays the true outline;
    # inflating hands the interior more cells than its area warrants, so the
    # interior mesh compresses and the surrounding field is stretched outward.
    sdf = (ndimage.distance_transform_edt(~inside)
           - ndimage.distance_transform_edt(inside))

    out = dict(word=word, inside=inside, sdf=sdf, pts=pts, tree=tree, size=size,
               ink_h=(bb[3] - bb[1]), ink_w=(bb[2] - bb[0]), stroke=stroke)
    cache[key] = out
    return out


# --------------------------------------------------------------------------
# lattice
# --------------------------------------------------------------------------

def base_lattice(nx, ny, t, amp):
    """Wobbled base lattice, (ny+1, nx+1, 2).  Periodic with LOOP seconds."""
    cw, ch = W / nx, H / ny
    i = np.arange(nx + 1)[None, :]
    j = np.arange(ny + 1)[:, None]
    x = i * cw
    y = j * ch
    V = np.zeros((ny + 1, nx + 1, 2))
    V[..., 0] = x
    V[..., 1] = y
    if amp > 0:
        ph = 2 * math.pi * t / LOOP
        a = amp * min(cw, ch)
        # three incommensurate-in-space but loop-periodic-in-time components
        k = WOBBLE_K   # spatial frequency of the wobble, in radians per cell
        V[..., 0] += a * (0.62 * np.sin(k * (1.7 * i + 2.3 * j) + 1 * ph)
                          + 0.38 * np.sin(k * (0.9 * i - 3.1 * j) + 2 * ph + 1.1))
        V[..., 1] += a * (0.62 * np.cos(k * (2.1 * i - 1.3 * j) + 1 * ph + 0.4)
                          + 0.38 * np.cos(k * (3.3 * i + 0.7 * j) - 2 * ph))
    # keep the frame filled: border vertices slide along the border only
    V[0, :, 1] = 0.0
    V[-1, :, 1] = H
    V[:, 0, 0] = 0.0
    V[:, -1, 0] = W
    V[..., 0] = np.clip(V[..., 0], 0, W)
    V[..., 1] = np.clip(V[..., 1], 0, H)
    return V


def cell_centroids(V):
    return 0.25 * (V[:-1, :-1] + V[:-1, 1:] + V[1:, 1:] + V[1:, :-1])


def cell_coverage(V, inside, n=3):
    """Fraction of n*n bilinear samples of each cell that land inside."""
    us = (np.arange(n) + 0.5) / n
    p00, p01 = V[:-1, :-1], V[:-1, 1:]
    p10, p11 = V[1:, :-1], V[1:, 1:]
    acc = np.zeros((V.shape[0] - 1, V.shape[1] - 1))
    for u in us:
        top = p00 + (p01 - p00) * u
        bot = p10 + (p11 - p10) * u
        for v in us:
            p = top + (bot - top) * v
            xi = np.clip(p[..., 0].astype(int), 0, W - 1)
            yi = np.clip(p[..., 1].astype(int), 0, H - 1)
            acc += inside[yi, xi]
    return acc / (n * n)


def label_cells(V, inside):
    """Inside/outside label per cell, with checkerboard vertices resolved.

    `inside` is the LABEL bitmap, which may be an inflated/eroded version of
    the glyph; the snap target is always the true outline.
    """
    c = cell_centroids(V)
    xi = np.clip(c[..., 0].astype(int), 0, W - 1)
    yi = np.clip(c[..., 1].astype(int), 0, H - 1)
    L = inside[yi, xi].copy()
    cov = cell_coverage(V, inside)

    # A vertex touched by a diagonal (in,out / out,in) 2x2 block would carry
    # four boundary edges and the chain would not be a manifold curve.  Flip
    # whichever of the four cells is least committed until none remain.
    for _ in range(40):
        a, b = L[:-1, :-1], L[:-1, 1:]
        cc, d = L[1:, :-1], L[1:, 1:]
        amb = (a == d) & (b == cc) & (a != b)
        if not amb.any():
            break
        jj, ii = np.nonzero(amb)
        # least-committed cell of each ambiguous block
        blocks = np.stack([cov[jj, ii], cov[jj, ii + 1],
                           cov[jj + 1, ii], cov[jj + 1, ii + 1]], 1)
        k = np.argmin(np.abs(blocks - 0.5), 1)
        tj = jj + (k >= 2)
        ti = ii + (k % 2 == 1)
        L[tj, ti] = ~L[tj, ti]
    return L


def boundary_chain(L, nx, ny):
    """Edges of the lattice that separate an inside cell from an outside cell.

    Returns (edges as vertex-index pairs, adjacency list, chain vertex ids).
    Vertex (j,i) has flat index j*(nx+1)+i.
    """
    def vid(j, i):
        return j * (nx + 1) + i

    edges = []
    # vertical lattice edges: between cell (j,i-1) and (j,i)
    dj, di = np.nonzero(L[:, 1:] != L[:, :-1])
    for j, i in zip(dj, di + 1):
        edges.append((vid(j, i), vid(j + 1, i)))
    # horizontal lattice edges: between cell (j-1,i) and (j,i)
    dj, di = np.nonzero(L[1:, :] != L[:-1, :])
    for j, i in zip(dj + 1, di):
        edges.append((vid(j, i), vid(j, i + 1)))
    # a glyph is never allowed to touch the frame, so no border edges needed

    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return edges, adj, np.array(sorted(adj.keys()), dtype=np.int64)


# --------------------------------------------------------------------------
# snap + relax
# --------------------------------------------------------------------------

def snap_and_relax(V, L, nx, ny, ga, rounds=6, relax_iters=160, omega=1.85,
                   snap_gain=1.0, cap_cells=0.85):
    """Return deformed vertices, plus stats.  V is the wobbled base lattice."""
    nv = (nx + 1) * (ny + 1)
    edges, adj, chain = boundary_chain(L, nx, ny)
    flat = V.reshape(nv, 2).copy()

    stats = dict(n_chain=int(chain.size), n_edges=len(edges))
    if chain.size == 0:
        return V.copy(), stats, chain, adj

    tree = ga["tree"]
    base_chain = flat[chain].copy()
    cap = cap_cells * min(W / nx, H / ny)
    P, _ = _project(base_chain, tree)
    P = _cap(base_chain, P, cap)
    idx_of = {int(v): k for k, v in enumerate(chain)}
    nbr = [[idx_of[n] for n in adj[int(v)] if n in idx_of] for v in chain]

    # alternate tangential smoothing with re-projection so the vertices spread
    # along the outline instead of clumping at the nearest staircase corner.
    # The cap stops a vertex snapping clear across a thin stroke, which is what
    # produced most of the folds.
    for _ in range(rounds):
        S = P.copy()
        for k, ns in enumerate(nbr):
            if ns:
                S[k] = 0.5 * P[k] + 0.5 * P[ns].mean(0)
        P, _ = _project(S, tree)
        P = _cap(base_chain, P, cap)
    _, resid = _project(P, tree)
    stats["snap_resid_px"] = float(np.mean(resid))

    # displacement field, harmonic away from the pinned chain
    d = np.zeros((ny + 1, nx + 1, 2))
    fixed = np.zeros((ny + 1, nx + 1), bool)
    cj, ci = chain // (nx + 1), chain % (nx + 1)
    d[cj, ci] = (P - flat[chain]) * snap_gain
    fixed[cj, ci] = True
    fixed[0, :] = fixed[-1, :] = True
    fixed[:, 0] = fixed[:, -1] = True
    stats["snap_mean_px"] = float(np.linalg.norm(d[cj, ci], axis=1).mean())
    stats["snap_max_px"] = float(np.linalg.norm(d[cj, ci], axis=1).max())

    cnt = np.zeros((ny + 1, nx + 1))
    cnt[1:, :] += 1
    cnt[:-1, :] += 1
    cnt[:, 1:] += 1
    cnt[:, :-1] += 1
    free = ~fixed
    jj, ii = np.mgrid[0:ny + 1, 0:nx + 1]
    parity = (jj + ii) % 2
    masks = [free & (parity == 0), free & (parity == 1)]
    # red/black Gauss-Seidel with over-relaxation (plain Jacobi diverges for
    # omega > 1, red/black does not)
    for _ in range(relax_iters):
        for m in masks:
            acc = np.zeros_like(d)
            acc[1:, :] += d[:-1, :]
            acc[:-1, :] += d[1:, :]
            acc[:, 1:] += d[:, :-1]
            acc[:, :-1] += d[:, 1:]
            avg = acc / cnt[..., None]
            d[m] += omega * (avg[m] - d[m])

    # fold repair.  Stage 1 blends only FREE vertices back toward their wobbled
    # base positions.  Cells that are still inverted after that are being
    # folded by the snapped chain itself (two chain vertices that projected
    # past each other), so stage 2 lets those chain vertices give way too --
    # they come off the outline, which is reported as `chain_yield`.
    alpha = np.ones((ny + 1, nx + 1))
    on_chain = np.zeros((ny + 1, nx + 1), bool)
    on_chain[cj, ci] = True
    border = np.zeros((ny + 1, nx + 1), bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    bad0 = fold_s1 = None
    for it in range(56):
        Vd = V + alpha[..., None] * d
        bad = _inverted(Vd)
        if bad0 is None:
            bad0 = int(bad.sum())
        if it == 28:
            fold_s1 = int(bad.sum())
        if not bad.any():
            break
        movable = free if it < 28 else (~border)
        bj, bi = np.nonzero(bad)
        for oj in (0, 1):
            for oi in (0, 1):
                m = movable[bj + oj, bi + oi]
                alpha[bj[m] + oj, bi[m] + oi] *= 0.62
    Vd = V + alpha[..., None] * d
    bad = _inverted(Vd)
    ncells = nx * ny
    stats["fold_raw"] = bad0 / ncells
    stats["fold_after_free_only"] = (fold_s1 if fold_s1 is not None else 0) / ncells
    stats["fold_after"] = int(bad.sum()) / ncells
    yielded = on_chain & (alpha < 0.999)
    stats["chain_yield"] = int(yielded.sum()) / max(1, int(chain.size))
    off = np.linalg.norm(Vd[cj, ci] - P, axis=1)
    stats["contour_err_px"] = float(off.mean())
    stats["contour_err_max_px"] = float(off.max())
    # how far the letter-induced distortion reaches: mean |displacement|
    # binned by graph distance (in cells) from the chain
    dist = np.full((ny + 1, nx + 1), 1e9)
    dist[cj, ci] = 0
    for _ in range(14):
        nd = dist.copy()
        nd[1:, :] = np.minimum(nd[1:, :], dist[:-1, :] + 1)
        nd[:-1, :] = np.minimum(nd[:-1, :], dist[1:, :] + 1)
        nd[:, 1:] = np.minimum(nd[:, 1:], dist[:, :-1] + 1)
        nd[:, :-1] = np.minimum(nd[:, :-1], dist[:, 1:] + 1)
        dist = nd
    mag = np.linalg.norm(alpha[..., None] * d, axis=-1)
    stats["spread_px"] = [round(float(mag[dist == k].mean()), 2)
                          if (dist == k).any() else None for k in range(0, 13)]
    return Vd, stats, chain, adj


def _cap(base, P, cap):
    d = P - base
    n = np.linalg.norm(d, axis=1, keepdims=True)
    f = np.minimum(1.0, cap / np.maximum(n, 1e-9))
    return base + d * f


def _project(P, tree):
    dd, ii = tree.query(P, k=1)
    return tree.data[ii].copy(), dd


def _tri(a, b, c):
    return ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))


def _inverted(V):
    """Per-cell fold test.

    A quad is sound when it is simple and positively oriented, which holds if
    EITHER diagonal splits it into two positively oriented triangles.  A merely
    non-convex quad is fine and must not be counted as a fold; a bow-tie or a
    negatively oriented quad fails both splits.
    """
    p0, p1 = V[:-1, :-1], V[:-1, 1:]
    p2, p3 = V[1:, 1:], V[1:, :-1]
    dA = (_tri(p0, p1, p2) > 0) & (_tri(p0, p2, p3) > 0)
    dB = (_tri(p1, p2, p3) > 0) & (_tri(p1, p3, p0) > 0)
    return ~(dA | dB)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_wireframe(V, ss=2, lw=2, ink=INK, paper=PAPER):
    """THE ACCEPTANCE TEST IMAGE: every lattice edge, one colour, nothing else."""
    im = Image.new("RGB", (W * ss, H * ss), paper)
    d = ImageDraw.Draw(im)
    A = V * ss
    for j in range(A.shape[0]):                       # rows
        d.line([tuple(p) for p in A[j]], fill=ink, width=lw, joint="curve")
    for i in range(A.shape[1]):                       # columns
        d.line([tuple(p) for p in A[:, i]], fill=ink, width=lw, joint="curve")
    return im.resize((W, H), Image.LANCZOS)


def id_field(L, nx, ny, n_in, n_out, seed=7):
    """Clip id per cell.  Inside and outside families are disjoint."""
    rng = np.random.default_rng(seed)
    ids = np.where(L, rng.integers(0, n_in, L.shape),
                   n_in + rng.integers(0, n_out, L.shape))
    return ids.astype(np.int32)


def rasterise_ids(V, ids, nx, ny):
    im = Image.new("I", (W, H), 0)
    d = ImageDraw.Draw(im)
    for j in range(ny):
        for i in range(nx):
            poly = [tuple(V[j, i]), tuple(V[j, i + 1]),
                    tuple(V[j + 1, i + 1]), tuple(V[j + 1, i])]
            d.polygon(poly, fill=int(ids[j, i]) + 1)
    return np.asarray(im).astype(np.int32) - 1


_CLIPS = {}


def clip_stack(copies=3):
    """[inside stills...] + [outside stills...], luminance separated."""
    key = copies
    if key in _CLIPS:
        return _CLIPS[key]
    # measured: spinstack(1) is the only broadly bright clip, digits(2) has
    # bright strokes on black; yope cloth(0) is dark and gravity(3) is black.
    bright, dark = [1, 2], [0, 3]
    ins, outs = [], []
    for c in bright:
        for v in range(copies):
            ins.append(_load(c, v % 3))
    for c in dark:
        for v in range(copies):
            outs.append(_load(c, v % 3))
    _CLIPS[key] = (ins, outs)
    return _CLIPS[key]


def _load(c, v):
    p = os.path.join(FRAMES, "clip%d_v%d.png" % (c, v))
    if not os.path.exists(p):
        p = os.path.join(FRAMES, "clip%d.png" % c)
    a = np.asarray(Image.open(p).convert("RGB").resize((W, H)))
    return a.astype(np.uint8)


def composite(V, L, nx, ny, copies=3, seed=7):
    ins, outs = clip_stack(copies)
    ids = id_field(L, nx, ny, len(ins), len(outs), seed)
    field = rasterise_ids(V, ids, nx, ny)
    stack = ins + outs
    out = np.zeros((H, W, 3), np.uint8)
    for k, src in enumerate(stack):
        m = field == k
        if m.any():
            # small per-id parallax so two cells of the same source differ
            sx, sy = (k * 137) % 260 - 130, (k * 83) % 180 - 90
            out[m] = np.roll(np.roll(src, sy, 0), sx, 1)[m]
    return Image.fromarray(out), field


def squint(img, inside, sigma):
    a = np.asarray(img.convert("L")).astype(np.float64) / 255.0
    a = ndimage.gaussian_filter(a, sigma)
    b = ndimage.gaussian_filter(inside.astype(np.float64), sigma)
    a -= a.mean()
    b -= b.mean()
    den = a.std() * b.std()
    return float((a * b).mean() / den) if den > 1e-9 else 0.0


# --------------------------------------------------------------------------
# one frame
# --------------------------------------------------------------------------

def build(word_key, nx, amp=0.28, t=0.0, rounds=None, relax=None,
          snap_gain=1.0, target_h=213, inflate=None):
    rounds = int(os.environ.get("AGENT_E_ROUNDS", 8)) if rounds is None else rounds
    relax = int(os.environ.get("AGENT_E_RELAX", 70)) if relax is None else relax
    if inflate is None:
        inflate = float(os.environ.get("AGENT_E_INFLATE", 0.0))
    ny = max(4, int(round(nx * H / W)))
    ga = glyph_assets(WORDS[word_key], target_h=target_h)
    t0 = time.perf_counter()
    V = base_lattice(nx, ny, t, amp)
    label_bmp = ga["sdf"] <= inflate * (W / nx)
    L = label_cells(V, label_bmp)
    Vd, st, chain, adj = snap_and_relax(
        V, L, nx, ny, ga, rounds=rounds, relax_iters=relax,
        snap_gain=snap_gain,
        cap_cells=float(os.environ.get("AGENT_E_CAP", 1.2)))
    st["ms"] = (time.perf_counter() - t0) * 1000.0
    # topology audit: how many of the glyph's own parts and counters survived
    # the centroid labelling
    lab_in, n_in = ndimage.label(L)
    lab_out, n_out = ndimage.label(~L)
    bg = lab_out[0, 0]
    st["parts_kept"] = int(n_in)
    st["counters_kept"] = int(n_out - (1 if bg else 0))
    tl_in, tn_in = ndimage.label(ga["inside"])
    tl_out, tn_out = ndimage.label(~ga["inside"])
    st["parts_true"] = int(tn_in)
    st["counters_true"] = int(tn_out - 1)
    st.update(word=WORDS[word_key], nx=nx, ny=ny, amp=amp, inflate=inflate,
              cell_px=round(W / nx, 1), stroke_px=round(ga["stroke"], 1),
              cells_per_stroke=round(ga["stroke"] / (W / nx), 2),
              inside_cells=int(L.sum()))
    return dict(V=V, Vd=Vd, L=L, st=st, ga=ga, nx=nx, ny=ny, chain=chain)


# --------------------------------------------------------------------------
# sheets
# --------------------------------------------------------------------------

def caption(tiles, cols, cell_w, pad=10, head=30, title=None, native=False):
    rows = (len(tiles) + cols - 1) // cols
    th = int(cell_w * H / W)
    ww = cols * (cell_w + pad) + pad
    hh = rows * (th + head + pad) + pad + (46 if title else 0)
    sheet = Image.new("RGB", (ww, hh), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT_UI, 15)
    fb = ImageFont.truetype(FONT_UI_B, 26)
    y0 = 0
    if title:
        d.text((pad, 12), title, font=fb, fill=(0, 0, 0))
        y0 = 46
    for k, (img, cap) in enumerate(tiles):
        r, c = divmod(k, cols)
        x = pad + c * (cell_w + pad)
        y = y0 + pad + r * (th + head + pad)
        sheet.paste(img if native else img.resize((cell_w, th), Image.LANCZOS),
                    (x, y))
        d.rectangle([x, y, x + cell_w - 1, y + th - 1], outline=(190, 190, 190))
        for li, line in enumerate(cap.split("\n")[:2]):
            d.text((x, y + th + 3 + li * 14), line, font=f, fill=(20, 20, 20))
    return sheet


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_single(argv):
    word = argv[0] if argv else "yope3d"
    nx = int(argv[1]) if len(argv) > 1 else 64
    amp = float(argv[2]) if len(argv) > 2 else 0.28
    r = build(word, nx, amp=amp)
    wf = render_wireframe(r["Vd"])
    wf.save(os.path.join(OUT, "wireframe.png"))
    render_wireframe(r["V"]).save(os.path.join(WORK, "wireframe-undeformed.png"))
    comp, field = composite(r["Vd"], r["L"], r["nx"], r["ny"])
    comp.save(os.path.join(OUT, "composite.png"))
    r["st"]["squint"] = squint(comp, r["ga"]["inside"], r["ga"]["stroke"] / 2)
    r["st"]["wire_squint"] = squint(wf, r["ga"]["inside"], r["ga"]["stroke"] / 2)
    print(json.dumps(r["st"], indent=1))
    return r


def cmd_density(argv):
    word = argv[0] if argv else "yope3d"
    amp = float(argv[1]) if len(argv) > 1 else 0.28
    tiles, rows = [], []
    for nx in (24, 32, 40, 48, 64, 80, 96, 128):
        r = build(word, nx, amp=amp)
        img = render_wireframe(r["Vd"])
        img.save(os.path.join(WORK, "wire-%s-%d-a%02d.png"
                              % (word, nx, int(amp * 100))))
        st = r["st"]
        tiles.append((img, "nx=%d   cell=%.0f px   %.2f cells per stroke   "
                           "chain=%d edges   folds after repair %.2f%%   "
                           "letter parts %d/%d   counters %d/%d   %.0f ms"
                      % (nx, st["cell_px"], st["cells_per_stroke"],
                         st["n_chain"], 100 * st["fold_after"],
                         st["parts_kept"], st["parts_true"],
                         st["counters_kept"], st["counters_true"], st["ms"])))
        rows.append(st)
        print("nx=%-4d %s" % (nx, json.dumps(st)))
    sheet = caption(tiles, 2, W, head=34, native=True,
                    title="Agent E - label+snap - ACCEPTANCE WIREFRAMES - "
                          "'%s' - wobble %.2f - every lattice edge, one colour,"
                          " one line weight, nothing overlaid"
                          % (WORDS[word], amp))
    sheet.save(os.path.join(OUT, "wireframe-sheet.png"))
    json.dump(rows, open(os.path.join(WORK, "density-%s.json" % word), "w"),
              indent=1)
    return rows


def cmd_matrix(argv):
    tiles, rows = [], []
    combos = []
    for word in ("yope3d", "spinstack", "gravity"):
        for nx in (32, 64, 96):
            for amp in (0.0, 0.28):
                combos.append((word, nx, amp))
    for word, nx, amp in combos:
        r = build(word, nx, amp=amp)
        comp, _ = composite(r["Vd"], r["L"], r["nx"], r["ny"])
        st = r["st"]
        st["squint"] = squint(comp, r["ga"]["inside"], r["ga"]["stroke"] / 2)
        comp.save(os.path.join(WORK, "comp-%s-%d-a%02d.png"
                               % (word, nx, int(amp * 100))))
        tiles.append((comp, "%s  nx=%d cell=%.0fpx wobble=%.2f\n"
                            "squint=%.2f folds=%.2f%% %.0fms"
                      % (WORDS[word], nx, st["cell_px"], amp, st["squint"],
                         100 * st["fold_after"], st["ms"])))
        rows.append(st)
        print(json.dumps(st))
    sheet = caption(tiles, 3, 420,
                    title="Agent E - label+snap - composites, inside family = "
                          "bright clips (spinstack/digits), outside = dark "
                          "(yope cloth/gravity)")
    sheet.save(os.path.join(OUT, "contact-sheet.png"))
    json.dump(rows, open(os.path.join(WORK, "matrix.json"), "w"), indent=1)
    return rows


def cmd_motion(argv):
    word = argv[0] if argv else "yope3d"
    nx = int(argv[1]) if len(argv) > 1 else 64
    amp = float(argv[2]) if len(argv) > 2 else 0.28
    n = int(argv[3]) if len(argv) > 3 else 36
    prevV = prevL = None
    firstV = firstL = None
    rows = []
    tiles = []
    for k in range(n):
        t = LOOP * k / n
        r = build(word, nx, amp=amp, t=t)
        V, L = r["Vd"], r["L"]
        row = dict(t=round(t, 3), fold=r["st"]["fold_after"],
                   chain=r["st"]["n_chain"], inside=int(L.sum()),
                   ms=round(r["st"]["ms"], 1))
        if prevV is not None:
            row["max_vertex_jump_px"] = float(
                np.abs(V - prevV).sum(-1).max())
            row["label_flips"] = int((L != prevL).sum())
        rows.append(row)
        prevV, prevL = V, L
        if firstV is None:
            firstV, firstL = V.copy(), L.copy()
        if k % (max(1, n // 6)) == 0 and len(tiles) < 6:
            tiles.append((render_wireframe(V), "t=%.2fs  folds=%.2f%%  "
                                               "chain=%d"
                          % (t, 100 * r["st"]["fold_after"], r["st"]["n_chain"])))
        print(json.dumps(row))
    # loop closure: frame n == frame 0
    r0 = build(word, nx, amp=amp, t=LOOP)
    close = float(np.abs(r0["Vd"] - firstV).sum(-1).max())
    print("loop closure max vertex delta px = %.4f" % close)
    sheet = caption(tiles, 3, 420,
                    title="Agent E - motion across the 12 s loop - '%s' nx=%d "
                          "wobble=%.2f (loop closure %.3f px)"
                          % (WORDS[word], nx, amp, close))
    sheet.save(os.path.join(OUT, "motion-sheet.png"))
    json.dump(dict(rows=rows, loop_closure_px=close),
              open(os.path.join(WORK, "motion.json"), "w"), indent=1)
    return rows


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "single"
    argv = sys.argv[2:]
    if cmd == "single":
        cmd_single(argv)
    elif cmd == "density":
        cmd_density(argv)
    elif cmd == "matrix":
        cmd_matrix(argv)
    elif cmd == "motion":
        cmd_motion(argv)
    elif cmd == "all":
        cmd_single([])
        cmd_density([])
        cmd_matrix([])
        cmd_motion([])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
