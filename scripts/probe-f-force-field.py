#!/usr/bin/env python3
"""
probe-f-force-field.py  --  Agent F.

Force-directed settling of ONE fine quad lattice onto a glyph signed-distance
field.  No labelling, no projection, no routing, nothing drawn on top.

Every vertex of a plain moving quad lattice feels:
  * an attraction toward the nearest zero crossing of the glyph SDF, weighted
    by a Gaussian falloff so only vertices already near the outline are
    captured;
  * axial springs along the lattice edges (rest length = undeformed spacing);
  * diagonal (shape) springs per quad;
  * a signed-area term with a barrier that resists collapse / inversion;
  * a weak "home" spring toward a time-varying wobble field.

Iterated to equilibrium with the attraction strength ramped 0 -> 1, which is
the formation animation.  The word appears because chains of ordinary lattice
edges settle onto the contour and the cells straddling it are squeezed thin,
while the surrounding field stretches to make room.

Self contained.  Writes into the agent-f scratchpad directory.
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# ----------------------------------------------------------------------------
# paths / constants
# ----------------------------------------------------------------------------

OUT = ("/private/tmp/claude-501/-Users-me-Desktop-dev-yugumishra-github-io/"
       "ea9e4c74-aeb0-4767-b13b-b99660eec2a7/scratchpad/agent-f")
SCRATCH = os.path.dirname(OUT)
REPO = "/Users/me/Desktop/dev/yugumishra.github.io"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Black.ttf"

W, H = 1280, 720
FPS = 30
LOOP_SEC = 12.0

CLIP_SOURCES = [
    os.path.join(REPO, "public/media/yope_cloth_b.mp4"),
    os.path.join(REPO, "public/media/showcase-test/spinstack.mp4"),
    os.path.join(REPO, "public/media/showcase-test/digits.mp4"),
    os.path.join(REPO, "public/media/showcase-test/gravity.mp4"),
]

os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# 1.  signed distance field for a word
# ----------------------------------------------------------------------------

PAD = 260  # SDF padding beyond canvas, so the overscanned lattice can sample it


def _ink_bbox(word, size):
    font = ImageFont.truetype(FONT_PATH, size)
    bb = font.getbbox(word)
    pad = 40
    im = Image.new("L", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), word, fill=255,
                            font=font)
    a = np.asarray(im) > 127
    ys, xs = np.where(a)
    return font, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), im, pad


def render_word_mask(word, cap_px=213.0, max_w_frac=0.86):
    """Binary glyph mask on a padded canvas.  Returns (mask, info).

    The size is chosen so the *inked* height of the word is cap_px, then
    shrunk if the word is too wide for the canvas."""
    lo, hi = 8, 900
    for _ in range(30):
        mid = int(round((lo + hi) / 2.0))
        _, bb, _, _ = _ink_bbox(word, mid)
        if (bb[3] - bb[1]) < cap_px:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1:
            break
    size = max(8, int(round((lo + hi) / 2.0)))
    font, bb, _, _ = _ink_bbox(word, size)
    if (bb[2] - bb[0]) > W * max_w_frac:
        size = max(8, int(size * (W * max_w_frac) / (bb[2] - bb[0])))
        font, bb, _, _ = _ink_bbox(word, size)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]

    img = Image.new("L", (W + 2 * PAD, H + 2 * PAD), 0)
    d = ImageDraw.Draw(img)
    fbb = font.getbbox(word)
    _, ibb, _, ipad = _ink_bbox(word, size)
    # ink-bbox origin relative to the text anchor
    ox = ibb[0] - (ipad - fbb[0])
    oy = ibb[1] - (ipad - fbb[1])
    x = PAD + (W - tw) / 2.0 - ox
    y = PAD + (H - th) / 2.0 - oy
    d.text((x, y), word, fill=255, font=font)
    mask = np.asarray(img) > 127

    # measure the real stroke width: 2 * max interior distance of the mask
    din = ndimage.distance_transform_edt(mask)
    stroke = 2.0 * np.percentile(din[mask], 97) if mask.any() else 0.0
    info = dict(font_size=size, text_w=int(tw), cap_h=int(th),
                stroke_px=round(float(stroke), 1))
    return mask, info


def build_sdf(mask):
    """Positive outside, negative inside, plus a unit gradient field."""
    dout = ndimage.distance_transform_edt(~mask)
    din = ndimage.distance_transform_edt(mask)
    sdf = (dout - din).astype(np.float32)
    # smooth a touch so the gradient is not staircased at the raster level
    sdf = ndimage.gaussian_filter(sdf, 1.2).astype(np.float32)
    gy, gx = np.gradient(sdf)
    n = np.sqrt(gx * gx + gy * gy)
    # |grad| of a true distance field is 1; it collapses on medial axes, where
    # the direction to the nearest zero crossing is ambiguous.  Use it as a
    # coherence weight so vertices there are not yanked around by noise.
    coh = np.clip(n, 0.0, 1.0).astype(np.float32)
    coh = ndimage.gaussian_filter(coh, 2.0).astype(np.float32)
    n = n + 1e-6
    ux, uy = gx / n, gy / n
    # divergence of the unit gradient = local curvature of the level set.  It
    # blows up in the fan outside a sharp corner, where a whole wedge of
    # vertices all project onto the same point.
    kxx = np.gradient(ux, axis=1)
    kyy = np.gradient(uy, axis=0)
    curv = np.abs(kxx + kyy).astype(np.float32)
    curv = ndimage.gaussian_filter(curv, 2.0).astype(np.float32)
    return (sdf, ux.astype(np.float32), uy.astype(np.float32), coh, curv)


def sample(field, x, y):
    """Bilinear sample of a padded field at canvas coords (x, y)."""
    fx = np.clip(x + PAD, 0, field.shape[1] - 1.001)
    fy = np.clip(y + PAD, 0, field.shape[0] - 1.001)
    x0 = fx.astype(np.int32)
    y0 = fy.astype(np.int32)
    tx = fx - x0
    ty = fy - y0
    f00 = field[y0, x0]
    f10 = field[y0, x0 + 1]
    f01 = field[y0 + 1, x0]
    f11 = field[y0 + 1, x0 + 1]
    return (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty) +
            f01 * (1 - tx) * ty + f11 * tx * ty)


# ----------------------------------------------------------------------------
# 2.  the lattice
# ----------------------------------------------------------------------------

class Lattice:
    """A quad lattice over an overscanned rectangle."""

    def __init__(self, nx_vis, overscan_cells=3.0, aspect_lock=True):
        self.h = W / float(nx_vis)                      # target cell size
        hx = self.h
        hy = self.h if aspect_lock else H / round(H / self.h)
        ov_x = overscan_cells * hx
        ov_y = overscan_cells * hy
        self.nx = int(round((W + 2 * ov_x) / hx))
        self.ny = int(round((H + 2 * ov_y) / hy))
        self.x0 = W / 2.0 - self.nx * hx / 2.0
        self.y0 = H / 2.0 - self.ny * hy / 2.0
        self.hx = hx
        self.hy = hy
        self.nx_vis = nx_vis
        self.ny_vis = int(round(H / hy))

        xs = self.x0 + np.arange(self.nx + 1) * hx
        ys = self.y0 + np.arange(self.ny + 1) * hy
        gx, gy = np.meshgrid(xs, ys)
        self.R = np.stack([gx, gy], axis=-1).astype(np.float64)   # rest grid
        self.P = self.R.copy()                                    # positions
        self.rect = (self.x0, self.y0,
                     self.x0 + self.nx * hx, self.y0 + self.ny * hy)

    @property
    def n_quads(self):
        return self.nx * self.ny

    def quad_corners(self, P=None):
        P = self.P if P is None else P
        a = P[:-1, :-1]
        b = P[:-1, 1:]
        c = P[1:, 1:]
        d = P[1:, :-1]
        return a, b, c, d

    def signed_areas(self, P=None):
        a, b, c, d = self.quad_corners(P)

        def cross(u, v):
            return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

        return 0.5 * (cross(a, b) + cross(b, c) + cross(c, d) + cross(d, a))

    def fold_stats(self, P=None):
        """Corner-wise cross products: a healthy quad has 4 of the same sign."""
        a, b, c, d = self.quad_corners(P)

        def cr(p, q, r):
            u = q - p
            v = r - q
            return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

        c1 = cr(a, b, c)
        c2 = cr(b, c, d)
        c3 = cr(c, d, a)
        c4 = cr(d, a, b)
        s = np.sign(self.signed_areas(P))
        bad = ((np.sign(c1) != s) | (np.sign(c2) != s) |
               (np.sign(c3) != s) | (np.sign(c4) != s))
        A = self.signed_areas(P)
        inverted = (A * np.sign(self.hx * self.hy)) <= 0
        return float(bad.mean()), float(inverted.mean())


# ----------------------------------------------------------------------------
# 3.  forces
# ----------------------------------------------------------------------------

def wobble_home(lat, t, amp, T=LOOP_SEC):
    """Loop-periodic wobble of the rest grid (the 'base lattice keeps moving')."""
    if amp <= 0:
        return lat.R
    R = lat.R
    x = R[..., 0]
    y = R[..., 1]
    ph = 2 * math.pi * (t / T)
    k = 2 * math.pi / (7.3 * lat.h)
    dx = (np.sin(k * x * 0.9 + 1.7 * y * k * 0.45 + ph) * 0.55 +
          np.sin(k * 1.9 * y - k * 0.7 * x + 2 * ph + 1.1) * 0.30 +
          np.sin(k * 0.45 * x + k * 1.3 * y - 3 * ph + 2.4) * 0.15)
    dy = (np.cos(k * 1.1 * y - k * 0.6 * x + ph + 0.6) * 0.55 +
          np.cos(k * 0.8 * x + k * 1.6 * y - 2 * ph + 2.2) * 0.30 +
          np.cos(k * 1.7 * x - k * 0.9 * y + 3 * ph + 0.3) * 0.15)
    out = R.copy()
    out[..., 0] += amp * lat.h * dx
    out[..., 1] += amp * lat.h * dy
    return out


def _rot(v):
    """90-degree rotation used by the shoelace area gradient."""
    return np.stack([v[..., 1], -v[..., 0]], axis=-1)


class Solver:
    def __init__(self, lat, sdf, gx, gy, cfg, coh=None, curv=None):
        self.lat = lat
        self.sdf = sdf
        self.gx = gx
        self.gy = gy
        self.coh = coh
        self.curv = curv
        self.cfg = cfg
        self.A0 = abs(lat.hx * lat.hy)
        self.Ld = math.hypot(lat.hx, lat.hy)

    def forces(self, P, alpha, home):
        lat = self.lat
        cfg = self.cfg
        F = np.zeros_like(P)

        # --- axial springs -------------------------------------------------
        for axis, L0 in ((1, lat.hx), (0, lat.hy)):
            if axis == 1:
                d = P[:, 1:] - P[:, :-1]
            else:
                d = P[1:, :] - P[:-1, :]
            L = np.sqrt((d * d).sum(-1))[..., None] + 1e-9
            f = cfg["k_spring"] * (L - L0) * (d / L)
            if axis == 1:
                F[:, :-1] += f
                F[:, 1:] -= f
            else:
                F[:-1, :] += f
                F[1:, :] -= f

        # --- diagonal (shape) springs --------------------------------------
        if cfg["k_shape"] > 0:
            a, b, c, d4 = lat.quad_corners(P)
            for (p, q, si, sj, ti, tj) in (
                    (a, c, slice(None, -1), slice(None, -1),
                     slice(1, None), slice(1, None)),
                    (b, d4, slice(None, -1), slice(1, None),
                     slice(1, None), slice(None, -1))):
                dd = q - p
                L = np.sqrt((dd * dd).sum(-1))[..., None] + 1e-9
                f = cfg["k_shape"] * (L - self.Ld) * (dd / L)
                F[si, sj] += f
                F[ti, tj] -= f

        # --- signed area with collapse barrier -----------------------------
        self.last_gain = None
        if cfg["k_area"] > 0:
            a, b, c, d4 = lat.quad_corners(P)
            A = lat.signed_areas(P)
            sgn = np.sign(self.A0) if self.A0 else 1.0
            Aa = A * np.sign(lat.hx * lat.hy)
            err = (self.A0 - Aa) / self.A0
            # barrier: gain explodes as the quad squeezes toward zero area
            ratio = np.clip(Aa / self.A0, 1e-3, None)
            gain = cfg["k_area"] * np.clip(np.where(
                ratio < cfg["bar_thresh"],
                1.0 + cfg["bar_gain"] * (cfg["bar_thresh"] / ratio - 1.0) ** 2,
                1.0), 0.0, cfg["bar_cap"])
            self.last_gain = gain
            g = (gain * err)[..., None] * np.sign(lat.hx * lat.hy)

            # dA/dp_k = 0.5 * rot(p_{k+1} - p_{k-1}),  rot(u,v) = (v, -u)
            F[:-1, :-1] += g * 0.5 * _rot(b - d4)
            F[:-1, 1:] += g * 0.5 * _rot(c - a)
            F[1:, 1:] += g * 0.5 * _rot(d4 - b)
            F[1:, :-1] += g * 0.5 * _rot(a - c)

        # --- attraction to the zero crossing -------------------------------
        x = P[..., 0]
        y = P[..., 1]
        d = sample(self.sdf, x, y)
        ux = sample(self.gx, x, y)
        uy = sample(self.gy, x, y)
        n = np.sqrt(ux * ux + uy * uy) + 1e-6
        ux, uy = ux / n, uy / n
        sig = cfg["sigma"] * lat.h
        w = np.exp(-(d / sig) ** 2)
        if self.coh is not None:
            c = np.clip(sample(self.coh, x, y), 0.0, 1.0)
            w = w * c ** cfg["coh_pow"]
        if self.curv is not None and cfg["k_curv"] > 0:
            kv = sample(self.curv, x, y) * lat.h
            w = w / (1.0 + cfg["k_curv"] * kv * kv)
        if cfg["leash"] > 0:
            # a vertex that has already walked a cell away from home stops
            # being recruited, so a corner fan cannot all pile on one point
            dl = home - P
            r2 = (dl * dl).sum(-1) / (cfg["leash"] * lat.h) ** 2
            w = w * np.exp(-r2)
        mag = alpha * cfg["k_att"] * w * d
        F[..., 0] -= mag * ux
        F[..., 1] -= mag * uy
        self.last_d = d
        self.last_w = w

        # --- home / wobble spring ------------------------------------------
        F += cfg["k_home"] * (home - P)

        # --- Jacobi diagonal (stiffness felt by each vertex) ---------------
        diag = np.full(P.shape[:2], 4.0 * cfg["k_spring"] +
                       4.0 * cfg["k_shape"] + cfg["k_home"] + 1e-3)
        diag += alpha * cfg["k_att"] * w
        if cfg["k_area"] > 0:
            g = self.last_gain
            da = np.zeros(P.shape[:2])
            contrib = g * 0.5 * (abs(lat.hx) + abs(lat.hy))
            da[:-1, :-1] += contrib
            da[:-1, 1:] += contrib
            da[1:, 1:] += contrib
            da[1:, :-1] += contrib
            diag += da
        self.last_diag = diag
        return F, diag

    def project_areas(self, P, ratio_min, relax, sub=3):
        """Gauss-Newton projection of every quad onto  area >= ratio_min*A0.

        A projection, not a force: it cannot overshoot, so it cannot explode."""
        lat = self.lat
        sgn = 1.0 if lat.hx * lat.hy > 0 else -1.0
        Amin = ratio_min * self.A0
        frac = 0.0
        for _ in range(sub):
            a, b, c, d = lat.quad_corners(P)
            A = lat.signed_areas(P) * sgn
            C = A - Amin
            viol = C < 0
            frac = max(frac, float(viol.mean()))
            if not viol.any():
                break
            ga = 0.5 * _rot(b - d) * sgn
            gb = 0.5 * _rot(c - a) * sgn
            gc = 0.5 * _rot(d - b) * sgn
            gd = 0.5 * _rot(a - c) * sgn
            den = ((ga * ga).sum(-1) + (gb * gb).sum(-1) +
                   (gc * gc).sum(-1) + (gd * gd).sum(-1) + 1e-9)
            lam = (np.where(viol, -C / den, 0.0) * relax)[..., None]
            acc = np.zeros_like(P)
            cnt = np.zeros(P.shape[:2])
            acc[:-1, :-1] += lam * ga
            acc[:-1, 1:] += lam * gb
            acc[1:, 1:] += lam * gc
            acc[1:, :-1] += lam * gd
            cnt[:-1, :-1] += 1
            cnt[:-1, 1:] += 1
            cnt[1:, 1:] += 1
            cnt[1:, :-1] += 1
            P = P + acc / np.maximum(cnt, 1)[..., None]
        return P, frac

    def step(self, P, alpha, home, dt):
        F, diag = self.forces(P, alpha, home)
        dP = dt * F / diag[..., None]
        # clamp the per-iteration step so nothing can tunnel through a cell
        m = np.sqrt((dP * dP).sum(-1))[..., None]
        cap = self.cfg["step_cap"] * self.lat.h
        dP = np.where(m > cap, dP * (cap / (m + 1e-12)), dP)
        P = P + dP
        if self.cfg["proj_min"] > 0:
            P, self.viol_frac = self.project_areas(
                P, self.cfg["proj_min"], self.cfg["proj_relax"],
                self.cfg["proj_sub"])
        else:
            self.viol_frac = 0.0
        # boundary: outer ring stays on the overscan rectangle, slides freely
        x0, y0, x1, y1 = self.lat.rect
        P[0, :, 1] = y0
        P[-1, :, 1] = y1
        P[:, 0, 0] = x0
        P[:, -1, 0] = x1
        return P, float(np.abs(dP).max())


DEFAULT_CFG = dict(
    k_att=6.00,      # attraction gain
    sigma=0.85,      # Gaussian falloff, in cell widths
    k_spring=0.42,   # axial springs
    k_shape=0.10,    # diagonal springs
    k_area=0.000,    # soft area force (the projection below does the work)
    bar_thresh=0.34,
    bar_gain=14.0,
    bar_cap=60.0,
    proj_min=0.14,   # hard floor on cell area, as a fraction of the rest area
    proj_relax=0.85,
    proj_sub=6,
    k_home=0.012,    # weak anchoring / wobble follow
    coh_pow=3.0,     # how hard to suppress attraction on medial axes
    k_curv=1.0,      # damp attraction inside high-curvature corner fans
    leash=0.0,       # attraction dies once a vertex is this many cells from home
    step_cap=0.20,
    dt=0.25,         # Jacobi over-relaxation factor
    cool_frac=0.18,  # anneal the last 18% of a static solve to equilibrium
    frame_iters=14,   # solver iterations per animation frame (warm started)
)


def settle(lat, sdf, gx, gy, cfg, iters=420, ramp_frac=0.55, wobble=0.0,
           t=0.0, record=None, P0=None, verbose=False, coh=None, curv=None):
    """Ramp alpha 0 -> 1 then hold, then cool.  Returns (P, history)."""
    sol = Solver(lat, sdf, gx, gy, cfg, coh, curv)
    P = lat.P.copy() if P0 is None else P0.copy()
    home = wobble_home(lat, t, wobble)
    hist = []
    vhist = []
    ramp_n = max(1, int(iters * ramp_frac))
    for i in range(iters):
        a = min(1.0, i / ramp_n)
        a = a * a * (3 - 2 * a)          # smoothstep on the ramp
        cool = 1.0
        if cfg["cool_frac"] > 0:
            c0 = 1.0 - cfg["cool_frac"]
            if i > iters * c0:
                cool = 1.0 - 0.85 * (i - iters * c0) / (iters * cfg["cool_frac"])
        P, mx = sol.step(P, a, home, cfg["dt"] * cool)
        hist.append(mx)
        vhist.append(sol.viol_frac)
        if record is not None:
            for frac, store in record:
                if i == min(iters - 1, int(frac * (iters - 1))):
                    store.append((a, P.copy()))
        if verbose and i % 60 == 0:
            log("    it %4d alpha=%.2f maxstep=%.3f" % (i, a, mx))
    sol.vhist = vhist
    return P, hist, sol


# ----------------------------------------------------------------------------
# 4.  rendering
# ----------------------------------------------------------------------------

def wireframe(lat, P, ss=3, line_w=2, fg=(30, 30, 36), bg=(250, 250, 248),
              size=(W, H)):
    """The acceptance test: every lattice edge, one colour, uniform width."""
    Wo, Ho = size
    img = Image.new("RGB", (Wo * ss, Ho * ss), bg)
    dr = ImageDraw.Draw(img)
    sx = Wo / float(W)
    sy = Ho / float(H)
    Q = P.copy()
    Q[..., 0] *= sx * ss
    Q[..., 1] *= sy * ss
    for r in range(Q.shape[0]):
        dr.line([tuple(v) for v in Q[r]], fill=fg, width=line_w, joint="curve")
    for c in range(Q.shape[1]):
        dr.line([tuple(v) for v in Q[:, c]], fill=fg, width=line_w,
                joint="curve")
    return img.resize((Wo, Ho), Image.LANCZOS)


def cell_ids(lat, P, sdf, rng, families):
    """Clip id per quad: family by SDF sign at the centroid, shuffled inside."""
    a, b, c, d = lat.quad_corners(P)
    cen = (a + b + c + d) / 4.0
    dv = sample(sdf, cen[..., 0], cen[..., 1])
    inside = dv < 0
    ids = np.empty(inside.shape, dtype=np.int32)
    fin = families["inside"]
    fout = families["outside"]
    ids[inside] = np.array(fin)[rng.integers(0, len(fin), int(inside.sum()))]
    ids[~inside] = np.array(fout)[rng.integers(0, len(fout),
                                               int((~inside).sum()))]
    return ids, inside


def composite(lat, P, ids, clips, size=(W, H)):
    """Paint each quad with its clip's pixels.  Nothing else is drawn."""
    Wo, Ho = size
    ss = 2
    idmap = Image.new("I", (Wo * ss, Ho * ss), 0)
    dr = ImageDraw.Draw(idmap)
    sx = Wo / float(W) * ss
    sy = Ho / float(H) * ss
    a, b, c, d = lat.quad_corners(P)
    ny, nx = ids.shape
    for j in range(ny):
        for i in range(nx):
            poly = [a[j, i], b[j, i], c[j, i], d[j, i]]
            dr.polygon([(p[0] * sx, p[1] * sy) for p in poly],
                       fill=int(ids[j, i]) + 1)
    idn = np.asarray(idmap)
    idn = idn[::ss, ::ss]
    idn = np.clip(idn - 1, 0, len(clips) - 1)
    stack = np.stack(clips)  # (K, H, W, 3)
    if stack.shape[1] != Ho or stack.shape[2] != Wo:
        stack = np.stack([np.asarray(Image.fromarray(s).resize((Wo, Ho)))
                          for s in stack])
    out = np.take_along_axis(stack, idn[None, ..., None], axis=0)[0]
    return Image.fromarray(out.astype(np.uint8))


def load_clips():
    fr = os.path.join(OUT, "frames")
    cached = os.path.join(SCRATCH, "agent-d", "frames")
    clips = []
    for i in range(4):
        p = os.path.join(fr, "clip%d.png" % i)
        if not os.path.exists(p):
            src = os.path.join(cached, "clip%d.png" % i)
            if os.path.exists(src):
                shutil.copyfile(src, p)
            else:
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-ss", "1.2", "-i",
                     CLIP_SOURCES[i], "-frames:v", "1", "-vf",
                     "scale=%d:%d:force_original_aspect_ratio=increase,"
                     "crop=%d:%d" % (W, H, W, H), "-y", p], check=True)
        clips.append(np.asarray(Image.open(p).convert("RGB")))
    return clips


def luma_families(clips):
    y = [float((0.2126 * c[..., 0] + 0.7152 * c[..., 1] +
                0.0722 * c[..., 2]).mean()) for c in clips]
    order = np.argsort(y)
    return dict(outside=[int(order[0]), int(order[1])],
                inside=[int(order[2]), int(order[3])]), y


# ----------------------------------------------------------------------------
# 5.  sheets
# ----------------------------------------------------------------------------

def caption_grid(tiles, cols, tile_w, pad=10, cap_h=26, title=None,
                 bg=(24, 24, 28), fg=(235, 235, 235)):
    from PIL import ImageFont as IF
    try:
        f = IF.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
        ft = IF.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                         21)
    except Exception:
        f = IF.load_default()
        ft = IF.load_default()
    rows = (len(tiles) + cols - 1) // cols
    th = int(tile_w * H / W)
    top = 40 if title else 0
    Wt = cols * tile_w + (cols + 1) * pad
    Ht = top + rows * (th + cap_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (Wt, Ht), bg)
    dr = ImageDraw.Draw(sheet)
    if title:
        dr.text((pad, 10), title, fill=fg, font=ft)
    for k, (img, cap) in enumerate(tiles):
        r, c = divmod(k, cols)
        x = pad + c * (tile_w + pad)
        y = top + pad + r * (th + cap_h + pad)
        sheet.paste(img.resize((tile_w, th), Image.LANCZOS), (x, y))
        for li, line in enumerate(cap.split("\n")[:2]):
            dr.text((x + 2, y + th + 2 + li * 13), line, fill=fg, font=f)
    return sheet


# ----------------------------------------------------------------------------
# 6.  experiments
# ----------------------------------------------------------------------------

def mk(word, nx_vis, cfg=None, iters=420, wobble=0.0, cap=213.0, t=0.0,
       verbose=False):
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    mask, info = render_word_mask(word, cap_px=cap)
    sdf, gx, gy, coh, curv = build_sdf(mask)
    lat = Lattice(nx_vis)
    P, hist, sol = settle(lat, sdf, gx, gy, cfg, iters=iters, wobble=wobble,
                          t=t, verbose=verbose, coh=coh, curv=curv)
    return dict(lat=lat, P=P, sdf=sdf, gx=gx, gy=gy, coh=coh, curv=curv,
                hist=hist, info=info, cfg=cfg, word=word, nx=nx_vis, sol=sol)


# ----------------------------------------------------------------------------
# 7.  WARP  --  one video, textured through the settled mesh
#
# The undeformed lattice R is a set of coordinates in the SOURCE frame; the
# settled lattice P is the same mesh in SCREEN space.  Rendering is therefore a
# texture-mapped mesh and nothing else: no selector mask, no clip ids, no
# compositing, no stroke, no shading, no tint.  Only the image is moved.
# ----------------------------------------------------------------------------

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    cv2 = None
    HAVE_CV2 = False


def inverse_map(lat, P, size=(W, H), newton=12, relax=0.92, warm=None):
    """Destination pixel -> source pixel.

    The forward map (source -> screen) is just bilinear interpolation of P over
    the *regular* undeformed grid, so it is cheap to evaluate and cheap to
    differentiate.  Invert it per destination pixel with damped Newton, started
    from the identity.  Fully vectorised: the six fields we need (Fx, Fy and
    the four Jacobian entries) are gathered together in one shot per step."""
    Wo, Ho = size
    Fx = P[..., 0]
    Fy = P[..., 1]
    stack = np.stack([
        Fx, Fy,
        np.gradient(Fx, axis=1) / lat.hx, np.gradient(Fx, axis=0) / lat.hy,
        np.gradient(Fy, axis=1) / lat.hx, np.gradient(Fy, axis=0) / lat.hy,
    ]).astype(np.float32)

    nx, ny = lat.nx, lat.ny

    def bil(u, v):
        u = np.clip(u, 0.0, nx - 1e-4)
        v = np.clip(v, 0.0, ny - 1e-4)
        i0 = u.astype(np.int32)
        j0 = v.astype(np.int32)
        tu = (u - i0)[None]
        tv = (v - j0)[None]
        return (stack[:, j0, i0] * (1 - tu) * (1 - tv) +
                stack[:, j0, i0 + 1] * tu * (1 - tv) +
                stack[:, j0 + 1, i0] * (1 - tu) * tv +
                stack[:, j0 + 1, i0 + 1] * tu * tv)

    ys, xs = np.mgrid[0:Ho, 0:Wo]
    px = (xs * (W / float(Wo))).astype(np.float32)
    py = (ys * (H / float(Ho))).astype(np.float32)
    if warm is None:
        sx, sy = px.copy(), py.copy()
    else:
        sx, sy = warm[0].copy(), warm[1].copy()
    x1 = lat.x0 + lat.nx * lat.hx
    y1 = lat.y0 + lat.ny * lat.hy
    cap = 1.5 * lat.h
    for _ in range(newton):
        f = bil((sx - lat.x0) / lat.hx, (sy - lat.y0) / lat.hy)
        rx = px - f[0]
        ry = py - f[1]
        a, b, c, d = f[2], f[3], f[4], f[5]
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-3, 1e-3, det)
        dx = relax * (d * rx - b * ry) / det
        dy = relax * (-c * rx + a * ry) / det
        # a clamped Newton step: the map is only locally invertible, and an
        # unclamped step at a near-singular cell throws the pixel across the
        # canvas and never comes back
        sc = np.minimum(1.0, cap / (np.hypot(dx, dy) + 1e-9))
        sx = np.clip(sx + dx * sc, lat.x0, x1)
        sy = np.clip(sy + dy * sc, lat.y0, y1)
    return sx.astype(np.float32), sy.astype(np.float32)


def warp_image(src, sx, sy):
    """Sample the source with the inverse map.  This is the entire renderer."""
    if HAVE_CV2:
        return cv2.remap(src, sx, sy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    out = np.empty(sx.shape + (3,), np.uint8)
    for ch in range(3):
        out[..., ch] = ndimage.map_coordinates(
            src[..., ch], [sy, sx], order=1, mode="reflect")
    return out


def map_residual(lat, P, sx, sy, size=(W, H)):
    """How well the inverse map inverts: |F(s) - p| in destination pixels."""
    Wo, Ho = size
    u = np.clip((sx - lat.x0) / lat.hx, 0, lat.nx - 1e-4)
    v = np.clip((sy - lat.y0) / lat.hy, 0, lat.ny - 1e-4)
    i0 = u.astype(np.int32)
    j0 = v.astype(np.int32)
    tu = u - i0
    tv = v - j0
    f = (P[j0, i0] * ((1 - tu) * (1 - tv))[..., None] +
         P[j0, i0 + 1] * (tu * (1 - tv))[..., None] +
         P[j0 + 1, i0] * ((1 - tu) * tv)[..., None] +
         P[j0 + 1, i0 + 1] * (tu * tv)[..., None])
    ys, xs = np.mgrid[0:Ho, 0:Wo]
    r = np.hypot(f[..., 0] - xs * (W / float(Wo)),
                 f[..., 1] - ys * (H / float(Ho)))
    return float(np.median(r)), float(np.percentile(r, 99.9)), float(r.max())


def tri_fold_rate(lat, P):
    """Fraction of mesh triangles that have flipped.  This is the metric that
    matters for a texture warp: a flipped triangle folds the image back on
    itself."""
    a, b, c, d = lat.quad_corners(P)
    sgn = 1.0 if lat.hx * lat.hy > 0 else -1.0

    def cr(p, q, r):
        u = q - p
        v = r - p
        return (u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]) * sgn

    t = np.stack([cr(a, b, c), cr(a, c, d), cr(a, b, d), cr(b, c, d)])
    return float((t <= 0).any(0).mean()), float((t <= 0).mean())


# ---- the attenuation cycle -------------------------------------------------

CYCLE = [(0.00, 0.0), (1.20, 0.0), (4.20, 1.0), (7.60, 1.0), (10.60, 0.0),
         (12.00, 0.0)]


def cycle_alpha(t, T=LOOP_SEC):
    """0 -> settle -> hold -> release -> 0.  Smoothstep segments, so the
    derivative is zero at every keyframe and the loop seam is C1."""
    t = t % T
    for (t0, a0), (t1, a1) in zip(CYCLE, CYCLE[1:]):
        if t0 <= t <= t1:
            if a0 == a1:
                return a0
            u = (t - t0) / (t1 - t0)
            return a0 + (a1 - a0) * (u * u * (3 - 2 * u))
    return 0.0


def load_src_frames(name, window=None):
    """Frame paths for a clip.  `window` = (start, length) selects a stretch of
    the clip and ping-pongs it, which both keeps the source loop seamless and
    lets me stay inside the part of a clip where the subject fills the frame."""
    d = os.path.join(OUT, "src", name)
    fs = sorted(os.path.join(d, f) for f in os.listdir(d)
                if f.endswith(".jpg"))
    if window is None:
        return fs
    a, n = window
    seq = fs[a:a + n]
    return seq + seq[-2:0:-1]


SRC_WINDOW = {"yope": (0, 60)}


def run_warp(word="Yope3D", nx=96, clip="yope", wob=0.30, sec=LOOP_SEC,
             tag=None, strip=True, video=True, cfg=None, iters_per_frame=None):
    tag = tag or clip
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    per = iters_per_frame or cfg["frame_iters"]
    mask, info = render_word_mask(word)
    sdf, gx, gy, coh, curv = build_sdf(mask)
    lat = Lattice(nx)
    sol = Solver(lat, sdf, gx, gy, cfg, coh, curv)
    srcs = load_src_frames(clip, SRC_WINDOW.get(clip))
    nfr = int(sec * FPS)

    # warm the idle state so frame 0 is already breathing, not a cold grid
    P = lat.P.copy()
    for i in range(240):
        P, _ = sol.step(P, 0.0, wobble_home(lat, 0.0, wob), cfg["dt"])

    proc = None
    if video:
        out_mp4 = os.path.join(OUT, "warp-cycle-%s.mp4" % tag)
        proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
             "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", out_mp4], stdin=subprocess.PIPE)

    want_t = [0.0, 1.6, 2.2, 2.8, 3.5, 4.6, 7.0, 8.6, 9.6, 11.0]
    want_f = {int(round(v * FPS)) % nfr: v for v in want_t}
    shots = {}
    tim = dict(solve=0.0, inv=0.0, remap=0.0, io=0.0)
    peak = None
    stats = []
    prev = P.copy()
    for f in range(nfr):
        t = f / float(FPS)
        a = cycle_alpha(t, sec)
        home = wobble_home(lat, t, wob, sec)
        t0 = time.time()
        for _ in range(per):
            P, _ = sol.step(P, a, home, cfg["dt"])
        tim["solve"] += time.time() - t0
        t0 = time.time()
        sx, sy = inverse_map(lat, P, newton=(14 if f == 0 else 4),
                             warm=None if f == 0 else (sx, sy))
        tim["inv"] += time.time() - t0
        t0 = time.time()
        src = np.asarray(Image.open(srcs[f % len(srcs)]).convert("RGB"))
        tim["io"] += time.time() - t0
        t0 = time.time()
        img = warp_image(src, sx, sy)
        tim["remap"] += time.time() - t0
        mv = float(np.sqrt(((P - prev) ** 2).sum(-1)).max())
        prev = P.copy()
        tq, tt = tri_fold_rate(lat, P)
        stats.append(dict(f=f, t=round(t, 3), alpha=round(a, 4),
                          maxmove=round(mv, 3), tri_fold=round(tt, 5),
                          quad_fold=round(tq, 5)))
        if proc:
            proc.stdin.write(img.tobytes())
        if f in want_f:
            shots[f] = (Image.fromarray(img), a, want_f[f])
        if abs(t - 6.0) < 1e-6 or (peak is None and a >= 1.0):
            peak = (Image.fromarray(img), lat, P.copy(), sx, sy)
    if proc:
        proc.stdin.close()
        proc.wait()

    if peak is not None:
        peak[0].save(os.path.join(OUT, "warp-hold-%s.png" % tag))
        if tag == "yope":
            peak[0].save(os.path.join(OUT, "warp-hold.png"))
    if strip:
        tiles = [(shots[f][0], "t=%.1fs   alpha=%.2f%s" %
                  (shots[f][2], shots[f][1],
                   ["", "   <- idle, breathing only"][shots[f][1] < 0.02]))
                 for f in sorted(shots)]
        sheet = caption_grid(
            tiles, 2, 880, cap_h=30,
            title="agent-f  attenuation cycle, PURE WARP of one clip  "
                  "(%s, %s, %d cells, wobble %.2f)   no stroke, no shading, "
                  "no tint - only the image moves" % (word, clip, nx, wob))
        sheet.save(os.path.join(OUT, "warp-strip-%s.png" % tag))
        if tag == "yope":
            sheet.save(os.path.join(OUT, "warp-strip.png"))
    res = map_residual(lat, peak[2] if peak else P, peak[3], peak[4]) \
        if peak else (0, 0, 0)
    tq, tt = tri_fold_rate(lat, peak[2] if peak else P)
    summary = dict(word=word, nx=nx, clip=clip, wobble=wob, cells_px=lat.h,
                   frames=nfr, iters_per_frame=per,
                   ms_solve=round(1000 * tim["solve"] / nfr, 2),
                   ms_inverse_map=round(1000 * tim["inv"] / nfr, 2),
                   ms_remap=round(1000 * tim["remap"] / nfr, 2),
                   ms_decode=round(1000 * tim["io"] / nfr, 2),
                   ms_total=round(1000 * sum(tim.values()) / nfr, 2),
                   map_residual_px=dict(median=round(res[0], 4),
                                        p999=round(res[1], 3),
                                        max=round(res[2], 3)),
                   peak_tri_fold=round(tt, 5), peak_quad_fold=round(tq, 5),
                   max_move_px=round(max(s["maxmove"] for s in stats), 3),
                   median_move_px=round(
                       float(np.median([s["maxmove"] for s in stats])), 3),
                   stats=stats)
    json.dump(summary, open(os.path.join(OUT, "warp-%s.json" % tag), "w"),
              indent=1)
    log("[warp:%s] %s" % (tag, {k: v for k, v in summary.items()
                                if k != "stats"}))
    return summary


def run_warp_compare(word="Yope3D", clip="yope", densities=(48, 64, 96, 128),
                     wob=0.30):
    """Same peak frame at several lattice densities."""
    srcs = load_src_frames(clip, SRC_WINDOW.get(clip))
    src = np.asarray(Image.open(srcs[len(srcs) // 3]).convert("RGB"))
    tiles = [(Image.fromarray(src), "source frame, undeformed")]
    for nx in densities:
        cfg = dict(DEFAULT_CFG)
        mask, info = render_word_mask(word)
        sdf, gx, gy, coh, curv = build_sdf(mask)
        lat = Lattice(nx)
        P, hist, sol = settle(lat, sdf, gx, gy, cfg, iters=iters_for(nx),
                              wobble=wob, coh=coh, curv=curv)
        sx, sy = inverse_map(lat, P)
        img = warp_image(src, sx, sy)
        tq, tt = tri_fold_rate(lat, P)
        tiles.append((Image.fromarray(img),
                      "%d x %d cells  h=%.1fpx\ntriangle folds %.3f%%" %
                      (lat.nx_vis, int(round(H / lat.hy)), lat.h, 100 * tt)))
        log("[compare] nx=%d h=%.1f trifold=%.5f" % (nx, lat.h, tt))
        caption_grid(tiles, 2, 880, cap_h=30,
                     title="agent-f  peak warp vs lattice density (%s, %s)"
                           % (word, clip)
                     ).save(os.path.join(OUT, "warp-compare.png"))


def run_alpha_ladder(word="Yope3D", nx=96, clip="yope", wob=0.30):
    """Static readability threshold: freeze alpha at a ladder of values."""
    srcs = load_src_frames(clip, SRC_WINDOW.get(clip))
    src = np.asarray(Image.open(srcs[len(srcs) // 3]).convert("RGB"))
    mask, info = render_word_mask(word)
    sdf, gx, gy, coh, curv = build_sdf(mask)
    cfg = dict(DEFAULT_CFG)
    lat = Lattice(nx)
    sol = Solver(lat, sdf, gx, gy, cfg, coh, curv)
    P = lat.P.copy()
    home = wobble_home(lat, 0.0, wob)
    tiles = []
    for a in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
        for _ in range(260):
            P, _ = sol.step(P, a, home, cfg["dt"])
        sx, sy = inverse_map(lat, P)
        tiles.append((Image.fromarray(warp_image(src, sx, sy)),
                      "alpha = %.2f   (warp)" % a))
        tiles.append((wireframe(lat, P), "alpha = %.2f   (mesh)" % a))
        caption_grid(tiles, 2, 780, cap_h=28,
                     title="agent-f  readability vs attraction strength "
                           "(%s, %d cells)" % (word, nx)
                     ).save(os.path.join(OUT, "warp-alpha-ladder.png"))
        log("[ladder] alpha=%.2f" % a)



# ----------------------------------------------------------------------------
# 10.  SHADING  --  agent G's Jacobian brightness term, inside the hybrid
#
# Strength is a Gaussian in AMPLITUDE, not in time, so it auto-centres on the
# forming moment whatever the settle duration is.  Schedule taken verbatim from
# agent G so the two are comparable.
# ----------------------------------------------------------------------------

FLARE_HOLD = 0.55
FLARE_PEAK = 1.15
FLARE_MU = 0.45
FLARE_SIG = 0.28


def flare_strength(alpha):
    return FLARE_HOLD + FLARE_PEAK * math.exp(
        -((alpha - FLARE_MU) / FLARE_SIG) ** 2)


def release_gate(t, sec=LOOP_SEC, frac=0.55):
    """On release the shading collapses to zero AHEAD of the amplitude, so the
    word loses its edge while the mesh is still rippling and disperses instead
    of rewinding."""
    t0, t1 = CYCLE[3][0], CYCLE[4][0]          # the release segment
    t = t % sec
    if t <= t0:
        return 1.0
    span = frac * (t1 - t0)
    if t >= t0 + span:
        return 0.0
    u = (t - t0) / span
    return 1.0 - u * u * (3 - 2 * u)


def jacobian_shade(mx, my, ss, strength, gain=1.1, lo=0.30, hi=1.95):
    """Brightness from the local area scale of the inverse map.

    det > 1  : the destination is compressed relative to the source, i.e. the
               material is being squeezed here  -> darker.
    det < 1  : stretched -> brighter.
    With this algorithm the compression is concentrated in a thin band on the
    contour, so the result is a dark traced outline: an engraving, not a
    filled deboss."""
    if strength <= 0:
        return None
    a = np.gradient(mx, axis=1) * ss
    b = np.gradient(mx, axis=0) * ss
    c = np.gradient(my, axis=1) * ss
    d = np.gradient(my, axis=0) * ss
    det = np.abs(a * d - b * c)
    det = ndimage.gaussian_filter(det, 1.0 * ss)
    lg = np.log(np.clip(det, 1e-3, 1e3))
    return np.clip(1.0 - strength * np.tanh(gain * lg), lo, hi)


def smooth_funnel_map(sdf, gx, gy, amp, sig_px):
    """Emulation of the OTHER look for comparison purposes (not agent G's
    code): a single smooth global displacement that funnels material across the
    whole letter interior toward the rim, which is what produces filled
    debossed letters rather than a traced outline."""
    ys, xs = np.mgrid[0:H, 0:W]
    px = xs.astype(np.float32)
    py = ys.astype(np.float32)
    d = sample(sdf, px, py)
    ux = sample(gx, px, py)
    uy = sample(gy, px, py)
    n = np.sqrt(ux * ux + uy * uy) + 1e-6
    ux, uy = ux / n, uy / n
    prof = amp * d * np.exp(-(d / sig_px) ** 2)
    return (px + prof * ux).astype(np.float32), \
           (py + prof * uy).astype(np.float32)


# ----------------------------------------------------------------------------
# 9.  HYBRID  --  warped geometry + clip contrast
#
# The force-settled lattice supplies the geometry: every cell warps the footage
# it carries, so the letterform is a real deformation of the image and the
# crease is genuine.  Clip selection supplies the contrast a smooth invertible
# warp provably cannot create: cells inside the letters draw from a family of
# clips disjoint from the cells outside, split by luminance, so no clip id ever
# appears on both sides of the contour.
# ----------------------------------------------------------------------------

NCLIP = 4


def clip_dirs():
    return [os.path.join(OUT, "src", "clip%d" % k) for k in range(NCLIP)]


def clip_frames():
    out = []
    for d in clip_dirs():
        out.append(sorted(os.path.join(d, f) for f in os.listdir(d)
                          if f.endswith(".jpg")))
    return out


def loop_seq(n, nfr):
    """Exactly nfr indices into an n-frame clip, ping-ponged, period == nfr."""
    half = nfr // 2
    fwd = [int(round(i * (n - 1) / float(half - 1))) for i in range(half)]
    return fwd + fwd[::-1]


def clip_luma(frames, samples=10):
    ys = []
    for fs in frames:
        v = []
        for f in fs[::max(1, len(fs) // samples)][:samples]:
            a = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
            v.append(float((0.2126 * a[..., 0] + 0.7152 * a[..., 1] +
                            0.0722 * a[..., 2]).mean()))
        ys.append(float(np.mean(v)))
    return ys


def luma_split(frames):
    """Round-2 palette rule: two brightest inside, two darkest outside."""
    y = clip_luma(frames)
    order = list(np.argsort(y))
    fam = dict(outside=[int(order[0]), int(order[1])],
               inside=[int(order[2]), int(order[3])])
    return fam, y


class CellPalette:
    """Per-cell clip assignment and its attenuation.

    `fam`  : the clip this cell settles to, drawn from its own family only.
    `idle` : what the cell shows at amplitude 0 (a jumble of all four clips,
             or a single clip, depending on idle_mode).
    `tau`  : the amplitude at which the cell switches over.  Cells cross at
             different amplitudes, so the palette dissolves rather than pops."""

    def __init__(self, lat, P_ref, sdf, fam, rng, idle_mode="jumble",
                 order="contour-first", lo=0.10, hi=0.80):
        a, b, c, d = lat.quad_corners(P_ref)
        cen = (a + b + c + d) / 4.0
        dv = sample(sdf, cen[..., 0], cen[..., 1])
        self.inside = dv < 0
        sh = self.inside.shape
        fin = np.array(fam["inside"])
        fout = np.array(fam["outside"])
        self.fam_clip = np.where(
            self.inside,
            fin[rng.integers(0, len(fin), sh)],
            fout[rng.integers(0, len(fout), sh)]).astype(np.int32)
        if idle_mode == "jumble":
            self.idle_clip = rng.integers(0, NCLIP, sh).astype(np.int32)
        elif idle_mode == "single":
            self.idle_clip = np.zeros(sh, np.int32)
        else:                                     # "family" = no dissolve
            self.idle_clip = self.fam_clip.copy()
        u = rng.random(sh)
        if order == "contour-first":
            # cells near the contour commit first, so the letter resolves from
            # its own edge outward instead of dissolving at random
            r = np.clip(np.abs(dv) / (6.0 * lat.h), 0, 1)
            u = np.clip(0.72 * r + 0.28 * u, 0, 1)
        self.tau = lo + (hi - lo) * u
        self.idle_mode = idle_mode

    def ids(self, alpha):
        return np.where(alpha >= self.tau, self.fam_clip,
                        self.idle_clip).astype(np.int32)


def hybrid_render(lat, sx, sy, ids, srcs, ss=2, shade_strength=0.0,
                  shade_gain=1.1):
    """One cell, one clip, warped.  The cell a destination pixel belongs to is
    read straight out of the inverse map: the source coordinate already says
    which undeformed cell the pixel came from."""
    if ss > 1:
        mx = cv2.resize(sx, (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(sy, (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
    else:
        mx, my = sx, sy
    i = np.clip(((mx - lat.x0) / lat.hx).astype(np.int32), 0, lat.nx - 1)
    j = np.clip(((my - lat.y0) / lat.hy).astype(np.int32), 0, lat.ny - 1)
    sel = ids[j, i]
    warped = np.stack([cv2.remap(s, mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
                       for s in srcs])
    out = np.take_along_axis(warped, sel[None, ..., None], 0)[0]
    if shade_strength > 0:
        shv = jacobian_shade(mx, my, ss, shade_strength, shade_gain)
        out = np.clip(out.astype(np.float32) * shv[..., None], 0, 255)
        out = out.astype(np.uint8)
    if ss > 1:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    return out


def identity_map():
    ys, xs = np.mgrid[0:H, 0:W]
    return xs.astype(np.float32), ys.astype(np.float32)


def seam_contrast(lat, P, ids, srcs):
    """Mean |luma| step across every cell edge whose two cells differ in clip.
    This is the contrast a pure warp cannot create."""
    lum = [float((0.2126 * s[..., 0] + 0.7152 * s[..., 1] +
                  0.0722 * s[..., 2]).mean()) for s in srcs]
    lum = np.array(lum)
    v = []
    for axis in (0, 1):
        if axis == 0:
            a, b = ids[:-1, :], ids[1:, :]
        else:
            a, b = ids[:, :-1], ids[:, 1:]
        m = a != b
        if m.any():
            v.append(np.abs(lum[a[m]] - lum[b[m]]))
    return float(np.concatenate(v).mean()) if v else 0.0


# ----------------------------------------------------------------------------
# 8.  wireframe experiments (the acceptance test and its sweeps)
# ----------------------------------------------------------------------------

def iters_for(nx):
    return int(600 + 9 * nx)


def run_density(word="Yope3D", densities=(24, 32, 40, 48, 56, 64, 80, 96,
                                          128)):
    tiles, rows = [], []
    for nx in densities:
        t = time.time()
        r = mk(word, nx, iters=iters_for(nx))
        lat = r["lat"]
        bad, inv = lat.fold_stats(r["P"])
        tq, tt = tri_fold_rate(lat, r["P"])
        tiles.append((wireframe(lat, r["P"]),
                      "%s  %d x %d cells  h=%.1fpx\nnonconvex=%.3f "
                      "inverted=%.4f trifold=%.4f" %
                      (word, lat.nx_vis, int(round(H / lat.hy)), lat.h, bad,
                       inv, tt)))
        rows.append(dict(word=word, nx=nx, h=round(lat.h, 2),
                         nonconvex=round(bad, 4), inverted=round(inv, 4),
                         tri_fold=round(tt, 5),
                         cells_per_stroke=round(r["info"]["stroke_px"] / lat.h,
                                                2),
                         secs=round(time.time() - t, 1)))
        log("[density] %s" % rows[-1])
        caption_grid(tiles, 3, 560,
                     title="agent-f  wireframe vs lattice density  "
                           "(single colour, no highlight, no overlay)"
                     ).save(os.path.join(OUT, "wireframe-sheet.png"))
        json.dump(rows, open(os.path.join(OUT, "density.json"), "w"), indent=1)


def run_formation(word="Yope3D", nx=64, stages=8):
    mask, info = render_word_mask(word)
    sdf, gx, gy, coh, curv = build_sdf(mask)
    cfg = dict(DEFAULT_CFG)
    lat = Lattice(nx)
    sol = Solver(lat, sdf, gx, gy, cfg, coh, curv)
    iters = iters_for(nx)
    ramp_n = int(iters * 0.55)
    P = lat.P.copy()
    home = lat.R
    marks = [int(k * (iters - 1) / (stages - 1)) for k in range(stages)]
    tiles, hist = [], []
    for i in range(iters):
        a = min(1.0, i / ramp_n)
        a = a * a * (3 - 2 * a)
        P, mx = sol.step(P, a, home, cfg["dt"])
        hist.append((a, mx))
        if i in marks:
            bad, inv = lat.fold_stats(P)
            tiles.append((wireframe(lat, P),
                          "iter %d   alpha=%.2f\nnonconvex=%.3f" %
                          (i, a, bad)))
    caption_grid(tiles, 4, 470,
                 title="agent-f  formation ramp: attraction 0 -> 1  "
                       "(%s, %d cells)" % (word, nx)
                 ).save(os.path.join(OUT, "formation-strip.png"))
    json.dump([dict(alpha=round(a, 3), maxstep=round(mx, 4))
               for a, mx in hist[::10]],
              open(os.path.join(OUT, "formation.json"), "w"), indent=1)
    log("[formation] %d iters, final maxstep %.4f" % (iters, hist[-1][1]))


def run_wobble(word="Yope3D", nx=64):
    tiles = []
    for wob in (0.0, 0.18, 0.35, 0.55):
        r = mk(word, nx, iters=iters_for(nx), wobble=wob)
        bad, inv = r["lat"].fold_stats(r["P"])
        tq, tt = tri_fold_rate(r["lat"], r["P"])
        tiles.append((wireframe(r["lat"], r["P"]),
                      "wobble=%.2f cells  nonconvex=%.3f trifold=%.4f" %
                      (wob, bad, tt)))
        log("[wobble] %.2f nonconvex=%.3f trifold=%.4f" % (wob, bad, tt))
    caption_grid(tiles, 2, 620,
                 title="agent-f  wobble amplitude (fraction of a cell)"
                 ).save(os.path.join(OUT, "wobble-sheet.png"))



def run_jacobian_probe(word="Yope3D", nx=96, clip="yope", wob=0.30):
    """SIDE EXPERIMENT, NOT THE DELIVERABLE.

    Ask whether a brightness term derived from the warp's own Jacobian (i.e.
    how much each destination pixel is locally compressed) would help
    readability.  This is still not painting a glyph on top -- the quantity
    comes from the deformation itself -- but it IS a change to the pixels
    beyond moving them, so it is reported separately and is off by default."""
    srcs = load_src_frames(clip, SRC_WINDOW.get(clip))
    src = np.asarray(Image.open(srcs[len(srcs) // 3]).convert("RGB"))
    mask, info = render_word_mask(word)
    sdf, gx, gy, coh, curv = build_sdf(mask)
    cfg = dict(DEFAULT_CFG)
    lat = Lattice(nx)
    P, hist, sol = settle(lat, sdf, gx, gy, cfg, iters=iters_for(nx),
                          wobble=wob, coh=coh, curv=curv)
    sx, sy = inverse_map(lat, P, newton=14)
    pure = warp_image(src, sx, sy)
    # local area scale of the inverse map: |d(source)/d(dest)|
    jxx = np.gradient(sx, axis=1)
    jxy = np.gradient(sx, axis=0)
    jyx = np.gradient(sy, axis=1)
    jyy = np.gradient(sy, axis=0)
    det = np.abs(jxx * jyy - jxy * jyx)
    det = ndimage.gaussian_filter(det, 1.5)
    tiles = [(Image.fromarray(src), "source, undeformed"),
             (Image.fromarray(pure), "PURE WARP - the deliverable")]
    for g in (0.25, 0.5):
        sh = np.clip(1.0 + g * (1.0 - np.clip(det, 0.2, 3.0)), 0.35, 1.9)
        out = np.clip(pure.astype(np.float32) * sh[..., None], 0, 255)
        tiles.append((Image.fromarray(out.astype(np.uint8)),
                      "SIDE EXPERIMENT (not the deliverable): Jacobian "
                      "brightness, gain %.2f" % g))
    caption_grid(tiles, 2, 880, cap_h=30,
                 title="agent-f  would a Jacobian-derived brightness term "
                       "help?  (reported separately, OFF by default)"
                 ).save(os.path.join(OUT, "warp-jacobian-side-experiment.png"))
    log("[jacobian] det range %.3f..%.3f" % (float(det.min()),
                                             float(det.max())))


def _hybrid_setup(word, nx, wob, cfg=None, seed=11, idle_mode="jumble",
                  order="contour-first"):
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    mask, info = render_word_mask(word)
    sdf, gx, gy, coh, curv = build_sdf(mask)
    lat = Lattice(nx)
    # reference settle at full attraction, used only to decide which side of
    # the contour each cell ends up on
    P_ref, _, _ = settle(lat, sdf, gx, gy, cfg, iters=iters_for(nx),
                         wobble=0.0, coh=coh, curv=curv)
    frames = clip_frames()
    fam, lum = luma_split(frames)
    pal = CellPalette(lat, P_ref, sdf, fam, np.random.default_rng(seed),
                      idle_mode=idle_mode, order=order)
    sol = Solver(lat, sdf, gx, gy, cfg, coh, curv)
    return dict(cfg=cfg, lat=lat, sdf=sdf, sol=sol, pal=pal, fam=fam,
                lum=lum, frames=frames, P_ref=P_ref, info=info)


def run_hybrid(word="Yope3D", nx=64, wob=0.30, sec=LOOP_SEC, tag="hybrid",
               idle_mode="jumble", video=True, strip=True, ss=2, seed=11,
               shading=False, shade_scale=1.0, preroll=2):
    S = _hybrid_setup(word, nx, wob, idle_mode=idle_mode, seed=seed)
    lat, sol, pal, cfg = S["lat"], S["sol"], S["pal"], S["cfg"]
    log("[hybrid] clip mean luma %s -> inside=%s outside=%s" %
        ([round(v, 1) for v in S["lum"]], S["fam"]["inside"],
         S["fam"]["outside"]))
    nfr = int(sec * FPS)
    seqs = [loop_seq(len(f), nfr) for f in S["frames"]]
    P = lat.P.copy()
    for _ in range(240):
        P, _ = sol.step(P, 0.0, wobble_home(lat, 0.0, wob, sec), cfg["dt"])
    # pre-roll one whole cycle with the solver only (no rendering).  The
    # solver is a continuous relaxation, so the state entering frame 0 has to
    # be the state leaving frame 359 or the loop visibly jumps; one silent
    # cycle costs ~15 s and takes the wrap error from 10 px to well under one.
    for _ in range(preroll):
        for f in range(nfr):
            t = f / float(FPS)
            a = cycle_alpha(t, sec)
            home = wobble_home(lat, t, wob, sec)
            for _ in range(cfg["frame_iters"]):
                P, _ = sol.step(P, a, home, cfg["dt"])
    proc = None
    if video:
        proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
             "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "17", os.path.join(OUT, "%s-cycle.mp4" % tag)],
            stdin=subprocess.PIPE)
    want_t = [0.0, 1.6, 2.2, 2.8, 3.5, 4.6, 7.0, 8.6, 9.6, 11.0]
    want_f = {int(round(v * FPS)) % nfr: v for v in want_t}
    shots, stats = {}, []
    t0all = time.time()
    sx = sy = None
    P0 = None
    for f in range(nfr):
        t = f / float(FPS)
        a = cycle_alpha(t, sec)
        home = wobble_home(lat, t, wob, sec)
        for _ in range(cfg["frame_iters"]):
            P, _ = sol.step(P, a, home, cfg["dt"])
        if f == 0:
            P0 = P.copy()
        sx, sy = inverse_map(lat, P, newton=(14 if f == 0 else 4),
                             warm=None if f == 0 else (sx, sy))
        srcs = [np.asarray(Image.open(S["frames"][k][seqs[k][f]])
                           .convert("RGB")) for k in range(NCLIP)]
        ids = pal.ids(a)
        st = (shade_scale * flare_strength(a) * release_gate(t, sec)
              if shading else 0.0)
        img = hybrid_render(lat, sx, sy, ids, srcs, ss=ss, shade_strength=st)
        if proc:
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        tq, tt = tri_fold_rate(lat, P)
        stats.append(dict(f=f, t=round(t, 3), alpha=round(a, 4),
                          switched=float((a >= pal.tau).mean()),
                          shade=round(float(st), 4), tri_fold=round(tt, 5)))
        if f in want_f:
            shots[f] = (Image.fromarray(img), a, want_f[f],
                        float((a >= pal.tau).mean()), float(st))
    if proc:
        proc.stdin.close()
        proc.wait()
    cost = (time.time() - t0all) / nfr
    wrap = float(np.sqrt(((P - P0) ** 2).sum(-1)).max())
    if strip:
        tiles = [(shots[f][0],
                  "t=%.1fs   amplitude=%.2f   committed %.0f%%   "
                  "shading strength %.2f%s" %
                  (shots[f][2], shots[f][1], 100 * shots[f][3], shots[f][4],
                   "   <- idle" if shots[f][1] < 0.02 else ""))
                 for f in sorted(shots)]
        caption_grid(
            tiles, 2, 880, cap_h=30,
            title="agent-f  HYBRID formation cycle%s: force-settled lattice "
                  "warps the footage, clip families supply the contrast  "
                  "(%s, %d cells, idle=%s)" %
                  (" + Jacobian shading" if shading else "", word, nx,
                   idle_mode)
        ).save(os.path.join(OUT, "%s-strip.png" % tag))
    json.dump(dict(word=word, nx=nx, idle_mode=idle_mode, ss=ss,
                   shading=shading, shade_scale=shade_scale, preroll=preroll,
                   ms_per_frame=round(cost * 1000, 1),
                   loop_wrap_px=round(wrap, 4),
                   inside=S["fam"]["inside"], outside=S["fam"]["outside"],
                   luma=[round(v, 1) for v in S["lum"]], stats=stats),
              open(os.path.join(OUT, "%s.json" % tag), "w"), indent=1)
    log("[hybrid:%s] %.0f ms/frame, loop wrap %.4f px, %d frames" %
        (tag, cost * 1000, wrap, nfr))
    return S


def run_hybrid_compare(word="Yope3D", nx=64, wob=0.30, seed=11):
    """Q1/Q2: warp+clips vs clips alone, same density, same palette."""
    S = _hybrid_setup(word, nx, wob, seed=seed)
    lat, pal, cfg = S["lat"], S["pal"], S["cfg"]
    srcs = [np.asarray(Image.open(f[len(f) // 3]).convert("RGB"))
            for f in S["frames"]]
    ids = pal.ids(1.0)
    P = S["P_ref"]
    sx, sy = inverse_map(lat, P, newton=14)
    ix, iy = identity_map()
    # clip contrast alone: same palette, but the cells are still square
    a, b, c, d = lat.quad_corners(lat.R)
    cen = (a + b + c + d) / 4.0
    dv = sample(S["sdf"], cen[..., 0], cen[..., 1])
    rng = np.random.default_rng(seed)
    fin = np.array(S["fam"]["inside"])
    fout = np.array(S["fam"]["outside"])
    flat_ids = np.where(dv < 0, fin[rng.integers(0, 2, dv.shape)],
                        fout[rng.integers(0, 2, dv.shape)]).astype(np.int32)
    tiles = [
        (Image.fromarray(hybrid_render(lat, ix, iy, flat_ids, srcs)),
         "A  clip contrast ALONE, square lattice (the round-2 look)\n"
         "seam luma step %.1f" % seam_contrast(lat, lat.R, flat_ids, srcs)),
        (Image.fromarray(hybrid_render(lat, sx, sy, ids, srcs)),
         "B  HYBRID: same palette carried by the force-settled lattice\n"
         "seam luma step %.1f" % seam_contrast(lat, P, ids, srcs)),
        (Image.fromarray(warp_image(srcs[0], sx, sy)),
         "C  pure warp of one clip, same mesh (the negative result)"),
        (wireframe(lat, P), "D  the mesh alone, for reference"),
    ]
    caption_grid(tiles, 2, 880, cap_h=34,
                 title="agent-f  Q1/Q2  does the warp make clip contrast "
                       "better, and do the seams stop looking pasted?  "
                       "(%s, %d cells)" % (word, nx)
                 ).save(os.path.join(OUT, "hybrid-vs-clips.png"))
    log("[hybrid-compare] saved")


def run_hybrid_idle(word="Yope3D", nx=64, wob=0.30, seed=11):
    """Q3: at amplitude 0, jumble of all four clips or a single clip?"""
    tiles = []
    for mode in ("jumble", "single"):
        S = _hybrid_setup(word, nx, wob, idle_mode=mode, seed=seed)
        lat, pal, cfg, sol = S["lat"], S["pal"], S["cfg"], S["sol"]
        srcs = [np.asarray(Image.open(f[len(f) // 3]).convert("RGB"))
                for f in S["frames"]]
        P = lat.P.copy()
        for a in (0.0, 0.35, 0.70, 1.0):
            for _ in range(300):
                P, _ = sol.step(P, a, wobble_home(lat, 0.0, wob), cfg["dt"])
            sx, sy = inverse_map(lat, P, newton=14)
            ids = pal.ids(a)
            tiles.append((Image.fromarray(hybrid_render(lat, sx, sy, ids,
                                                        srcs)),
                          "idle = %s   amplitude %.2f   committed %.0f%%" %
                          (mode, a, 100 * float((a >= pal.tau).mean()))))
            caption_grid(tiles, 4, 470, cap_h=28,
                         title="agent-f  Q3  what the field is at idle: a "
                               "jumble of all four clips, or one clip"
                         ).save(os.path.join(OUT, "hybrid-idle.png"))
        log("[hybrid-idle] %s done" % mode)


def run_hybrid_density(word="Yope3D", densities=(16, 24, 32, 48, 64, 96),
                       wob=0.30, seed=11):
    """Q4: does clip contrast let the lattice go coarser?"""
    tiles = []
    for nx in densities:
        S = _hybrid_setup(word, nx, wob, seed=seed)
        lat, pal = S["lat"], S["pal"]
        srcs = [np.asarray(Image.open(f[len(f) // 3]).convert("RGB"))
                for f in S["frames"]]
        sx, sy = inverse_map(lat, S["P_ref"], newton=14)
        img = hybrid_render(lat, sx, sy, pal.ids(1.0), srcs)
        tq, tt = tri_fold_rate(lat, S["P_ref"])
        tiles.append((Image.fromarray(img),
                      "%d x %d cells   h=%.1f px   %.2f cells per stroke   "
                      "trifold %.1f%%" %
                      (lat.nx_vis, int(round(H / lat.hy)), lat.h,
                       S["info"]["stroke_px"] / lat.h, 100 * tt)))
        log("[hybrid-density] nx=%d h=%.1f" % (nx, lat.h))
        caption_grid(tiles, 2, 880, cap_h=30,
                     title="agent-f  Q4  hybrid at peak amplitude vs lattice "
                           "density (%s)" % word
                     ).save(os.path.join(OUT, "hybrid-density.png"))


def run_shading_grid(word="Yope3D", nx=64, wob=0.30, seed=11,
                     levels=(0.0, 0.35, 0.55, 0.75, 1.10)):
    """Deliverable 1: hybrid at the hold with shading off, low and high."""
    S = _hybrid_setup(word, nx, wob, seed=seed)
    lat, pal = S["lat"], S["pal"]
    srcs = [np.asarray(Image.open(f[len(f) // 3]).convert("RGB"))
            for f in S["frames"]]
    sx, sy = inverse_map(lat, S["P_ref"], newton=14)
    ids = pal.ids(1.0)
    tiles = []
    for st in levels:
        img = hybrid_render(lat, sx, sy, ids, srcs, ss=2, shade_strength=st)
        lab = ("shading OFF (the hybrid as delivered)" if st == 0 else
               "shading strength %.2f%s" %
               (st, {0.35: "   (bottom of G's usable window)",
                     0.55: "   (G's hold default)",
                     0.75: "   (top of G's usable window)",
                     1.10: "   (above the window: embossed sticker)"}
                .get(st, "")))
        tiles.append((Image.fromarray(img), lab))
        log("[shade-grid] %.2f" % st)
        caption_grid(tiles, 2, 880, cap_h=30,
                     title="agent-f  Jacobian shading inside the hybrid, at "
                           "the hold (amplitude 1.0, %s, %d cells, same seed)"
                           % (word, nx)
                     ).save(os.path.join(OUT, "hybrid-shading-grid.png"))


def run_look_compare(word="Yope3D", nx=64, wob=0.30, seed=11, st=0.9):
    """Deliverable 4: my traced-outline look vs a filled-deboss look."""
    S = _hybrid_setup(word, nx, wob, seed=seed)
    lat, pal, cfg = S["lat"], S["pal"], S["cfg"]
    mask, info = render_word_mask(word)
    sdf, gx, gy, coh, curv = build_sdf(mask)
    srcs = [np.asarray(Image.open(f[len(f) // 3]).convert("RGB"))
            for f in S["frames"]]
    one = srcs[0]
    sx, sy = inverse_map(lat, S["P_ref"], newton=14)
    tiles = []

    def shaded(mx, my, base):
        shv = jacobian_shade(mx, my, 1, st)
        return Image.fromarray(np.clip(base.astype(np.float32) *
                                       shv[..., None], 0, 255).astype(np.uint8))

    tiles.append((shaded(sx, sy, warp_image(one, sx, sy)),
                  "A  MINE: force attraction piles vertices onto the contour, "
                  "so compression is a thin band.\nOne clip + shading = a dark "
                  "traced outline.  An engraving."))
    fx, fy = smooth_funnel_map(sdf, gx, gy, 0.55, 0.60 * info["stroke_px"])
    tiles.append((shaded(fx, fy, warp_image(one, fx, fy)),
                  "B  EMULATED filled-deboss look (a single smooth funnel "
                  "across the whole interior, not agent G's code).\nOne clip + "
                  "the same shading = filled letters with a bright rim."))
    # a wide capture band pushes my algorithm toward the filled look
    cfg2 = dict(DEFAULT_CFG)
    cfg2["sigma"] = 1.8
    lat2 = Lattice(nx)
    P2, _, _ = settle(lat2, sdf, gx, gy, cfg2, iters=iters_for(nx), wobble=wob,
                      coh=coh, curv=curv)
    wx, wy = inverse_map(lat2, P2, newton=14)
    tiles.append((shaded(wx, wy, warp_image(one, wx, wy)),
                  "C  MINE with the capture band widened to 1.8 cells: "
                  "compression spreads inward,\nthe look moves toward B, and "
                  "the surrounding field shreds (19%% triangle folds)."))
    tiles.append((Image.fromarray(
        hybrid_render(lat, sx, sy, pal.ids(1.0), srcs, ss=2,
                      shade_strength=0.55)),
        "D  MINE, full hybrid: clip contrast + the traced outline together."))
    tiles.append((wireframe(lat, S["P_ref"]),
                  "E  the mesh behind A and D: the contour is a chain of "
                  "lattice edges, which is why\nthe compression is a band and "
                  "not a basin."))
    caption_grid(tiles, 2, 880, cap_h=38,
                 title="agent-f  look comparison: traced outline (attraction "
                       "to a contour) vs filled deboss (a smooth funnel)"
                 ).save(os.path.join(OUT, "look-compare.png"))
    log("[look-compare] saved")


# ----------------------------------------------------------------------------
# 11.  FIXED PALETTE  --  the palette never changes; only geometry animates
#
# Clip assignment is drawn once, at random, from all four clips, and is never
# recomputed, never committed and never partitioned.  Idle and hold use the
# identical palette; the only difference between them is that the force field
# has been applied.  The contour reads because it is a chain of cell edges, and
# with four clips drawn at random about three quarters of neighbouring pairs
# already differ, so most of the contour carries a clip change for free.
#
# DEAD FOR THIS PATH (kept only for the earlier committing variant):
#   CellPalette.tau / .fam_clip / .idle_clip, the staggered commit, the
#   luma_split partition, and the reference-mesh family decision in
#   _hybrid_setup.  None of them are consulted below.
# ----------------------------------------------------------------------------

def fixed_palette(lat, rng, nclip=NCLIP):
    """One draw, once, uniform over every clip.  That is the whole palette."""
    return rng.integers(0, nclip, (lat.ny, lat.nx)).astype(np.int32)


def contour_clip_fraction(lat, P, sdf, ids):
    """What fraction of the contour LENGTH carries a clip change across it.

    A cell edge is on the contour when its two cells land on opposite sides of
    the zero crossing.  Weighted by the deformed length of that edge, so this
    is the fraction of the drawn outline that is actually visible, not a count
    of cells."""
    a, b, c, d = lat.quad_corners(P)
    cen = (a + b + c + d) / 4.0
    s = sample(sdf, cen[..., 0], cen[..., 1]) < 0

    # cells (j,i)|(j,i+1) share the vertical mesh edge at vertex column i+1
    mh = s[:, :-1] != s[:, 1:]
    dh = ids[:, :-1] != ids[:, 1:]
    lh = np.linalg.norm(P[1:, 1:-1] - P[:-1, 1:-1], axis=-1)
    # cells (j,i)|(j+1,i) share the horizontal mesh edge at vertex row j+1
    mv = s[:-1, :] != s[1:, :]
    dv_ = ids[:-1, :] != ids[1:, :]
    lv = np.linalg.norm(P[1:-1, 1:] - P[1:-1, :-1], axis=-1)

    tot = lh[mh].sum() + lv[mv].sum()
    hit = lh[mh & dh].sum() + lv[mv & dv_].sum()
    ncnt = int(mh.sum() + mv.sum())
    nhit = int((mh & dh).sum() + (mv & dv_).sum())
    return dict(length_fraction=float(hit / max(tot, 1e-9)),
                edge_fraction=float(nhit / max(ncnt, 1)),
                contour_edges=ncnt, contour_px=float(tot))


def spread_cells(cfg):
    """Elastic decay length of the deformation, in cells.

    The home spring is what localises the response: a spring network with
    stiffness k_spring pinned by a per-vertex spring k_home relaxes with a
    characteristic length of sqrt(k_spring / k_home) cells."""
    return math.sqrt(cfg["k_spring"] / max(cfg["k_home"], 1e-9))


def _fixed_setup(word, nx, cfg=None, seed=11, wob=0.30, single=False):
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    mask, info = render_word_mask(word)
    sdf, gx, gy, coh, curv = build_sdf(mask)
    lat = Lattice(nx)
    sol = Solver(lat, sdf, gx, gy, cfg, coh, curv)
    frames = clip_frames()
    rng = np.random.default_rng(seed)
    ids = (np.zeros((lat.ny, lat.nx), np.int32) if single
           else fixed_palette(lat, rng))
    return dict(cfg=cfg, lat=lat, sdf=sdf, sol=sol, ids=ids, frames=frames,
                info=info, coh=coh, curv=curv, gx=gx, gy=gy)


def _settle_to(S, alpha, wob=0.30, iters=None, P=None):
    lat, sol, cfg = S["lat"], S["sol"], S["cfg"]
    P = lat.P.copy() if P is None else P
    n = iters or iters_for(lat.nx_vis)
    home = wobble_home(lat, 0.0, wob)
    for i in range(n):
        a = alpha * min(1.0, i / (0.55 * n))
        P, _ = sol.step(P, a, home, cfg["dt"])
    return P


def run_fixed_variants(word="Yope3D", nx=64, wob=0.30, seed=11, gain=0.9):
    """Deliverable: the three variants at the hold, fixed palette throughout."""
    Sj = _fixed_setup(word, nx, seed=seed)
    Ss = _fixed_setup(word, nx, seed=seed, single=True)
    srcs = [np.asarray(Image.open(f[len(f) // 3]).convert("RGB"))
            for f in Sj["frames"]]
    P = _settle_to(Sj, 1.0, wob)
    sx, sy = inverse_map(Sj["lat"], P, newton=14)
    lat = Sj["lat"]
    cc = contour_clip_fraction(lat, P, Sj["sdf"], Sj["ids"])
    log("[fixed] contour clip fraction %s" % cc)
    idle = hybrid_render(lat, *identity_map(), Sj["ids"], srcs, ss=2)
    tiles = [
        (Image.fromarray(idle),
         "IDLE, amplitude 0: the fixed jumble, undeformed.\n"
         "The hold below uses this EXACT palette - only the geometry moves."),
        (Image.fromarray(hybrid_render(lat, sx, sy, Sj["ids"], srcs, ss=2)),
         "1  fixed jumble + warp, NO shading.\n"
         "%.0f%% of the contour length carries a clip change (dashed outline)."
         % (100 * cc["length_fraction"])),
        (Image.fromarray(hybrid_render(lat, sx, sy, Sj["ids"], srcs, ss=2,
                                       shade_strength=gain)),
         "2  fixed jumble + warp + Jacobian outline, gain %.2f.\n"
         "The engraving fills the gaps in the dashed contour." % gain),
        (Image.fromarray(hybrid_render(lat, sx, sy, Ss["ids"], srcs, ss=2,
                                       shade_strength=gain)),
         "3  single clip + warp + outline, gain %.2f.\n"
         "The warp-hold-yope look, with the outline added." % gain),
    ]
    caption_grid(tiles, 2, 880, cap_h=34,
                 title="agent-f  FIXED PALETTE: the palette never changes, "
                       "only the geometry animates  (%s, %d cells)"
                       % (word, nx)
                 ).save(os.path.join(OUT, "fixed-variants.png"))
    json.dump(cc, open(os.path.join(OUT, "contour-contrast.json"), "w"),
              indent=1)


def run_outline_sweep(word="Yope3D", nx=64, wob=0.30, seed=11,
                      gains=(0.0, 0.45, 0.9, 1.35, 1.8)):
    for single in (False, True):
        S = _fixed_setup(word, nx, seed=seed, single=single)
        srcs = [np.asarray(Image.open(f[len(f) // 3]).convert("RGB"))
                for f in S["frames"]]
        P = _settle_to(S, 1.0, wob)
        sx, sy = inverse_map(S["lat"], P, newton=14)
        tiles = []
        for g in gains:
            tiles.append((Image.fromarray(
                hybrid_render(S["lat"], sx, sy, S["ids"], srcs, ss=2,
                              shade_strength=g)),
                "outline gain %.2f%s" % (g, "   (off)" if g == 0 else "")))
            log("[outline-sweep] single=%s gain=%.2f" % (single, g))
        caption_grid(tiles, 2, 880, cap_h=30,
                     title="agent-f  Jacobian outline gain sweep, fixed "
                           "palette, at the hold  (%s, %d cells, %s)" %
                           (word, nx,
                            "single clip" if single else "four-clip jumble")
                     ).save(os.path.join(
                         OUT, "fixed-outline-sweep%s.png" %
                         ("-single" if single else "")))


def contour_visibility(img, sdf, band=7.0, far=70.0):
    """The word now reads ONLY from the contour, so measure the contour.

    Ratio of mean luminance-gradient magnitude in a narrow band on the zero
    crossing to the mean well away from it.  1.0 means the outline is no more
    visible than the surrounding mosaic."""
    L = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] +
         0.0722 * img[..., 2]).astype(np.float32)
    g = np.hypot(ndimage.sobel(L, 0), ndimage.sobel(L, 1))
    d = np.abs(sdf[PAD:PAD + H, PAD:PAD + W])
    near = d < band
    away = d > far
    return float(g[near].mean() / max(g[away].mean(), 1e-6))


def run_spread_sweep(word="Yope3D", nx=64, wob=0.30, seed=11, gain=0.9,
                     pairs=((0.012, 6.0), (0.06, 8.0), (0.15, 11.0),
                           (0.35, 16.0), (0.35, 24.0))):
    """How far the distortion fans out into the field, and what tightening it
    costs in legibility."""
    tiles, rows = [], []
    glyph, _ = render_word_mask(word)
    gin = glyph[PAD:PAD + H, PAD:PAD + W]
    for kh, ka in pairs:
        cfg = dict(DEFAULT_CFG)
        cfg["k_home"] = kh
        cfg["k_att"] = ka
        S = _fixed_setup(word, nx, cfg=cfg, seed=seed)
        srcs = [np.asarray(Image.open(f[len(f) // 3]).convert("RGB"))
                for f in S["frames"]]
        P = _settle_to(S, 1.0, wob)
        lat = S["lat"]
        sx, sy = inverse_map(lat, P, newton=14)
        img = hybrid_render(lat, sx, sy, S["ids"], srcs, ss=2,
                            shade_strength=gain)
        vis = contour_visibility(img, S["sdf"])
        vis0 = contour_visibility(
            hybrid_render(lat, sx, sy, S["ids"], srcs, ss=2), S["sdf"])
        # how far the displacement actually reaches, measured on the mesh
        disp = np.sqrt(((P - wobble_home(lat, 0.0, wob)) ** 2).sum(-1))
        dv = sample(S["sdf"], lat.R[..., 0], lat.R[..., 1])
        far = disp[np.abs(dv) > 4 * lat.h].mean()
        near = disp[np.abs(dv) < 1.5 * lat.h].mean()
        cc = contour_clip_fraction(lat, P, S["sdf"], S["ids"])
        tq, tt = tri_fold_rate(lat, P)
        rows.append(dict(k_home=kh, k_att=ka,
                         decay_cells=round(spread_cells(cfg), 2),
                         mean_disp_near_px=round(float(near), 2),
                         mean_disp_far_px=round(float(far), 2),
                         far_over_near=round(float(far / max(near, 1e-6)), 3),
                         contour_vis_outline=round(vis, 3),
                         contour_vis_no_outline=round(vis0, 3),
                         tri_fold=round(tt, 4),
                         contour_frac=round(cc["length_fraction"], 3)))
        tiles.append((Image.fromarray(img),
                      "k_home %.3f  k_att %.0f  decay %.1f cells   field "
                      "beyond 4 cells moves %.1f px = %.0f%% of the contour's "
                      "own movement\ncontour visibility %.2fx surroundings "
                      "(%.2fx without the outline)   triangle folds %.1f%%" %
                      (kh, ka, spread_cells(cfg), far,
                       100 * far / max(near, 1e-6), vis, vis0, 100 * tt)))
        log("[spread] %s" % rows[-1])
        caption_grid(tiles, 1, 1000, cap_h=34,
                     title="agent-f  deformation spread sweep: how far the "
                           "distortion fans out past the word  (%s, %d cells, "
                           "fixed palette, outline %.2f)" % (word, nx, gain)
                     ).save(os.path.join(OUT, "fixed-spread-sweep.png"))
        json.dump(rows, open(os.path.join(OUT, "spread.json"), "w"), indent=1)


def run_fixed_cycle(word="Yope3D", nx=64, wob=0.30, sec=LOOP_SEC,
                    tag="fixed", seed=11, single=False, gain=0.9,
                    k_home=None, video=True, preroll=2, ss=2):
    """Formation strip + exact-loop video, fixed palette, only geometry moves."""
    cfg = dict(DEFAULT_CFG)
    if k_home is not None:
        cfg["k_home"] = k_home
    S = _fixed_setup(word, nx, cfg=cfg, seed=seed, single=single)
    lat, sol, ids = S["lat"], S["sol"], S["ids"]
    nfr = int(sec * FPS)
    seqs = [loop_seq(len(f), nfr) for f in S["frames"]]
    P = lat.P.copy()
    for _ in range(240):
        P, _ = sol.step(P, 0.0, wobble_home(lat, 0.0, wob, sec), cfg["dt"])
    for _ in range(preroll):
        for f in range(nfr):
            t = f / float(FPS)
            for _ in range(cfg["frame_iters"]):
                P, _ = sol.step(P, cycle_alpha(t, sec),
                                wobble_home(lat, t, wob, sec), cfg["dt"])
    proc = None
    if video:
        proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
             "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "17", os.path.join(OUT, "%s-cycle.mp4" % tag)],
            stdin=subprocess.PIPE)
    want = {int(round(v * FPS)) % nfr: v
            for v in [0.0, 1.6, 2.2, 2.8, 3.5, 4.6, 7.0, 8.6, 9.6, 11.0]}
    shots = {}
    sx = sy = None
    P0 = None
    t0all = time.time()
    for f in range(nfr):
        t = f / float(FPS)
        a = cycle_alpha(t, sec)
        home = wobble_home(lat, t, wob, sec)
        for _ in range(cfg["frame_iters"]):
            P, _ = sol.step(P, a, home, cfg["dt"])
        if f == 0:
            P0 = P.copy()
        sx, sy = inverse_map(lat, P, newton=(14 if f == 0 else 4),
                             warm=None if f == 0 else (sx, sy))
        srcs = [np.asarray(Image.open(S["frames"][k][seqs[k][f]])
                           .convert("RGB")) for k in range(NCLIP)]
        st = gain / FLARE_HOLD * flare_strength(a) * release_gate(t, sec) \
            if gain > 0 else 0.0
        img = hybrid_render(lat, sx, sy, ids, srcs, ss=ss, shade_strength=st)
        if proc:
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        if f in want:
            shots[f] = (Image.fromarray(img), a, want[f], st)
    if proc:
        proc.stdin.close()
        proc.wait()
    wrap = float(np.sqrt(((P - P0) ** 2).sum(-1)).max())
    cost = (time.time() - t0all) / nfr
    tiles = [(shots[f][0], "t=%.1fs   amplitude %.2f   outline %.2f   "
                           "(palette identical in every frame)" %
              (shots[f][2], shots[f][1], shots[f][3]))
             for f in sorted(shots)]
    caption_grid(tiles, 2, 880, cap_h=30,
                 title="agent-f  FIXED-PALETTE formation cycle: nothing but "
                       "the geometry changes  (%s, %d cells, %s, outline "
                       "%.2f, decay %.1f cells)" %
                       (word, nx, "single clip" if single else "four-clip "
                        "jumble", gain, spread_cells(cfg))
                 ).save(os.path.join(OUT, "%s-strip.png" % tag))
    log("[fixed-cycle:%s] %.0f ms/frame  loop wrap %.3f px" %
        (tag, cost * 1000, wrap))
    return wrap


def main():
    t0 = time.time()
    todo = set(sys.argv[1:]) or {"all"}

    def want(k):
        return "all" in todo or k in todo

    if want("wire"):
        r = mk("Yope3D", 64, iters=1176)
        bad, inv = r["lat"].fold_stats(r["P"])
        tq, tt = tri_fold_rate(r["lat"], r["P"])
        log("[wire] %s cell=%.1f nonconvex=%.4f inverted=%.4f trifold=%.5f" %
            (r["info"], r["lat"].h, bad, inv, tt))
        wireframe(r["lat"], r["P"]).save(os.path.join(OUT, "wireframe.png"))
    if want("density"):
        run_density()
    if want("formation"):
        run_formation()
    if want("wobble"):
        run_wobble()
    if want("ladder"):
        run_alpha_ladder()
    if want("jac"):
        run_jacobian_probe()
    if want("compare"):
        run_warp_compare()
    if want("warp"):
        run_warp(clip="yope", tag="yope")
    if want("warp2"):
        run_warp(clip="spin", tag="spin")
    if want("hyb-compare"):
        run_hybrid_compare()
    if want("hyb-idle"):
        run_hybrid_idle()
    if want("hyb-density"):
        run_hybrid_density()
    if want("hyb"):
        run_hybrid(nx=64, tag="hybrid")
    if want("hyb96"):
        run_hybrid(nx=96, tag="hybrid96", video=True)
    if want("fixed"):
        run_fixed_variants()
    if want("osweep"):
        run_outline_sweep()
    if want("spread"):
        run_spread_sweep()
    if want("fixed-cycle"):
        run_fixed_cycle()
    if want("shade-grid"):
        run_shading_grid()
    if want("look"):
        run_look_compare()
    if want("hyb-shade"):
        run_hybrid(nx=64, tag="hybrid-shading", shading=True, video=True)
    if want("hyb-single"):
        run_hybrid(nx=64, tag="hybrid-single", idle_mode="single",
                   video=False)
    if want("warpwob"):
        run_warp(clip="yope", tag="yope-wob0", wob=0.0, video=False)
    log("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
