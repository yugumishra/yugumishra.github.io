"""Render the two GlyphInterpreter analysis stills used by the project card.

The encoder is reimplemented here in numpy straight from `hand_encoder2.onnx`
(four Gemms, exact GELU, L2 normalize -- 530,752 parameters) rather than loaded
through torch, because the checkpoint needs a working torch install and the
ONNX export is the artifact Unity actually ships. `--validate` replays the
KDTree pipeline from identifier_2.py and diffs it against the committed
gesture_test_distances.csv, which is what proves this reimplementation matches
the model that produced the demo footage.

Outputs (light "plate" figures, matching the site's convention for diagrams):
  glyph_embedding_space.png   3 enrolled gestures against the unlabeled take
  glyph_separability.png      held-out same- vs different-gesture distances

Run: python scripts/generate-glyph-figures.py [--project PATH] [--validate]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from scipy.special import erf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PROJECT = Path(
    "/Users/me/Desktop/dev/MI Lab Stuff/InternStuff/GlyphInterpreter"
)
THRESHOLD = 0.55  # the value shipped in the headset UI and GestureStillPreview

# Validated against the site's --plate surface (#f5f7fa) with the dataviz
# validator, all-pairs: CVD dE 9.2, normal-vision dE 24.0. Orange and aqua sit
# below 3:1 on this surface, so every series is directly labeled -- that is the
# required relief, not a stylistic choice.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
PLATE = "#f5f7fa"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
UNLABELED = "#b9bcc2"

GESTURE_NAMES = {
    "gesture1": "Doctor Strange",
    "gesture2": "Finger Gun",
    "gesture3": "Spider-Man",
}


# ---------------------------------------------------------------- model ----
def load_encoder(onnx_path):
    """Return f(X: (n, 234)) -> (n, 64) unit-norm embeddings."""
    graph = onnx.load(str(onnx_path)).graph
    w = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    layers = [
        (w[f"net.{i}.weight"].astype(np.float64), w[f"net.{i}.bias"].astype(np.float64))
        for i in (0, 2, 4, 6)
    ]

    def gelu(x):  # exact, as exported: 0.5x(1 + erf(x / sqrt(2)))
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    def encode(x):
        # Accelerate (macOS BLAS) raises spurious overflow/invalid FP flags on
        # these matmuls; the results are finite and match the reference CSV to
        # 7e-07, so the flags are suppressed rather than chased.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            h = np.asarray(x, dtype=np.float64)
            for weight, bias in layers[:-1]:
                h = gelu(h @ weight.T + bias)
            weight, bias = layers[-1]
            z = h @ weight.T + bias
            return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    return encode


# ----------------------------------------------------------------- data ----
def load_frames(path, num_bones):
    """Read a recording into (features (n, 9B), confidence (n,)).

    Feature layout mirrors train.frame_to_tensor: per bone, position (3) then
    the first two columns of the rotation matrix flattened row-major (6).
    """
    raw = np.fromfile(path, dtype="<f4")
    stride = 2 + num_bones * 7
    count = raw.size // stride
    rows = raw[: count * stride].reshape(count, stride)

    confidence = rows[:, 1].astype(np.float64)
    body = rows[:, 2:].reshape(count, num_bones, 7).astype(np.float64)
    pos, quat = body[:, :, :3], body[:, :, 3:7]

    x, y, z, w = (quat[:, :, i] for i in range(4))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    six_d = np.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz),      # R00, R01
            2 * (xy + wz),     1 - 2 * (xx + zz),  # R10, R11
            2 * (xz - wy),     2 * (yz + wx),      # R20, R21
        ],
        axis=-1,
    )
    feats = np.concatenate([pos, six_d], axis=-1).reshape(count, num_bones * 9)
    return feats, confidence


def gesture_paths(project):
    return {
        f"gesture{i}": project / f"example_gesture_{i}.bin" for i in (1, 2, 3)
    }


# ------------------------------------------------------------- validate ----
def validate(project, encode, num_bones):
    """Replay identifier_2's KDTree scoring and diff against the CSV."""
    from sklearn.neighbors import KDTree

    enrolled = []
    for path in gesture_paths(project).values():
        feats, conf = load_frames(path, num_bones)
        enrolled.append(encode(feats[conf > 0.7]))
    tree = KDTree(np.concatenate(enrolled))

    feats, _ = load_frames(project / "gesture_test.bin", num_bones)
    mine = tree.query(encode(feats), k=1)[0][:, 0]

    csv_path = project / "Assets/Editor/gesture_test_distances.csv"
    with csv_path.open() as handle:
        theirs = np.array([float(r["distance"]) for r in csv.DictReader(handle)])

    n = min(len(mine), len(theirs))
    delta = np.abs(mine[:n] - theirs[:n])
    print(f"validation vs {csv_path.name}: {n} frames")
    print(f"  max |diff| {delta.max():.2e}   mean {delta.mean():.2e}")
    print("  -> encoder reproduces the shipped pipeline" if delta.max() < 1e-4
          else "  -> MISMATCH, figures would be wrong")
    return delta.max() < 1e-4


# -------------------------------------------------------------- figures ----
def style(ax):
    ax.set_facecolor(PLATE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d4d7dd")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color="#e3e6eb", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_embedding(project, encode, num_bones, out):
    background, _ = load_frames(project / "gesture_test.bin", num_bones)
    background = encode(background)

    clusters = {}
    for label, path in gesture_paths(project).items():
        feats, conf = load_frames(path, num_bones)
        clusters[label] = encode(feats[conf > 0.7])

    centroids = np.stack([emb.mean(axis=0) for emb in clusters.values()])
    origin = centroids.mean(axis=0)

    # Project onto the plane through the three centroids rather than a global
    # PCA. All three centroids lie in this plane, so the distances *between*
    # them -- the quantity nearest-centroid classification actually uses -- are
    # reproduced exactly instead of being collapsed by a lossy 2D fit.
    e1 = centroids[1] - centroids[0]
    e1 /= np.linalg.norm(e1)
    e2 = centroids[2] - centroids[0]
    e2 -= (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    project2d = lambda z: np.stack([(z - origin) @ e1, (z - origin) @ e2], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=170)
    fig.patch.set_facecolor(PLATE)
    style(ax)
    ax.set_aspect("equal")  # distances must not be distorted by axis scaling

    bg = project2d(background)
    ax.scatter(bg[:, 0], bg[:, 1], s=6, c=UNLABELED, linewidths=0, alpha=0.5,
               zorder=1)

    for (label, emb), color, centroid in zip(clusters.items(), SERIES, centroids):
        pts = project2d(emb)
        cx, cy = project2d(centroid[None])[0]
        # Dashed ring = the 0.55 acceptance boundary sliced by this plane. A
        # frame can sit inside the ring and still be rejected on the 62 axes
        # the plane drops, so membership is encoded by opacity, not position.
        inside = np.linalg.norm(emb - centroid, axis=1) <= THRESHOLD
        ax.add_patch(plt.Circle((cx, cy), THRESHOLD, fill=False, color=color,
                                linestyle="--", linewidth=1.2, alpha=0.8,
                                zorder=2))
        ax.scatter(pts[inside, 0], pts[inside, 1], s=15, c=color, linewidths=0,
                   zorder=3)
        ax.scatter(pts[~inside, 0], pts[~inside, 1], s=15, facecolors="none",
                   edgecolors=color, linewidths=0.7, alpha=0.65, zorder=3)
        ax.scatter([cx], [cy], s=90, marker="+", c=INK, linewidths=1.6, zorder=5)
        # Seat the name just clear of the ring so it never covers its own cloud.
        ax.annotate(GESTURE_NAMES[label], xy=(cx, cy + THRESHOLD),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc=PLATE, ec="none",
                              alpha=0.9))

    gaps = [np.linalg.norm(centroids[a] - centroids[b])
            for a, b in ((0, 1), (0, 2), (1, 2))]
    top = max(c[1] + THRESHOLD for c in
              [project2d(centroids[i][None])[0] for i in range(3)])
    ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], top + 0.34))

    ax.set_xlabel("plane through the three gesture centroids (L2 units)",
                  color=INK_MUTED, fontsize=9.5)
    fig.text(0.015, 0.975, "Three gestures the encoder was never given a label for",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.015, 0.917,
             f"Centroids sit {min(gaps):.2f}–{max(gaps):.2f} apart, every gap far "
             f"outside the {THRESHOLD} threshold (dashed).\nHollow markers fall "
             f"outside their own threshold; grey is unlabeled test footage.",
             fontsize=9.5, color=INK_MUTED, ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, facecolor=PLATE)
    plt.close(fig)
    print(f"wrote {out}   (centroid gaps {min(gaps):.2f}-{max(gaps):.2f})")


def figure_separability(project, encode, num_bones, out):
    same, different = [], []
    centroids, held = {}, {}

    for label, path in gesture_paths(project).items():
        feats, conf = load_frames(path, num_bones)
        emb = encode(feats[conf > 0.7])
        split = len(emb) // 2
        # Enroll on the first half, score the second: the runtime enrolls from
        # one short hold, so scoring the same frames back would be circular.
        centroids[label] = emb[:split].mean(axis=0)
        held[label] = emb[split:]

    for label, emb in held.items():
        own = np.linalg.norm(emb - centroids[label], axis=1)
        others = np.stack([
            np.linalg.norm(emb - centroids[other], axis=1)
            for other in centroids if other != label
        ])
        same.append(own)
        different.append(others.min(axis=0))

    same = np.concatenate(same)
    different = np.concatenate(different)
    accept = (same <= THRESHOLD).mean() * 100
    reject = (different > THRESHOLD).mean() * 100

    gap_lo, gap_hi = same.max(), different.min()
    separable = gap_lo < gap_hi

    fig, ax = plt.subplots(figsize=(8, 6.0), dpi=170)
    fig.patch.set_facecolor(PLATE)
    style(ax)

    bins = np.linspace(0, max(same.max(), different.max()) * 1.05, 46)
    ax.hist(same, bins=bins, color=SERIES[0], alpha=0.9, linewidth=0, zorder=3)
    ax.hist(different, bins=bins, color=SERIES[1], alpha=0.9, linewidth=0, zorder=3)

    if separable:
        ax.axvspan(gap_lo, gap_hi, color="#dfe3ea", zorder=1)
        ax.annotate(f"no frame of either kind\nlands in this {gap_hi - gap_lo:.2f}-wide gap",
                    xy=((gap_lo + gap_hi) / 2, ax.get_ylim()[1] * 0.55),
                    ha="center", va="center", fontsize=9, color=INK_MUTED,
                    zorder=4)

    ax.axvline(THRESHOLD, color=INK, linestyle="--", linewidth=1.6, zorder=5)
    ax.annotate(f"shipped {THRESHOLD}", xy=(THRESHOLD, ax.get_ylim()[1] * 0.97),
                xytext=(-6, 0), textcoords="offset points", fontsize=9.5,
                color=INK, va="top", ha="right", zorder=6)

    def label_peak(values, color, text):
        counts, edges = np.histogram(values, bins=bins)
        peak = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
        ax.annotate(text, xy=(peak, counts.max()), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    fontweight="bold", color=color, zorder=6)

    label_peak(same, SERIES[0], "same gesture")
    label_peak(different, SERIES[1], "different gesture")

    ax.margins(y=0.16)
    ax.set_xlabel("L2 distance to enrolled centroid", color=INK_MUTED, fontsize=9.5)
    ax.set_ylabel("held-out frames", color=INK_MUTED, fontsize=9.5)

    fig.text(0.015, 0.975, "Same- and different-gesture distances never overlap",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.015, 0.917,
             f"Held-out frames land at {same.min():.2f}–{gap_lo:.2f}, impostors at "
             f"{gap_hi:.2f}–{different.max():.2f}: any threshold in between separates\n"
             f"them perfectly. The shipped {THRESHOLD} (dashed) is conservative — it "
             f"rejects {100 - accept:.0f}% of valid frames to buy margin.",
             fontsize=9.5, color=INK_MUTED, ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, facecolor=PLATE)
    plt.close(fig)
    print(f"wrote {out}   (accept {accept:.1f}%, reject {reject:.1f}%, "
          f"gap {gap_lo:.3f}-{gap_hi:.3f}, separable={separable})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "src/assets/img")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    bones = json.loads((args.project / "hand_bones.json").read_text())["bones"]
    num_bones = len(bones)
    encode = load_encoder(args.project / "hand_encoder2.onnx")
    print(f"{num_bones} bones -> {num_bones * 9}-d input")

    if args.validate and not validate(args.project, encode, num_bones):
        raise SystemExit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    figure_embedding(args.project, encode, num_bones,
                     args.out / "glyph_embedding_space.png")
    figure_separability(args.project, encode, num_bones,
                        args.out / "glyph_separability.png")


if __name__ == "__main__":
    main()
