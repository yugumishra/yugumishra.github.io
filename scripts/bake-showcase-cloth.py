"""Bake a showcase project's cloth, which carries the origin-flow frontier.

Run from the repository root, once per project:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/bake-showcase-cloth.py -- --word yope3d
  ... --raw         also save the .blend and render the raw animation for
                    review in agent-work/showcase-cloth-raw/<word>/
  ... --every N     with --raw, render only every Nth frame
  ... --out DIR     write everything to DIR instead, to try settings
  ... --height H --hit H --depth D --friction F --clamp C --end FRAME
                    try other falls
  ... --pour        the cloth wipe's fall instead: the letters are thrown
                    through the floor and keep going, the floor barely holds
                    the cloth, and the sheet pours down through the holes
                    after them. Written beside the flow's bakes, in
                    src/data/showcase-cloth-pour/ and
                    public/media/showcase-test/cloth-pour/, so the flow keeps
                    its own. --seconds S sets how much real time the 145
                    frames show.
  ... --solid       cut the holes and the letters from their outer outlines
                    only, so a letter with a counter (o, p, e, D) is a solid
                    plug over a hole with no floor island inside it for the
                    cloth to drape over and tangle on
  ... --tension T --compress C --shear S --bend B
                    the cloth's stiffnesses (40, 0.5, 4, 0.05)
  ... --air A       air damping (1): higher keeps loose cloth from flinging
                    itself over the letters
  ... --air-slide A with --lift, the damping from the slide on, eased in over
                    a few frames, so the cloth still follows the letters off
  ... --slice       cut the sheet into one piece per letter, along vertical
                    lines halfway between neighbouring letters (and a
                    horizontal one between the lines of a two-line word), so
                    each letter drags only its own strip down its own hole
  ... --bridges     leave a thin strip of floor (--bridge-px, 6) standing on
                    each of those cuts, so letters whose widened holes ran
                    together each keep a hole of their own
  ... --hull        one hole for the whole word, the convex hull of the
                    letters' holes, so the sheet drains through a single
                    opening instead of snagging between letters
  ... --join R --smooth S
                    round the holes (scripts/round-showcase-holes.py, with
                    the system's python3): holes less than 2R px apart run
                    together and concave corners fill in to R px, then every
                    corner rounds off by about S px. Yope3D's o and pe3 are
                    8 px apart, pe3 and D 12, so --join 5 joins just o-pe3.
                    --join auto joins every hole of the word into one.
  ... --tear        a sheet that can tear: cut into irregular patches about
                    --tear-px (60) across, stitched back together by
                    zero-length sewing springs whose force is capped at
                    --tear-force (5). Below the cap it holds as one cloth;
                    pulled harder, a seam gives. Blender's cloth cannot drop
                    a spring mid-bake, so a capped stitch is how it tears.
  ... --tear-seam   one weak seam and nothing else: a two-line word's sheet
                    is cut once, halfway between the lines, and stitched by
                    springs capped at --tear-force, so when the two lines
                    pull against each other it rips there and each half
                    drains down its own line. Elsewhere, and for a one-line
                    word, it is the ordinary cloth. The seam is stitched only
                    across the word, and --seam-reach px (0) past its ends:
                    beyond the letters nothing pulls the halves apart, so
                    stitches there would never break and would pin the
                    halves together at the sheet's edges.
  ... --release D   the letters let go of the cloth once their tops are D
                    units below the floor's top (3.0 is its underside): their
                    collision is keyframed off from that frame, so from then
                    on the cloth drains by its own weight instead of being
                    stretched ever harder by letters still speeding up.
  ... --mouth F     a two-line word's holes linked into one void: a channel
                    between each pair of lines, centred under the shorter
                    line and F of its width across (0.6 is wide), merged with
                    the letter holes and rounded by --smooth
  ... --rise H      the inverse fall: the letters are pillars standing in their
                    holes, tops just under the floor's surface, and rise H
                    units (1.2) up through the cloth resting on it, easing to
                    a stop over --rise-frac (0.7) of the frames, so it drapes
                    over the word and slides in off the floor. Mesh export
                    only, to src/data/showcase-cloth-rise/ and
                    public/media/showcase-test/cloth-rise/.
  ... --lift H      rising letters that then leave: no holes, the letters grow
                    up out of the floor to H units (0.5) under the resting
                    cloth, hold, then slide --slide D units (9) down the
                    screen, dragging the cloth draped on them off the frame.
                    --grip (20) is their friction on the cloth; the floor's
                    stays --friction. --grow / --hold set those phases'
                    shares of the frames (0.35, 0.2); the slide takes the
                    rest. Mesh export only, to showcase-cloth-lift.
  ... --dish D      with --lift, dish each letter's top: a flat rim --dish-rim
                    (0.12 of the stroke) wide at full height along every
                    outline (counters too), then a slanted ring --dish-inset
                    (0.20 of the stroke) in and D units (0.25) down to a
                    lower floor, so the rim draws the letter. The cloth sags into each stroke and
                    pulls tight over the rim. A few hundred faces more.
  ... --simplify P  thin the letters' outlines to within P px (1.0 with --lift)
                    before building them: the traced outlines carry a point
                    every half pixel, thousands a word, and every one of them
                    becomes faces
  ... --dense P     refine the sheet to P px cells (5) over the word's area,
                    the letters' bounds and --dense-pad px (40) round them,
                    leaving --spacing cells elsewhere: the cloth can then
                    follow the letters' rims and dishes, which are smaller
                    than a coarse cell, without the whole sheet paying for it
  ... --preview-letters DIR
                    build the letters, render their wireframe from above and
                    at an angle into DIR, print their size, and stop
  ... --mesh        also export the cloth mesh itself, for the page to draw
                    in 3D: the triangles and each point's rest position (its
                    UV) once, its position every --mesh-every (2) frames, and
                    the floor's hole outlines. Written to
                    src/data/showcase-cloth-mesh/ and
                    public/media/showcase-test/cloth-mesh/ (or --out).
  ... --small       review renders at half size, oblique and top only
  ... --fast        for trying settings: a sheet half as dense (20 px), 8
                    solver steps a frame and collision quality 4, several
                    times quicker. --spacing / --quality / --cq set them
                    one at a time.
The words are the files in scripts/showcase-glyphs/, which
scripts/export-showcase-glyphs.py traces from the page's own glyphs.

A cloth a little larger than the frame rests on a floor. The floor has a
hole the shape of the word, a little wider than the page's glyph. The solid
letters, a little narrower than it, hit the cloth over their holes together,
push it in and drag the rest of it across the floor, where friction gathers
it into pleats. The cloth's edges are free. Nothing is pinned: the letters
and the floor move the cloth only by colliding with it.

The export is the cloth seen from above, frame by frame: on a grid of screen
points, the material point that is visible there, top layer first where the
cloth folds over itself, and its height. The page reads the flood's distance
to the word at that material point, so the frontier rides the fabric.
"""

import json
import os
import sys

import bpy
import mathutils
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
RAW = "--raw" in ARGS
POUR = "--pour" in ARGS
SOLID = "--solid" in ARGS
SLICE = "--slice" in ARGS
TEAR = "--tear" in ARGS
TEAR_SEAM = "--tear-seam" in ARGS
HULL = "--hull" in ARGS
SMALL = "--small" in ARGS
BRIDGES = "--bridges" in ARGS
FAST = "--fast" in ARGS
OUT_DIR = ARGS[ARGS.index("--out") + 1] if "--out" in ARGS else None
EVERY = int(ARGS[ARGS.index("--every") + 1]) if "--every" in ARGS else 1
WORD = ARGS[ARGS.index("--word") + 1] if "--word" in ARGS else "yope3d"


def arg(name, default):
    return float(ARGS[ARGS.index(name) + 1]) if name in ARGS else default


BRIDGE_PX = arg("--bridge-px", 6.0)
TEAR_PX = arg("--tear-px", 60.0)
TEAR_FORCE = arg("--tear-force", 5.0)
SEAM_REACH = arg("--seam-reach", 0.0)
# "--join auto" joins every hole of the word into one: half the widest gap
# from any hole to its nearest neighbour, and a px over.
JOIN_AUTO = "--join" in ARGS and ARGS[ARGS.index("--join") + 1] == "auto"
JOIN = 0.0 if JOIN_AUTO else arg("--join", 0.0)
MOUTH = arg("--mouth", 0.0)
SMOOTH = arg("--smooth", 0.0)


GLYPHS = os.path.join(ROOT, "scripts", "showcase-glyphs", WORD + ".json")
KIND = "showcase-cloth-pour" if POUR else "showcase-cloth"
META = os.path.join(OUT_DIR or os.path.join(ROOT, "src", "data", KIND), WORD + ".json")
DATA = os.path.join(OUT_DIR or os.path.join(ROOT, "public", "media", "showcase-test", "cloth-pour" if POUR else "cloth"), WORD + ".bin")
RAW_DIR = OUT_DIR or os.path.join(ROOT, "agent-work", KIND + "-raw", WORD)
RISE = "--rise" in ARGS
LIFT = "--lift" in ARGS
MESH = "--mesh" in ARGS or RISE or LIFT
MESH_KIND = "lift" if LIFT else "rise" if RISE else "mesh"
MESH_META = os.path.join(OUT_DIR or os.path.join(ROOT, "src", "data", "showcase-cloth-" + MESH_KIND), WORD + ("-mesh.json" if OUT_DIR else ".json"))
MESH_DATA = os.path.join(OUT_DIR or os.path.join(ROOT, "public", "media", "showcase-test", "cloth-" + MESH_KIND), WORD + ("-mesh.bin" if OUT_DIR else ".bin"))
# the floor's holes as cut, canvas px, for the mesh export
HOLE_LOOPS = []

# One Blender unit is 100 canvas px. The canvas is 1280 x 720, centred on
# the origin, with canvas y running down and Blender y up.
PX = 100.0
CANVAS_W, CANVAS_H = 12.8, 7.2

# The cloth overhangs the frame by 1.2 units on every side, so a frame edge
# is still covered after the letters have drawn it in. It starts just above
# the floor, whose top is z = 0.
MARGIN = 2.4 if "--lift" in ARGS else 1.2
SPACING = arg("--spacing", 0.2 if FAST else 0.1)
QUALITY = int(arg("--quality", 8 if FAST else 15))
COLLISION_QUALITY = int(arg("--cq", 4 if FAST else 8))
SHEET_Z = 0.035
FRAME_END = 145
SAMPLE_FRAMES = tuple(range(1, FRAME_END + 1, 6))

# The letters and holes are the glyph's outlines 4 px inside and 8 px outside
# it (see export-showcase-glyphs.py), leaving a 12 px gap round each letter
# for the cloth to line the hole.
LETTER_HALF = 0.12
FLOOR_DEPTH = 3.0
RELEASE = "--release" in ARGS


def release_depth():
    at = ARGS.index("--release") + 1
    try:
        return float(ARGS[at])
    except (IndexError, ValueError):
        return FLOOR_DEPTH


RELEASE_DEPTH = release_depth() if RELEASE else None


def rise_height():
    at = ARGS.index("--rise") + 1
    try:
        return float(ARGS[at])
    except (IndexError, ValueError):
        return 1.2


RISE_HEIGHT = rise_height() if RISE else 0.0


def lift_height():
    at = ARGS.index("--lift") + 1
    try:
        return float(ARGS[at])
    except (IndexError, ValueError):
        return 0.5


LIFT_HEIGHT = lift_height() if LIFT else 0.0
LIFT_SLIDE = arg("--slide", 9.0)
LIFT_GRIP = arg("--grip", 20.0)
LIFT_GROW = arg("--grow", 0.35)
LIFT_HOLD = arg("--hold", 0.2)
DISH = arg("--dish", 0.0)
# The rim and the slope as shares of the letters' stroke (the glyph's, less
# the letters' inset either side), so from both sides of a stroke they stop
# well short of its middle and can never cross.
DISH_RIM = arg("--dish-rim", 0.12)
DISH_INSET = arg("--dish-inset", 0.20)
SIMPLIFY = arg("--simplify", 1.0 if "--lift" in ARGS else 0.0)
DENSE = arg("--dense", 0.0)
DENSE_PAD = arg("--dense-pad", 40.0)
PREVIEW = ARGS[ARGS.index("--preview-letters") + 1] if "--preview-letters" in ARGS else None
# frames where the lift's phases end, for the page
LIFT_PHASES = {}
RISE_FRAC = arg("--rise-frac", 0.7)
# The letters hit the cloth moving as fast as if dropped from HIT_DROP above
# it, then fall on under gravity: they throw the cloth rather than easing it
# along from rest. The animation starts just before the hit, HEIGHT above the
# cloth, since the fall through the air before it moved no cloth and left the
# start of the formation still. A faster hit gives the cloth less time to
# follow, so the fall is deep and the hit modest: at 0.7 deep the cloth
# moved a tenth of the flood's travel, at 1.2 deep and low friction it keeps
# up with it. They land DEPTH below the cloth on the last frame; landing
# earlier let the cloth slump over them afterwards. The cloth runs in slow
# motion so that this drop fills the formation. Because the letters fall at
# gravity's pace, cloth under a hole can never fall faster than the letter
# on it: they push it in, rather than the cloth pouring in ahead of them. A
# deeper drop swallows more cloth and slides the whole sheet off the frame.
FALL_END = int(arg("--end", FRAME_END))
HEIGHT = arg("--height", 0.02)
HIT_DROP = arg("--hit", 2.0 if POUR else 0.75)
DROP = HIT_DROP - HEIGHT
DEPTH = arg("--depth", 1.2)
FRICTION = arg("--friction", 1.0 if POUR else 15.0)
# The pour: the letters hit as if dropped from HIT_DROP and fall on under
# gravity to the last frame, never slowing, through the floor and out of its
# bottom. The frames show SECONDS of real time, so the cloth has that long to
# follow them down. Nothing is timed to land. At 1.6 s the sheet was through
# by a third of the way and the rest was a wad caught in the holes, so the
# frames cover the pour alone.
SECONDS = arg("--seconds", 0.55)
GRAVITY = 9.81
FALL_BEGIN = (2 * DROP / GRAVITY) ** 0.5
FALL_TIME = (2 * (DROP + HEIGHT + DEPTH) / GRAVITY) ** 0.5 - FALL_BEGIN
HIT_TIME = (2 * (DROP + HEIGHT) / GRAVITY) ** 0.5 - FALL_BEGIN
TIME_SCALE = (SECONDS if POUR or RISE or LIFT else FALL_TIME) / ((FALL_END - 1) / 24)
# Collision impulses are clamped so a hard hit cannot blow the solver up.
IMPULSE_CLAMP = arg("--clamp", 2.0)

# The exported view: screen points every 16 canvas px, from 64 px outside
# the frame, so the page's overscan lattice nodes are covered.
GRID_STEP = 16
GRID_X0, GRID_Y0 = -64, -64
GRID_W = (1280 + 128) // GRID_STEP + 1
GRID_H = (720 + 128) // GRID_STEP + 1
# A cloth point further than this from where it started, in canvas px, or a
# triangle stretched past this many times its rest edge, has blown up. It is
# left out of the view from above, so it cannot reach the page.
BLOWN_PX = 5000 if POUR else 600
BLOWN_STRETCH = 6
# In the pour, cloth that has gone below the floor's underside has fallen
# through a hole and out of sight: from above it reads as uncovered.
GONE_Z = -FLOOR_DEPTH - 0.05
# int16 storage: material offsets in eighths of a px, height in thousandths.
OFFSET_Q = 8.0
HEIGHT_Q = 1000.0


def to_canvas(x, y):
    return (x + CANVAS_W / 2) * PX, (CANVAS_H / 2 - y) * PX


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.materials):
        for item in list(block):
            block.remove(item)


def inside(point, loop):
    x, y = point
    hit = False
    for (x0, y0), (x1, y1) in zip(loop, loop[1:] + loop[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) / (y1 - y0) * (x1 - x0):
            hit = not hit
    return hit


def outer_loops(loops):
    """The loops no other loop encloses: each letter's outline without its
    counters."""
    return [loop for loop in loops if not any(other is not loop and inside(loop[0], other) for other in loops)]


def convex_hull(points):
    """Andrew's monotone chain, counter-clockwise, no repeated end."""
    pts = sorted(set((round(x, 3), round(y, 3)) for x, y in points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def simplify_loop(loop, tolerance):
    """Douglas-Peucker on a closed loop, canvas px: split at the two points
    farthest apart, then keep only what departs more than `tolerance`."""
    pts = np.asarray(loop, float)
    if tolerance <= 0 or len(pts) < 8:
        return [list(p) for p in pts]

    def open_chain(chain):
        keep = np.zeros(len(chain), bool)
        keep[0] = keep[-1] = True
        stack = [(0, len(chain) - 1)]
        while stack:
            a, b = stack.pop()
            if b <= a + 1:
                continue
            seg = chain[b] - chain[a]
            length = np.hypot(*seg)
            rel = chain[a + 1:b] - chain[a]
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length if length > 0 else np.hypot(rel[:, 0], rel[:, 1])
            k = int(np.argmax(dist))
            if dist[k] > tolerance:
                keep[a + 1 + k] = True
                stack += [(a, a + 1 + k), (a + 1 + k, b)]
        return chain[keep]

    far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    first = open_chain(pts[:far + 1])
    second = open_chain(np.vstack([pts[far:], pts[:1]]))
    out = np.vstack([first[:-1], second[:-1]])
    return [list(p) for p in out]


def letter_loops(glyphs, loops):
    if SIMPLIFY <= 0:
        return loops
    thin = [simplify_loop(loop, SIMPLIFY) for loop in loops]
    print("SIMPLIFY outlines %d -> %d points at %.1f px" % (sum(len(l) for l in loops), sum(len(l) for l in thin), SIMPLIFY))
    return thin


def dished_letters(glyphs, half):
    """The letters extruded from their outlines, counters open, spanning -half
    to +half, with a dished top: flat at full height out to the rim, sloping
    down to a floor DISH units lower. The rim's inner edge and the floor's
    outer edge are contours of the letters' distance field
    (scripts/dish-showcase-letters.py), which never cross, so the top is
    filled between nested loops and cannot fold."""
    import bmesh
    import subprocess
    letters = outline_object("Letters", letter_loops(glyphs, glyphs["letters"]), half)
    stroke = glyphs["strokePx"] - 2 * glyphs.get("letterInsetPx", 4)
    rim = DISH_RIM * stroke
    edge = rim + DISH_INSET * stroke
    job = json.dumps({"loops": glyphs["letters"], "levels": [rim, edge], "simplify": SIMPLIFY or 1.0})
    done = subprocess.run(["python3", os.path.join(ROOT, "scripts", "dish-showcase-letters.py")], input=job, capture_output=True, text=True, check=True)
    rim_rings, floor_rings = json.loads(done.stdout)["levels"]
    from mathutils import Vector
    from mathutils.geometry import delaunay_2d_cdt
    bm = bmesh.new()
    bm.from_mesh(letters.data)
    cap = [f for f in bm.faces if f.normal.z > 0.9 and f.calc_center_median().z > half - 1e-4]
    bmesh.ops.delete(bm, geom=cap, context="FACES_ONLY")
    bm.verts.ensure_lookup_table()
    # The outline's own top vertices, kept, so the top stays joined to the
    # walls; then the rim and floor rings. One constrained triangulation over
    # all of them: the rings are its constraints, so no triangle crosses one,
    # and a triangle's height comes from the rings its corners sit on.
    outline_edges = [e for e in bm.edges if len(e.link_faces) == 1 and all(v.co.z > half - 1e-4 for v in e.verts)]
    outline_verts = sorted({v for e in outline_edges for v in e.verts}, key=lambda v: v.index)
    points, heights, keep, edges = [], [], [], []
    slot = {}
    for v in outline_verts:
        slot[v] = len(points)
        points.append(Vector((v.co.x, v.co.y)))
        heights.append(half)
        keep.append(v)
    for e in outline_edges:
        edges.append((slot[e.verts[0]], slot[e.verts[1]]))

    def ring(rings, z):
        for r in rings:
            first = len(points)
            for x, y in r:
                points.append(Vector((x / PX - CANVAS_W / 2, CANVAS_H / 2 - y / PX)))
                heights.append(z)
                keep.append(None)
            edges.extend((first + k, first + (k + 1) % len(r)) for k in range(len(r)))

    ring(rim_rings, half)
    ring(floor_rings, half - DISH)
    out_verts, _, out_faces, vert_orig, _, _ = delaunay_2d_cdt(points, edges, [], 0, 1e-6)
    made_verts = []
    for n, co in enumerate(out_verts):
        origin = vert_orig[n]
        if origin and keep[origin[0]] is not None:
            made_verts.append(keep[origin[0]])
        else:
            z = heights[origin[0]] if origin else half
            made_verts.append(bm.verts.new((co.x, co.y, z)))
    # inside the letters: even-odd over the outline loops, counters open
    loops = [[(x / PX - CANVAS_W / 2, CANVAS_H / 2 - y / PX) for x, y in loop] for loop in letter_loops(glyphs, glyphs["letters"])]

    def inside_letters(px, py):
        hit = False
        for loop in loops:
            for (ax, ay), (bx, by) in zip(loop, loop[1:] + loop[:1]):
                if (ay > py) != (by > py) and px < ax + (py - ay) / (by - ay) * (bx - ax):
                    hit = not hit
        return hit

    made = 0
    for face in out_faces:
        cx = sum(out_verts[k].x for k in face) / len(face)
        cy = sum(out_verts[k].y for k in face) / len(face)
        if not inside_letters(cx, cy):
            continue
        try:
            f = bm.faces.new([made_verts[k] for k in face])
        except ValueError:
            continue
        f.normal_update()
        if f.normal.z < 0:
            f.normal_flip()
        made += 1
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    # the top must be closed; the bottom cap is left unjoined by the curve
    # conversion, as the flat letters' is, which nothing touches
    open_edges = sum(1 for e in bm.edges if len(e.link_faces) < 2 and min(v.co.z for v in e.verts) > half - DISH - 1e-4)
    bm.normal_update()
    bm.to_mesh(letters.data)
    bm.free()
    letters.data.update()
    print("DISH closed: %d open edges on the top" % open_edges)
    print("DISH top: stroke %.0f px, rim to %.1f px, slope to %.1f px, %d rim and %d floor rings, %d triangles" % (stroke, rim, edge, len(rim_rings), len(floor_rings), made))
    print("DISH letters: %d vertices, %d faces, %.2f units down" % (len(letters.data.vertices), len(letters.data.polygons), DISH))
    return letters


def outline_object(name, loops, half_depth):
    """A solid from closed outlines in canvas px, extruded `half_depth` either
    side of z = 0. Outlines inside others are holes, as in the glyph."""
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    curve.extrude = half_depth
    for loop in loops:
        spline = curve.splines.new("POLY")
        spline.points.add(len(loop) - 1)
        for point, (x, y) in zip(spline.points, loop):
            point.co = (x / PX - CANVAS_W / 2, CANVAS_H / 2 - y / PX, 0.0, 1.0)
        spline.use_cyclic_u = True
    ob = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(ob)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.convert(target="MESH")
    return bpy.context.object


def make_floor(glyphs):
    """A thick floor, its top at z = 0, with a hole the shape of the word, a
    little wider than the glyph so the cloth can line the walls."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -FLOOR_DEPTH / 2))
    floor = bpy.context.object
    floor.name = "Floor with letter holes"
    floor.scale = (CANVAS_W + 2 * MARGIN + 4, CANVAS_H + 2 * MARGIN + 4, FLOOR_DEPTH)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if LIFT:
        # the letters grow out of the floor itself: no holes
        floor.modifiers.new("Floor collision", "COLLISION")
        floor.collision.thickness_outer = 0.01
        floor.collision.cloth_friction = FRICTION
        return floor
    if HULL:
        loops = [convex_hull([point for loop in glyphs["holes"] for point in loop])]
    elif SOLID:
        loops = outer_loops(glyphs["holes"])
    else:
        loops = glyphs["holes"]
    join = JOIN
    if JOIN_AUTO:
        pts = [np.asarray(loop, float) for loop in loops]
        widest = 0.0
        for i, a in enumerate(pts):
            near = min(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min() for j, b in enumerate(pts) if j != i) if len(pts) > 1 else 0.0
            widest = max(widest, near)
        join = widest / 2 + 1
    union = False
    if MOUTH > 0:
        # The channel between each pair of lines: the letter boxes by line,
        # as the seam finds them, and a rectangle from the middle of one
        # line's letters to the middle of the next's.
        boxes = []
        for loop in outer_loops(glyphs["letters"]):
            xs = [x for x, _ in loop]
            ys = [y for _, y in loop]
            boxes.append((min(xs), max(xs), min(ys), max(ys)))
        rows, _, _ = letter_cuts(glyphs)
        lines = [[b for b in boxes if sum((b[2] + b[3]) / 2 > row for row in rows) == n] for n in range(len(rows) + 1)]
        for upper, lower in zip(lines, lines[1:]):
            spans = [(min(b[0] for b in line), max(b[1] for b in line)) for line in (upper, lower)]
            short = min(spans, key=lambda span: span[1] - span[0])
            cx, half = (short[0] + short[1]) / 2, MOUTH * (short[1] - short[0]) / 2
            top = (min(b[2] for b in upper) + max(b[3] for b in upper)) / 2
            bottom = (min(b[2] for b in lower) + max(b[3] for b in lower)) / 2
            loops = loops + [[[cx - half, top], [cx + half, top], [cx + half, bottom], [cx - half, bottom]]]
            union = True
            print("MOUTH %.0f px wide between lines, x %.0f-%.0f" % (2 * half, cx - half, cx + half))
    if join > 0 or SMOOTH > 0 or union:
        import subprocess
        job = json.dumps({"loops": loops, "join": join, "smooth": SMOOTH, "union": union})
        done = subprocess.run(["python3", os.path.join(ROOT, "scripts", "round-showcase-holes.py")], input=job, capture_output=True, text=True, check=True)
        loops = json.loads(done.stdout)["loops"]
        # joining can trap floor between letters; with solid letters that
        # island is exactly what the cloth should not drape over
        if SOLID:
            loops = outer_loops(loops)
        print("HOLES rounded: join %.1f px%s, smooth %.0f px, %d outline(s)" % (join, " (auto)" if JOIN_AUTO else "", SMOOTH, len(loops)))
    HOLE_LOOPS[:] = loops
    cutter = outline_object("Hole cutter", loops, FLOOR_DEPTH + 1)
    holes = floor.modifiers.new("Letter holes", "BOOLEAN")
    holes.operation = "DIFFERENCE"
    holes.solver = "EXACT"
    holes.object = cutter
    holes.use_self = True
    holes.use_hole_tolerant = True
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = floor
    floor.select_set(True)
    bpy.ops.object.modifier_apply(modifier=holes.name)
    bpy.data.objects.remove(cutter)
    if BRIDGES:
        # Thin walls of floor standing across the holes, full depth, top flush
        # with the floor. The letters are keyframed, so they pass through.
        _, _, bars = letter_cuts(glyphs)
        for n, (bx0, by0, bx1, by1) in enumerate(bars):
            (ax, ay), (cx, cy) = ((bx0 / PX - CANVAS_W / 2, CANVAS_H / 2 - by0 / PX), (bx1 / PX - CANVAS_W / 2, CANVAS_H / 2 - by1 / PX))
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=((ax + cx) / 2, (ay + cy) / 2, -FLOOR_DEPTH / 2))
            bar = bpy.context.object
            bar.name = "Bridge %d" % n
            bar.scale = (abs(cx - ax), abs(cy - ay), FLOOR_DEPTH)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bar.modifiers.new("Bridge collision", "COLLISION")
            bar.collision.thickness_outer = 0.01
            bar.collision.cloth_friction = FRICTION
    floor.modifiers.new("Floor collision", "COLLISION")
    floor.collision.thickness_outer = 0.01
    floor.collision.cloth_friction = FRICTION
    print("FLOOR %d faces" % len(floor.data.polygons))
    return floor


def make_letters(glyphs):
    """The letters, solid and a little narrower than the glyph, above the
    cloth over their holes. They fall together, hit it and push it in. With
    --rise they are pillars in their holes instead, and rise up through it."""
    if LIFT:
        # Tall slabs that start wholly under the floor, their collision shell
        # clear of the resting cloth, and rise through its surface: the floor
        # has no holes, but nothing makes them collide with it. Moved, never
        # scaled, so the cloth only ever meets a solid.
        half = (LIFT_HEIGHT + 0.6) / 2
        letters = dished_letters(glyphs, half) if DISH > 0 else outline_object("Letters", letter_loops(glyphs, outer_loops(glyphs["letters"])), half)
        start = -0.06
        grow_end = 1 + max(2, int(round((FRAME_END - 1) * LIFT_GROW)))
        slide_start = min(FRAME_END - 2, grow_end + max(1, int(round((FRAME_END - 1) * LIFT_HOLD))))
        for frame in range(1, FRAME_END + 1):
            u = min(1.0, (frame - 1) / (grow_end - 1))
            letters.location.z = start + (LIFT_HEIGHT - start) * u * u * (3 - 2 * u) - half
            v = max(0.0, (frame - slide_start) / (FRAME_END - slide_start))
            # easing away: slow to leave the hold, faster off the frame
            letters.location.y = -LIFT_SLIDE * v * v
            letters.keyframe_insert(data_path="location", index=2, frame=frame)
            letters.keyframe_insert(data_path="location", index=1, frame=frame)
        for curve in letters.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
        LIFT_PHASES.update({"growEnd": grow_end, "slideStart": slide_start})
        print("LETTERS grow to %.2f by frame %d, hold, slide %.1f units from frame %d" % (LIFT_HEIGHT, grow_end, LIFT_SLIDE, slide_start))
        letters.modifiers.new("Letter collision", "COLLISION")
        letters.collision.thickness_outer = 0.04
        letters.collision.cloth_friction = LIFT_GRIP
        return letters
    if RISE:
        half = (RISE_HEIGHT + 0.6) / 2
        letters = outline_object("Letters", outer_loops(glyphs["letters"]), half)
        start = -0.03
        rising = max(2, int(round((FRAME_END - 1) * RISE_FRAC)))
        for frame in range(1, FRAME_END + 1):
            u = min(1.0, (frame - 1) / rising)
            letters.location.z = start + RISE_HEIGHT * u * u * (3 - 2 * u) - half
            letters.keyframe_insert(data_path="location", index=2, frame=frame)
        for point in letters.animation_data.action.fcurves[0].keyframe_points:
            point.interpolation = "LINEAR"
        print("LETTERS rise %.2f units over frames 1-%d, %.2f tall" % (RISE_HEIGHT, rising + 1, 2 * half))
        letters.modifiers.new("Letter collision", "COLLISION")
        letters.collision.thickness_outer = 0.04
        letters.collision.cloth_friction = FRICTION
        return letters
    letters = outline_object("Letters", outer_loops(glyphs["letters"]) if SOLID else glyphs["letters"], LETTER_HALF)
    # the height at which the letters' undersides touch the cloth
    touch = SHEET_Z + LETTER_HALF + 0.02
    # A keyframe on every frame, so the fall is exactly a drop under gravity.
    for frame in range(1, FRAME_END + 1):
        if POUR:
            t = (frame - 1) / (FRAME_END - 1) * SECONDS + FALL_BEGIN
        else:
            t = min(1.0, (frame - 1) / (FALL_END - 1)) * FALL_TIME + FALL_BEGIN
        fallen = 0.5 * GRAVITY * t * t - DROP
        letters.location.z = touch + HEIGHT - fallen
        letters.keyframe_insert(data_path="location", index=2, frame=frame)
    for point in letters.animation_data.action.fcurves[0].keyframe_points:
        point.interpolation = "LINEAR"
    if POUR:
        print("LETTERS at %.1f m/s on the hit, %.1f m/s and %.1f below the cloth on the last frame" % (GRAVITY * FALL_BEGIN, GRAVITY * (SECONDS + FALL_BEGIN), 0.5 * GRAVITY * (SECONDS + FALL_BEGIN) ** 2 - DROP - HEIGHT))
    else:
        print("LETTERS hit the cloth at frame %.0f of %d" % (1 + HIT_TIME / FALL_TIME * (FALL_END - 1), FALL_END))
    letters.modifiers.new("Letter collision", "COLLISION")
    # A thick shell: pressed against a thin one, the cloth slips through.
    letters.collision.thickness_outer = 0.04
    letters.collision.cloth_friction = FRICTION
    if RELEASE:
        # The first frame the letters' tops are past the release depth: from
        # there they no longer collide, so they stop holding the cloth up.
        let_go = FRAME_END + 1
        for frame in range(1, FRAME_END + 1):
            if POUR:
                t = (frame - 1) / (FRAME_END - 1) * SECONDS + FALL_BEGIN
            else:
                t = min(1.0, (frame - 1) / (FALL_END - 1)) * FALL_TIME + FALL_BEGIN
            if touch + HEIGHT - (0.5 * GRAVITY * t * t - DROP) + LETTER_HALF <= -RELEASE_DEPTH:
                let_go = frame
                break
        for frame, on in ((1, True), (let_go - 1, True), (let_go, False)):
            if 1 <= frame <= FRAME_END:
                letters.collision.use = on
                letters.keyframe_insert(data_path="collision.use", frame=frame)
        for curve in letters.animation_data.action.fcurves:
            if curve.data_path == "collision.use":
                for point in curve.keyframe_points:
                    point.interpolation = "CONSTANT"
        print("RELEASE the letters let go at frame %d of %d, %.1f below the floor's top" % (let_go, FRAME_END, RELEASE_DEPTH))
    return letters


def letter_cuts(glyphs):
    """Where --slice cuts the sheet, in canvas px: the y between the lines of
    the word, and for each line the x between its neighbouring letters. A
    letter is an outline and anything above or below it that overlaps it
    across, like the i and its dot."""
    boxes = []
    for loop in outer_loops(glyphs["letters"]):
        xs = [x for x, _ in loop]
        ys = [y for _, y in loop]
        boxes.append([min(xs), max(xs), min(ys), max(ys)])
    font = glyphs["fontPx"]
    merged = True
    while merged:
        merged = False
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                p, q = boxes[a], boxes[b]
                # a small piece stacked on a letter, not beside it: mostly
                # overlapping across and apart or nearly so up and down, as
                # the i's dot is on its stem
                across = min(p[1], q[1]) - max(p[0], q[0])
                upright = min(p[3], q[3]) - max(p[2], q[2])
                small = min(p[3] - p[2], q[3] - q[2]) < 0.3 * font
                if small and across > 0.5 * min(p[1] - p[0], q[1] - q[0]) and -0.5 * font < upright < 0.1 * font:
                    boxes[a] = [min(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), max(p[3], q[3])]
                    del boxes[b]
                    merged = True
                    break
            if merged:
                break
    boxes.sort(key=lambda box: (box[2] + box[3]) / 2)
    lines = [[boxes[0]]]
    for box in boxes[1:]:
        if (box[2] + box[3]) / 2 - (lines[-1][-1][2] + lines[-1][-1][3]) / 2 > 0.5 * font:
            lines.append([])
        lines[-1].append(box)
    rows = [(max(b[3] for b in upper) + min(b[2] for b in lower)) / 2 for upper, lower in zip(lines, lines[1:])]
    columns = []
    # the bridges: (x0, y0, x1, y1) in canvas px, one on each cut that runs
    # between two letters rather than through them
    bridges = []
    pad = glyphs.get("holeOutsetPx", 8) + 12
    for line in lines:
        line.sort(key=lambda box: box[0])
        columns.append([(left[1] + right[0]) / 2 for left, right in zip(line, line[1:])])
        top = min(b[2] for b in line) - pad
        bottom = max(b[3] for b in line) + pad
        for left, right in zip(line, line[1:]):
            if right[0] > left[1]:
                x = (left[1] + right[0]) / 2
                bridges.append((x - BRIDGE_PX / 2, top, x + BRIDGE_PX / 2, bottom))
    for row in rows:
        left = min(b[0] for b in boxes) - pad
        right = max(b[1] for b in boxes) + pad
        bridges.append((left, row - BRIDGE_PX / 2, right, row + BRIDGE_PX / 2))
    print("SLICE %d line(s), %s letters, %d bridges" % (len(lines), "+".join(str(len(line)) for line in lines), len(bridges)))
    return rows, columns, bridges


def make_sheet(glyphs):
    x0, x1 = -CANVAS_W / 2 - MARGIN, CANVAS_W / 2 + MARGIN
    y0, y1 = -CANVAS_H / 2 - MARGIN, CANVAS_H / 2 + MARGIN
    nx = int(round((x1 - x0) / SPACING)) + 1
    ny = int(round((y1 - y0) / SPACING)) + 1
    grid = [(x0 + i * SPACING, y1 - j * SPACING) for j in range(ny) for i in range(nx)]
    if SLICE or TEAR_SEAM:
        rows, columns, _ = letter_cuts(glyphs)
    if TEAR:
        # Patch centres on a jittered grid, the same every bake, and each
        # face goes to the nearest: irregular cells, like broken glass.
        rng = np.random.default_rng(7)
        cx0, cy0 = to_canvas(x0, y1)
        cx1, cy1 = to_canvas(x1, y0)
        gx = np.arange(cx0, cx1 + TEAR_PX, TEAR_PX)
        gy = np.arange(cy0, cy1 + TEAR_PX, TEAR_PX)
        seeds = np.array([(x, y) for y in gy for x in gx], float)
        seeds += rng.uniform(-0.45, 0.45, seeds.shape) * TEAR_PX
    vertices = []
    index = {}
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            corners = (a, a + nx, a + nx + 1, a + 1)
            region = 0
            if TEAR:
                cx, cy = to_canvas(grid[a][0] + SPACING / 2, grid[a][1] - SPACING / 2)
                region = int(np.argmin((seeds[:, 0] - cx) ** 2 + (seeds[:, 1] - cy) ** 2))
            elif TEAR_SEAM:
                _, cy = to_canvas(grid[a][0] + SPACING / 2, grid[a][1] - SPACING / 2)
                region = sum(cy > row for row in rows)
            elif SLICE:
                cx, cy = to_canvas(grid[a][0] + SPACING / 2, grid[a][1] - SPACING / 2)
                line = sum(cy > row for row in rows)
                region = (line, sum(cx > column for column in columns[line]))
            # A corner is shared only within a piece: across a cut each side
            # has its own vertex, so nothing holds the pieces together.
            face = []
            for k in corners:
                key = (k, region)
                if key not in index:
                    index[key] = len(vertices)
                    vertices.append((grid[k][0], grid[k][1], SHEET_Z))
                face.append(index[key])
            faces.append(tuple(face))
    # With --tear, the copies of a grid point on either side of a seam are
    # stitched by loose edges, which the cloth solver takes as sewing springs.
    edges = []
    if TEAR or TEAR_SEAM:
        if TEAR_SEAM:
            xs = [x for loop in outer_loops(glyphs["letters"]) for x, _ in loop]
            reach = (min(xs) - SEAM_REACH, max(xs) + SEAM_REACH)
        copies = {}
        for (k, _), v in index.items():
            copies.setdefault(k, []).append(v)
        for k, vs in copies.items():
            if TEAR_SEAM and not reach[0] <= to_canvas(*grid[k])[0] <= reach[1]:
                continue
            edges.extend(zip(vs, vs[1:]))
        print("TEAR %d piece(s), %d stitches, force cap %.2f" % (len(set(r for _, r in index)), len(edges), TEAR_FORCE))
    mesh = bpy.data.meshes.new("Cloth")
    mesh.from_pydata(vertices, edges, faces)
    if DENSE > 0:
        # Subdivide the cells over the word; the seam to the coarse cells is
        # filled with triangles. Then everything is triangles, so the export
        # and the solver see one kind of face.
        import bmesh
        xs = [x for loop in glyphs["letters"] for x, _ in loop]
        ys = [y for loop in glyphs["letters"] for _, y in loop]
        bx0, bx1 = min(xs) - DENSE_PAD, max(xs) + DENSE_PAD
        by0, by1 = min(ys) - DENSE_PAD, max(ys) + DENSE_PAD
        cuts = max(1, int(round(SPACING * PX / DENSE)) - 1)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        band = []
        for f in bm.faces:
            c = f.calc_center_median()
            cx, cy = to_canvas(c.x, c.y)
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                band.append(f)
        edges_in = list({e for f in band for e in f.edges})
        bmesh.ops.subdivide_edges(bm, edges=edges_in, cuts=cuts, use_grid_fill=True, use_single_edge=False)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        print("DENSE %d cells over the word cut %d ways: %d points, %d triangles" % (len(band), cuts + 1, len(mesh.vertices), len(mesh.polygons)))
    sheet = bpy.data.objects.new("Cloth", mesh)
    bpy.context.collection.objects.link(sheet)

    cloth = sheet.modifiers.new("Cloth", "CLOTH")
    s = cloth.settings
    s.quality = QUALITY
    s.time_scale = TIME_SCALE
    s.mass = 0.3
    # Stiff in tension, so the fabric slides in rather than stretching, and
    # soft in compression and bending, so it gathers into pleats on the floor.
    s.air_damping = arg("--air", 1.0)
    s.tension_stiffness = arg("--tension", 40)
    s.compression_stiffness = arg("--compress", 0.5)
    s.shear_stiffness = arg("--shear", 4)
    s.bending_stiffness = arg("--bend", 0.05)
    s.use_sewing_springs = TEAR or TEAR_SEAM
    if TEAR or TEAR_SEAM:
        s.sewing_force_max = TEAR_FORCE
    c = cloth.collision_settings
    c.use_collision = True
    c.distance_min = 0.02
    c.collision_quality = COLLISION_QUALITY
    c.impulse_clamp = IMPULSE_CLAMP
    c.self_impulse_clamp = IMPULSE_CLAMP
    c.use_self_collision = True
    c.self_distance_min = 0.015
    c.self_friction = 5.0
    cloth.point_cache.frame_start = 1
    cloth.point_cache.frame_end = FRAME_END
    if LIFT and "--air-slide" in ARGS and LIFT_PHASES:
        # Heavy damping while the word rises and holds, so loose cloth cannot
        # fling itself over the letters; light again for the slide, so the
        # cloth follows the letters off the frame.
        slide = LIFT_PHASES["slideStart"]
        for frame, value in ((1, s.air_damping), (slide - 1, s.air_damping), (slide + 4, arg("--air-slide", 1.0))):
            s.air_damping = value
            s.keyframe_insert(data_path="air_damping", frame=frame)
        print("AIR %.1f until frame %d, %.1f from frame %d" % (arg("--air", 1.0), slide - 1, arg("--air-slide", 1.0), slide + 4))
    return sheet, nx, ny


def grid_nodes():
    gx = GRID_X0 + GRID_STEP * np.arange(GRID_W, dtype=np.float64)
    gy = GRID_Y0 + GRID_STEP * np.arange(GRID_H, dtype=np.float64)
    return np.meshgrid(gx, gy)


def raster_top(pos, rest, tris):
    """The sheet seen from above on the export grid: at each screen point the
    material position (canvas px) and height of the topmost layer. Points the
    sheet no longer covers read as material far outside the frame."""
    if len(tris) == 0:
        # the whole sheet has gone: every point is floor
        gx, gy = grid_nodes()
        gx, gy = gx.ravel(), gy.ravel()
        return gx - 640.0, gy - 360.0, np.zeros_like(gx), np.zeros(gx.shape, bool)
    sx, sy = pos[:, 0], pos[:, 1]
    gu = (sx - GRID_X0) / GRID_STEP
    gv = (sy - GRID_Y0) / GRID_STEP
    tu, tv = gu[tris], gv[tris]
    lo_u = np.ceil(tu.min(axis=1)).astype(np.int64)
    lo_v = np.ceil(tv.min(axis=1)).astype(np.int64)
    span_u = (np.floor(tu.max(axis=1)).astype(np.int64) - lo_u).clip(-1, 8)
    span_v = (np.floor(tv.max(axis=1)).astype(np.int64) - lo_v).clip(-1, 8)
    a_u, b_u, c_u = tu[:, 0], tu[:, 1], tu[:, 2]
    a_v, b_v, c_v = tv[:, 0], tv[:, 1], tv[:, 2]
    det = (b_v - c_v) * (a_u - c_u) + (c_u - b_u) * (a_v - c_v)
    ok = np.abs(det) > 1e-12
    nodes, depth, mat_x, mat_y, height = [], [], [], [], []
    pz, rx, ry = pos[:, 2], rest[:, 0], rest[:, 1]
    for du in range(int(span_u.max()) + 1):
        for dv in range(int(span_v.max()) + 1):
            sel = ok & (du <= span_u) & (dv <= span_v)
            if not sel.any():
                continue
            u = (lo_u + du)[sel].astype(np.float64)
            v = (lo_v + dv)[sel].astype(np.float64)
            d = det[sel]
            w0 = ((b_v - c_v)[sel] * (u - c_u[sel]) + (c_u - b_u)[sel] * (v - c_v[sel])) / d
            w1 = ((c_v - a_v)[sel] * (u - c_u[sel]) + (a_u - c_u)[sel] * (v - c_v[sel])) / d
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            inside &= (u >= 0) & (u < GRID_W) & (v >= 0) & (v < GRID_H)
            if not inside.any():
                continue
            t = tris[sel][inside]
            w = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
            nodes.append(v[inside].astype(np.int64) * GRID_W + u[inside].astype(np.int64))
            z = (pz[t] * w).sum(axis=1)
            depth.append(z)
            mat_x.append((rx[t] * w).sum(axis=1))
            mat_y.append((ry[t] * w).sum(axis=1))
            height.append(z)
    nodes = np.concatenate(nodes)
    depth = np.concatenate(depth)
    mat_x = np.concatenate(mat_x)
    mat_y = np.concatenate(mat_y)
    height = np.concatenate(height)
    order = np.lexsort((-depth, nodes))
    first = np.unique(nodes[order], return_index=True)[1]
    top = order[first]
    gx, gy = grid_nodes()
    gx, gy = gx.ravel(), gy.ravel()
    # Uncovered: pretend the material there came from twice as far out, so
    # the flood always counts it as far from the word.
    out_x = gx + (gx - 640.0)
    out_y = gy + (gy - 360.0)
    out_z = np.zeros_like(gx)
    covered = np.zeros(gx.shape, bool)
    covered[nodes[top]] = True
    out_x[nodes[top]] = mat_x[top]
    out_y[nodes[top]] = mat_y[top]
    out_z[nodes[top]] = height[top]
    return out_x - gx, out_y - gy, out_z, covered


def export_mesh(sheet, glyphs):
    """The cloth itself, for the page to draw in 3D. int16 little-endian:
    rest x, y (canvas px x OFFSET_Q) for each point, which is also where the
    clip is sampled; then the triangles as uint16 point indices; then for each
    exported frame each point's x, y (canvas px x OFFSET_Q) and height (Blender
    units x HEIGHT_Q)."""
    every = int(arg("--mesh-every", 2))
    frames = list(range(1, FRAME_END + 1, every))
    if frames[-1] != FRAME_END:
        frames.append(FRAME_END)
    scene = bpy.context.scene
    rest = np.array([to_canvas(v.co.x, v.co.y) for v in sheet.data.vertices])
    # every face as a fan of triangles: quads, or triangles once refined
    tris = np.array([(p.vertices[0], p.vertices[k], p.vertices[k + 1]) for p in sheet.data.polygons for k in range(1, len(p.vertices) - 1)], dtype=np.int64)
    if len(rest) > 65535:
        raise SystemExit("mesh export: %d points will not index as uint16" % len(rest))
    blocks = [np.clip(np.round(rest * OFFSET_Q), -32767, 32767).astype("<i2").ravel(), tris.astype("<u2").ravel()]
    for frame in frames:
        scene.frame_set(frame)
        evaluated = sheet.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        evaluated.to_mesh_clear()
        co = np.nan_to_num(co.reshape(-1, 3))
        x, y = to_canvas(co[:, 0], co[:, 1])
        xyz = np.stack([np.round(x * OFFSET_Q), np.round(y * OFFSET_Q), np.round(co[:, 2] * HEIGHT_Q)], axis=1)
        blocks.append(np.clip(xyz, -32767, 32767).astype("<i2").ravel())
    os.makedirs(os.path.dirname(MESH_DATA), exist_ok=True)
    with open(MESH_DATA, "wb") as handle:
        for block in blocks:
            handle.write(block.tobytes())
    public = os.path.join(ROOT, "public")
    meta = {
        "version": 1,
        "word": glyphs["word"],
        "fontPx": glyphs["fontPx"],
        "points": int(len(rest)),
        "triangles": int(len(tris)),
        "frames": frames,
        "fps": 24,
        "seconds": SECONDS if POUR or RISE or LIFT else FALL_TIME,
        "rise": RISE_HEIGHT,
        "lift": {"height": LIFT_HEIGHT, "slide": LIFT_SLIDE, "dish": DISH, **LIFT_PHASES} if LIFT else None,
        "offsetScale": 1 / OFFSET_Q,
        "heightScale": 1 / HEIGHT_Q,
        "floorDepth": FLOOR_DEPTH,
        "sheetZ": SHEET_Z,
        "holes": [[[round(float(x), 1), round(float(y), 1)] for x, y in loop] for loop in HOLE_LOOPS],
        "layout": "int16 rest x,y per point; uint16 triangles; per frame int16 x,y,z per point",
        "data": os.path.relpath(MESH_DATA, public) if MESH_DATA.startswith(public) else MESH_DATA,
    }
    os.makedirs(os.path.dirname(MESH_META), exist_ok=True)
    with open(MESH_META, "w", encoding="utf8") as handle:
        json.dump(meta, handle, ensure_ascii=False, separators=(",", ":"))
    print("MESH %d points, %d triangles, %d frames, %.1f MB -> %s" % (len(rest), len(tris), len(frames), os.path.getsize(MESH_DATA) / 1e6, MESH_DATA))
    # Does the last frame leave the frame clear? Cloth that carries the
    # picture (its rest point inside the frame), still inside the frame.
    # Every point counts: the page carries picture on the margin too,
    # mirrored past the frame's edge.
    x, y = to_canvas(co[:, 0], co[:, 1])
    inside = (x >= 0) & (x <= 1280) & (y >= 0) & (y <= 720) & (co[:, 2] > -0.05)
    print("LAST FRAME %d points still in the frame" % int(inside.sum()))


def export(sheet, nx, ny, glyphs):
    scene = bpy.context.scene
    rest = np.array([to_canvas(v.co.x, v.co.y) for v in sheet.data.vertices])
    quads = np.array([list(p.vertices) for p in sheet.data.polygons], dtype=np.int64)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    fields = []
    stats = []
    for frame in SAMPLE_FRAMES:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = sheet.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        evaluated.to_mesh_clear()
        co = co.reshape(-1, 3)
        pos = np.empty_like(co)
        pos[:, 0], pos[:, 1] = to_canvas(co[:, 0], co[:, 1])
        pos[:, 2] = co[:, 2]
        sound = np.isfinite(pos).all(axis=1) & (np.hypot(pos[:, 0] - rest[:, 0], pos[:, 1] - rest[:, 1]) < BLOWN_PX) & (np.abs(pos[:, 2]) < (100 if POUR else 10))
        flat = np.stack([np.hypot(*(pos[tris[:, a], :2] - pos[tris[:, b], :2]).T) for a, b in ((0, 1), (1, 2), (2, 0))], axis=1)
        keep = sound[tris].all(axis=1) & (flat.max(axis=1) < BLOWN_STRETCH * SPACING * PX * 2 ** 0.5)
        if POUR:
            keep &= pos[tris, 2].mean(axis=1) > GONE_Z
        wx, wy, z, covered = raster_top(np.nan_to_num(pos), rest, tris[keep])
        move = np.hypot(pos[:, 0] - rest[:, 0], pos[:, 1] - rest[:, 1])[sound]
        stats.append((frame, float(move.max()), float(np.median(move)), float(np.hypot(wx, wy)[covered].max()) if covered.any() else 0.0, int((~covered).sum()), float(co[:, 2].min()), int((~sound).sum()), int((~keep).sum())))
        fields.append((wx, wy, z))
    quantised = []
    for wx, wy, z in fields:
        quantised.append(np.clip(np.round(wx * OFFSET_Q), -32767, 32767).astype("<i2"))
        quantised.append(np.clip(np.round(wy * OFFSET_Q), -32767, 32767).astype("<i2"))
        quantised.append(np.clip(np.round(z * HEIGHT_Q), -32767, 32767).astype("<i2"))
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "wb") as handle:
        for block in quantised:
            handle.write(block.tobytes())
    meta = {
        "version": 2,
        "word": glyphs["word"],
        "fontPx": glyphs["fontPx"],
        "frames": list(SAMPLE_FRAMES),
        "grid": {"x0": GRID_X0, "y0": GRID_Y0, "step": GRID_STEP, "width": GRID_W, "height": GRID_H},
        "layout": "per sample frame: material x offset, material y offset, height; int16 row-major grids",
        "offsetScale": 1 / OFFSET_Q,
        "heightScale": 1 / HEIGHT_Q,
        "heightUnit": "Blender units, 100 canvas px each",
        "sheet": {"columns": nx, "rows": ny, "spacingPx": SPACING * PX, "marginPx": MARGIN * PX},
        # The depth the page shades a letter's hole to: the floor's own, when
        # the cloth pours right through it.
        "cloth": {"mouth": MOUTH, "release": RELEASE_DEPTH, "tear": {"px": TEAR_PX, "force": TEAR_FORCE} if TEAR else {"seam": True, "force": TEAR_FORCE, "reach": SEAM_REACH} if TEAR_SEAM else None, "holeJoin": "auto" if JOIN_AUTO else JOIN, "holeSmooth": SMOOTH, "hull": HULL, "sliced": SLICE, "bridges": BRIDGE_PX if BRIDGES else 0, "spacing": SPACING, "quality": QUALITY, "collisionQuality": COLLISION_QUALITY, "tension": arg("--tension", 40), "compression": arg("--compress", 0.5), "shear": arg("--shear", 4), "bending": arg("--bend", 0.05), "solidLetters": SOLID},
        "fall": {"end": FALL_END, "height": HEIGHT, "hitDrop": HIT_DROP, "depth": FLOOR_DEPTH if POUR else DEPTH, "friction": FRICTION, "timeScale": TIME_SCALE, "impulseClamp": IMPULSE_CLAMP, **({"pour": True, "seconds": SECONDS} if POUR else {})},
        "data": os.path.relpath(DATA, os.path.join(ROOT, "public")) if DATA.startswith(os.path.join(ROOT, "public")) else DATA,
        "description": ("A cloth on a floor with holes the shape of the word; the letters are thrown through them and the cloth pours down after them. Seen from above, top layer first; cloth below the floor reads as uncovered." if POUR else "A cloth on a floor with holes the shape of the word; the letters fall through together and pull it in. Seen from above, top layer first."),
    }
    os.makedirs(os.path.dirname(META), exist_ok=True)
    with open(META, "w", encoding="utf8") as handle:
        json.dump(meta, handle, indent=1)
        handle.write("\n")
    for frame, most, median, carried, uncovered, lowest, blown, dropped in stats:
        print("FRAME %3d  sheet moved max %6.1f px median %5.1f  |  frontier carried max %6.1f px  uncovered %4d  lowest z %.2f  |  blown points %d, triangles left out %d" % (frame, most, median, carried, uncovered, lowest, blown, dropped))


def section_row(glyphs):
    """The canvas y crossing the most letter, for the section view: a
    two-line word has no letters on its middle line."""
    best, best_y = -1.0, 360.0
    loops = glyphs["letters"]
    ys = [y for loop in loops for _, y in loop]
    for y in np.arange(min(ys) + 2, max(ys) - 2, 2.0):
        crossings = []
        for loop in loops:
            for (x0, y0), (x1, y1) in zip(loop, loop[1:] + loop[:1]):
                if (y0 <= y) != (y1 <= y):
                    crossings.append(x0 + (y - y0) / (y1 - y0) * (x1 - x0))
        crossings.sort()
        inside = sum(b - a for a, b in zip(crossings[0::2], crossings[1::2]))
        if inside > best:
            best, best_y = inside, float(y)
    return best_y


def render_raw(sheet, floor, letters, glyphs):
    """The raw bake, for review, every frame: oblique; top-down, the view the
    page reads, cloth only; and a section through the word at y = 0."""
    scene = bpy.context.scene
    frames_dir = os.path.join(RAW_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    def material(name, rgba):
        m = bpy.data.materials.new(name)
        m.diffuse_color = rgba
        return m

    sheet.data.materials.append(material("cloth", (0.85, 0.35, 0.12, 1)))
    floor.data.materials.append(material("floor", (0.16, 0.17, 0.2, 1)))
    letters.data.materials.append(material("letters", (0.8, 0.82, 0.86, 1)))
    os.makedirs(RAW_DIR, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(RAW_DIR, "cloth-bake.blend"))

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = 480 if SMALL else 960
    scene.render.resolution_y = 270 if SMALL else 540
    scene.render.image_settings.file_format = "PNG"

    def camera(name, location, target, ortho=None, clip=None):
        data = bpy.data.cameras.new(name)
        if ortho:
            data.type = "ORTHO"
            data.ortho_scale = ortho
        if clip:
            data.clip_start = clip
        data.clip_end = 200
        cam = bpy.data.objects.new(name, data)
        cam.location = location
        cam.rotation_euler = (mathutils.Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(cam)
        return cam

    cut = CANVAS_H / 2 - section_row(glyphs) / PX
    views = (
        # label, camera, wireframe, objects shown besides the cloth
        ("oblique", camera("oblique", (0, -9.5, 7.5), (0, 0, -0.6)), False, (floor, letters)),
        ("top", camera("top", (0, 0, 20), (0, 0, 0), ortho=CANVAS_W), True, ()),
        ("section", camera("section", (0, cut - 30, -0.7), (0, cut, -0.7), ortho=10.4, clip=30), False, (floor, letters)),
    )
    for label, cam, wire, shown in views[:2] if SMALL else views:
        scene.camera = cam
        sheet.show_wire = wire
        for ob in (floor, letters):
            ob.hide_render = ob not in shown
        for frame in range(1, FRAME_END + 1, EVERY):
            scene.frame_set(frame)
            scene.render.filepath = os.path.join(frames_dir, "%s_%04d.png" % (label, frame))
            bpy.ops.render.render(write_still=True)
        # and a video of the view, when every frame was rendered
        if EVERY == 1:
            import shutil
            import subprocess
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24",
                                "-i", os.path.join(frames_dir, label + "_%04d.png"),
                                "-c:v", "libvpx-vp9", "-b:v", "3M", "-pix_fmt", "yuv420p",
                                os.path.join(RAW_DIR, "%s-%s.webm" % (WORD, label))], check=False)


def preview_letters(letters, floor):
    """The letters' wireframe over their solid, from above and at an angle."""
    scene = bpy.context.scene
    os.makedirs(PREVIEW, exist_ok=True)
    scene.frame_set(FRAME_END // 3)
    floor.hide_render = True
    wire = letters.copy()
    wire.data = letters.data.copy()
    wire.animation_data_clear()
    wire.matrix_world = letters.matrix_world.copy()
    scene.collection.objects.link(wire)
    for mod in list(wire.modifiers):
        wire.modifiers.remove(mod)
    frame = wire.modifiers.new("wire", "WIREFRAME")
    frame.thickness = 0.005
    frame.use_replace = True
    solid = bpy.data.materials.new("solid")
    solid.diffuse_color = (0.85, 0.86, 0.9, 1)
    edges = bpy.data.materials.new("edges")
    edges.diffuse_color = (0.05, 0.05, 0.08, 1)
    letters.data.materials.clear()
    letters.data.materials.append(solid)
    wire.data.materials.clear()
    wire.data.materials.append(edges)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x, scene.render.resolution_y = 1600, 900
    top_z = max((letters.matrix_world @ v.co).z for v in letters.data.vertices)
    for name, loc, target, ortho in (("top", (0, 0, top_z + 20), (0, 0, top_z), CANVAS_W * 0.8), ("oblique", (-3.0, -3.2, top_z + 2.2), (-2.0, 0.0, top_z - 0.5), None)):
        data = bpy.data.cameras.new(name)
        if ortho:
            data.type = "ORTHO"
            data.ortho_scale = ortho
        cam = bpy.data.objects.new(name, data)
        cam.location = loc
        cam.rotation_euler = (mathutils.Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(cam)
        scene.camera = cam
        scene.render.filepath = os.path.join(PREVIEW, "letters-%s.png" % name)
        bpy.ops.render.render(write_still=True)
    print("PREVIEW %d vertices, %d faces -> %s" % (len(letters.data.vertices), len(letters.data.polygons), PREVIEW))


def main():
    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    scene.render.fps = 24
    with open(GLYPHS, encoding="utf8") as handle:
        glyphs = json.load(handle)
    print("WORD %s at %.1f px" % (glyphs["word"].replace("\n", " / "), glyphs["fontPx"]))
    floor = make_floor(glyphs)
    letters = make_letters(glyphs)
    if PREVIEW:
        preview_letters(letters, floor)
        return
    sheet, nx, ny = make_sheet(glyphs)
    print("SHEET %d x %d" % (nx, ny))
    bpy.context.view_layer.objects.active = sheet
    sheet.select_set(True)
    bpy.ops.ptcache.bake_all(bake=True)
    if not (RISE or LIFT):
        export(sheet, nx, ny, glyphs)
    if MESH:
        export_mesh(sheet, glyphs)
    if RAW:
        render_raw(sheet, floor, letters, glyphs)
    print("CLOTH_BAKE", json.dumps({"meta": META, "data": DATA}))


if __name__ == "__main__":
    main()
