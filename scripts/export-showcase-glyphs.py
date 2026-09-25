"""Export each showcase project's glyph, as the page draws it, for the cloth bakes.

Run from the repository root with the dev server up (npx astro dev):
  python3 scripts/export-showcase-glyphs.py [http://localhost:4321]

Firefox builds each project's word with the page's own buildGlyph, inside a
module worker as the page's solver does, at the size the Origin flow preset
gives it. The script traces three outlines of that field, in canvas px, into
scripts/showcase-glyphs/<slug>.json:
  glyph    the ink edge
  letters  LETTER_INSET px inside it: the solid letters that fall
  holes    HOLE_OUTSET px outside it: the holes in the floor they fall through
scripts/bake-showcase-cloth.py reads them, so the cloth lines up with the page
whatever the word, the font's fallbacks or the line layout.

Needs Firefox, and Python with numpy, contourpy and websockets.
"""

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile

import contourpy
import numpy as np
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scripts", "showcase-glyphs")
FIREFOX = "/Applications/Firefox.app/Contents/MacOS/firefox"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4321"

# The sequence in src/scripts/showcase-test.ts: projectWords at the panel's
# default 240 px, which buildGlyph squeezes to 90% of the frame's width when a
# word would run wider.
PROJECTS = (
    ("yope3d", "Yope3D", 240, False),
    ("spinstack", "SpinStack", 240, False),
    ("beta-vae", "β-VAE", 240, False),
    ("glyphinterpreter", "Glyph\nInterpreter", 240, False),
)
# SDF_SCALE and SDF_PAD in src/scripts/showcase-mesh.ts: field samples per
# canvas px, and canvas px of padding round the field.
SDF_SCALE = 0.5
SDF_PAD = 224
LETTER_INSET = 4.0
HOLE_OUTSET = 8.0

PAGE_SCRIPT = """(async () => {
  const words = %s;
  const mesh = await import('/src/scripts/showcase-mesh.ts');
  const source = "import { buildGlyph } from '" + location.origin + "/src/scripts/showcase-mesh.ts';" +
    "onmessage = (e) => { const g = buildGlyph(e.data.text, e.data.size, e.data.auto);" +
    " postMessage({ w: g.w, h: g.h, sdf: g.sdf, fontPx: g.fontPx, strokePx: g.strokePx }); };";
  const worker = new Worker(URL.createObjectURL(new Blob([source], { type: 'text/javascript' })), { type: 'module' });
  const encode = (field) => {
    const bytes = new Uint8Array(field.buffer, field.byteOffset, field.byteLength);
    let text = '';
    for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
    return btoa(text);
  };
  const out = [];
  for (const [text, size, auto] of words) {
    const g = await new Promise((resolve) => { worker.onmessage = (e) => resolve(e.data); worker.postMessage({ text, size, auto }); });
    const main = mesh.buildGlyph(text, size, auto);
    let differ = 0;
    for (let i = 0; i < g.sdf.length; i++) if (Math.abs(g.sdf[i] - main.sdf[i]) > 1e-3) differ++;
    out.push({ w: g.w, h: g.h, fontPx: g.fontPx, strokePx: g.strokePx, differ, sdf: encode(g.sdf) });
  }
  worker.terminate();
  return JSON.stringify(out);
})()"""


async def page_glyphs():
    profile = tempfile.mkdtemp(prefix="glyph-export-")
    port = 9377
    firefox = subprocess.Popen(
        [FIREFOX, "--headless", "--remote-debugging-port", str(port), "--profile", profile, "--no-remote", "--new-instance"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                ws = await websockets.connect("ws://127.0.0.1:%d/session" % port, max_size=None)
                break
            except OSError:
                await asyncio.sleep(0.2)
        ids = iter(range(1, 1000))

        async def command(method, params):
            mid = next(ids)
            await ws.send(json.dumps({"id": mid, "method": method, "params": params}))
            while True:
                message = json.loads(await ws.recv())
                if message.get("id") == mid:
                    if message.get("type") == "error":
                        raise RuntimeError("%s: %s" % (method, message.get("message")))
                    return message["result"]

        await command("session.new", {"capabilities": {}})
        context = (await command("browsingContext.create", {"type": "tab"}))["context"]
        await command("browsingContext.navigate", {"context": context, "url": BASE + "/showcase-test/", "wait": "complete"})
        words = json.dumps([[text, size, auto] for _, text, size, auto in PROJECTS], ensure_ascii=False)
        result = await command("script.evaluate", {
            "expression": PAGE_SCRIPT % words, "target": {"context": context},
            "awaitPromise": True, "resultOwnership": "none"})
        if result.get("type") == "exception":
            raise RuntimeError(json.dumps(result["exceptionDetails"])[:1000])
        await ws.close()
        return json.loads(result["result"]["value"])
    finally:
        firefox.terminate()
        firefox.wait(10)
        shutil.rmtree(profile, ignore_errors=True)


def outlines(field, level):
    """Closed outlines of the field at `level`, in canvas px."""
    lines = contourpy.contour_generator(z=field, line_type=contourpy.LineType.Separate).lines(level)
    return [[[round(float(i) / SDF_SCALE - SDF_PAD, 2), round(float(j) / SDF_SCALE - SDF_PAD, 2)] for i, j in line]
            for line in lines if len(line) >= 4]


def main():
    os.makedirs(OUT, exist_ok=True)
    for (slug, text, size, auto), glyph in zip(PROJECTS, asyncio.run(page_glyphs())):
        field = np.frombuffer(base64.b64decode(glyph["sdf"]), dtype="<f4").reshape(glyph["h"], glyph["w"])
        data = {
            "word": text,
            "fontSize": size,
            "fontAuto": auto,
            "fontPx": glyph["fontPx"],
            "strokePx": glyph["strokePx"],
            "letterInsetPx": LETTER_INSET,
            "holeOutsetPx": HOLE_OUTSET,
            "glyph": outlines(field, 0.0),
            "letters": outlines(field, -LETTER_INSET),
            "holes": outlines(field, HOLE_OUTSET),
        }
        with open(os.path.join(OUT, slug + ".json"), "w", encoding="utf8") as handle:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
        points = sum(len(line) for line in data["glyph"])
        print("%-17s %5.1f px font, %2d px stroke, %2d outlines, %5d points; worker and main thread differ at %d samples"
              % (slug, glyph["fontPx"], glyph["strokePx"], len(data["glyph"]), points, glyph["differ"]))


if __name__ == "__main__":
    main()
