"""Build the moving selector movie and small demo videos for /showcase-test/.

Requires Python with Pillow, and ffmpeg on PATH. No footage is downloaded.
Run: python scripts/generate-showcase-assets.py [--font /path/to/bold.ttf]
"""

import argparse
import math
from pathlib import Path
import random
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/media/showcase-test'
WIDTH, HEIGHT, FPS, SECONDS = 1280, 720, 30, 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, default=Path('/System/Library/Fonts/Supplemental/Arial Black.ttf'))
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error('Provide a bold TrueType font using --font.')
    OUT.mkdir(parents=True, exist_ok=True)

    # Keep the footage fixed in screen space; only this selector movie moves.
    # Four discrete gray levels select clips. They are IDs, not opacity.
    rng = random.Random(7)
    cols, rows = 4, 3
    vertices = []
    for row in range(rows + 1):
        vertices.append([
            (round(col * WIDTH / cols + (rng.uniform(-85, 85) if 0 < col < cols else 0)),
             round(row * HEIGHT / rows + (rng.uniform(-65, 65) if 0 < row < rows else 0)))
            for col in range(cols + 1)
        ])
    background = Image.new('L', (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(background)
    for row in range(rows):
        for col in range(cols):
            draw.polygon([
                vertices[row][col], vertices[row][col + 1],
                vertices[row + 1][col + 1], vertices[row + 1][col],
            ], fill=85 * ((row + col) % 2))
    inside = background.point([170 if value < 43 else 255 for value in range(256)])

    font = ImageFont.truetype(str(args.font), 230)
    bounds = font.getbbox('Yope3D')
    word = Image.new('L', (bounds[2] - bounds[0], bounds[3] - bounds[1]), 0)
    ImageDraw.Draw(word).text((-bounds[0], -bounds[1]), 'Yope3D', font=font, fill=255)
    # Hard labels avoid intermediate gray values accidentally selecting clips.
    word = word.point(lambda value: 255 if value >= 128 else 0)

    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libvpx-vp9', '-lossless', '1', '-b:v', '0', '-cpu-used', '4',
        # The source is grayscale and lossless, so its luma keeps the four
        # ID levels exact while yuv420 remains broadly browser-decodable.
        '-pix_fmt', 'yuv420p', str(OUT / 'letter-mask.webm'),
    ], stdin=subprocess.PIPE)
    try:
        for frame in range(FPS * SECONDS):
            phase = frame / (FPS * SECONDS) * math.tau
            x = round((WIDTH - word.width) / 2 + 125 * math.sin(phase))
            y = round((HEIGHT - word.height) / 2 + 120 * math.sin(phase * 2))
            selector = background.copy()
            selector.paste(inside.crop((x, y, x + word.width, y + word.height)), (x, y), word)
            if frame == 0:
                selector.save(OUT / 'letter-mask.png')
            encoder.stdin.write(selector.tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit('Mask encoding failed.')

    # H.264 fallback for browsers without WebM support. Keep this in the
    # widely-decodable High/yuv420 profile; the browser compositor samples the
    # generated canvas (the actual ID field), not this fallback clock source.
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-i', str(OUT / 'letter-mask.webm'),
        '-an', '-c:v', 'libx264', '-crf', '1', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'letter-mask.mp4'),
    ], check=True)

    for name, source in [('spinstack', 'spinstack_demo.gif'), ('digits', 'digit_animation.gif'), ('gravity', '3d_grav_b_3.gif')]:
        subprocess.run([
            'ffmpeg', '-y', '-v', 'error', '-i', str(ROOT / 'public/media' / source),
            '-an', '-vf', 'scale=640:-2,fps=30', '-c:v', 'libx264', '-crf', '21',
            '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(OUT / f'{name}.mp4'),
        ], check=True)
    for path in sorted(OUT.iterdir()):
        print(f'{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
