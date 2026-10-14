#!/usr/bin/env python3
"""Generate a Rubik's-cube sticker texture (3x3 colored stickers on black) and a
per-face cubemap-style sheet. Used as the cube material in the Isaac playground.

Writes isaac/assets/rubik_face_<color>.png (one per standard face color) plus a single
isaac/assets/rubik_sticker.png used as a uniform fallback. The textures already ship with
the release; regenerate with any Python that has Pillow:  python isaac/make_rubik_texture.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

_ASSETS = Path(__file__).resolve().parent / "assets"

# Standard Rubik's face colors (name -> RGB).
FACE_COLORS = {
    "white": (245, 245, 245),
    "yellow": (255, 213, 0),
    "red": (196, 30, 58),
    "orange": (255, 88, 0),
    "blue": (0, 81, 186),
    "green": (0, 158, 96),
}


def make_face(color, size=512, border_frac=0.06, sticker_frac=0.86):
    """A single Rubik's face: 3x3 stickers of `color` on a black plastic body."""
    img = Image.new("RGB", (size, size), (15, 15, 15))
    d = ImageDraw.Draw(img)
    cell = size / 3.0
    pad = cell * (1.0 - sticker_frac) / 2.0
    for r in range(3):
        for c in range(3):
            x0 = c * cell + pad
            y0 = r * cell + pad
            x1 = (c + 1) * cell - pad
            y1 = (r + 1) * cell - pad
            d.rounded_rectangle([x0, y0, x1, y1], radius=cell * 0.12,
                                fill=color, outline=(10, 10, 10),
                                width=max(2, int(size * border_frac * 0.15)))
    return img


def main():
    _ASSETS.mkdir(parents=True, exist_ok=True)
    for name, rgb in FACE_COLORS.items():
        make_face(rgb).save(_ASSETS / f"rubik_face_{name}.png")
    # A neutral multi-color sticker sheet as a uniform fallback texture.
    make_face((230, 230, 230)).save(_ASSETS / "rubik_sticker.png")
    print(f"wrote {len(FACE_COLORS) + 1} textures to {_ASSETS}")


if __name__ == "__main__":
    main()
