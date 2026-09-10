"""TrueType text rendering for video overlays.

OpenCV's ``putText`` only has the Hershey fonts — single-stroke vector shapes
from the 1960s, with no real weight and no kerning. A 100px percentage drawn
with them looks like a plotter made it. Pillow can draw any TrueType face onto
the same frames, which is the whole difference between a panel that looks
generated and one that looks designed.

Pillow is already an indirect dependency (matplotlib pulls it in), so this costs
no new package. If it is somehow missing, or no usable face is found, callers
fall back to Hershey rather than failing the run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# Ordered preference. Each entry is (path, face index within a .ttc, name).
# Indices are verified, not guessed: in HelveticaNeue.ttc face 1 is Bold, and in
# "Avenir Next.ttc" face 0 is Bold while face 1 is Bold *Italic* — picking by
# eye is how a panel ends up silently slanted.
FONT_CANDIDATES: tuple[tuple[str, int, str], ...] = (
    ("/System/Library/Fonts/Avenir Next.ttc", 0, "Avenir Next Bold"),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1, "Helvetica Neue Bold"),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0, "Arial Bold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, "DejaVu Sans Bold"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0, "Liberation Sans Bold"),
)


def _matplotlib_dejavu() -> tuple[str, int, str] | None:
    """DejaVu Sans Bold ships inside matplotlib — the portable last resort."""
    try:
        import matplotlib
    except ImportError:
        return None
    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return (str(path), 0, "DejaVu Sans Bold (bundled)") if path.is_file() else None


def resolve_font(spec: str = "auto", index: int | None = None) -> tuple[str, int, str] | None:
    """Pick a font face. Returns None to mean "use the OpenCV fallback".

    ``spec`` is "auto" (first available candidate), "opencv" (force Hershey), or
    an explicit path to a .ttf/.otf/.ttc.
    """
    try:
        from PIL import ImageFont  # noqa: F401
    except ImportError:
        return None

    if spec == "opencv":
        return None

    if spec != "auto":
        path = Path(spec).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"PANEL_FONT points at a missing file: {path}")
        return (str(path), index or 0, path.stem)

    for path, idx, name in FONT_CANDIDATES:
        if Path(path).is_file():
            return (path, index if index is not None else idx, name)
    return _matplotlib_dejavu()


@lru_cache(maxsize=2048)
def _patch(text: str, font_path: str, font_index: int, size: int,
           color: tuple[int, int, int]) -> np.ndarray:
    """One string as a tight RGBA patch, plus its ink offset.

    Cached because a 547-frame render only ever draws a few hundred distinct
    strings — the percentage takes about a hundred values and every label is
    constant — so this runs the text engine a few hundred times instead of
    thousands.
    """
    from PIL import Image, ImageDraw, ImageFont

    face = ImageFont.truetype(font_path, size, index=font_index)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=face)
    w, h = max(1, right - left), max(1, bottom - top)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-left, -top), text, font=face, fill=(*color[::-1], 255))
    return np.array(img)


def measure(text: str, font, size: int) -> tuple[int, int]:
    """Ink width and height of *text* at *size*, in pixels."""
    if font is None:
        import cv2
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 2)
        return w, h
    patch = _patch(text, font[0], font[1], size, (255, 255, 255))
    return patch.shape[1], patch.shape[0]


def blend(img: np.ndarray, patch: np.ndarray, x: int, y: int, *,
          opacity: float = 1.0) -> np.ndarray:
    """Alpha-composite an RGBA patch onto a BGR frame, in place, clipped."""
    ph, pw = patch.shape[:2]
    fh, fw = img.shape[:2]

    # Clip rather than shift: a label that runs past the edge should lose its
    # tail, not silently jump left and overlap its neighbour.
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + pw), min(fh, y + ph)
    if x0 >= x1 or y0 >= y1:
        return img

    sub = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1]
    alpha = (sub[..., 3:4].astype(np.float32) / 255.0) * opacity
    rgb = sub[..., 2::-1].astype(np.float32)   # RGBA -> BGR
    img[y0:y1, x0:x1] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    return img


def draw(img: np.ndarray, text: str, font, *, size: int, xy: tuple[int, int],
         color: tuple[int, int, int] = (255, 255, 255), anchor: str = "lt",
         opacity: float = 1.0, rotate: int = 0) -> np.ndarray:
    """Draw *text* with its *anchor* corner at *xy*.

    *anchor* is two characters, horizontal then vertical, from ``l/c/r`` and
    ``t/m/b`` — so ``"ct"`` centers horizontally and hangs from the top. Anchors
    are computed from the glyphs' **ink box**, not the font's line box, so a
    centered number is optically centered rather than centered-with-leading.
    """
    if font is None:
        import cv2
        scale = size / 30
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x, y = xy
        x -= {"l": 0, "c": tw // 2, "r": tw}[anchor[0]]
        y += {"t": th, "m": th // 2, "b": 0}[anchor[1]]
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return img

    patch = _patch(text, font[0], font[1], size, color)
    if rotate:
        # Rotating the rendered patch is how a vertical axis label gets drawn —
        # Pillow has no rotated-text primitive, and the anchor maths below then
        # works off the rotated box rather than the original one.
        patch = np.rot90(patch, k=(rotate // 90) % 4)
    ph, pw = patch.shape[:2]
    x, y = xy
    x -= {"l": 0, "c": pw // 2, "r": pw}[anchor[0]]
    y -= {"t": 0, "m": ph // 2, "b": ph}[anchor[1]]
    return blend(img, patch, x, y, opacity=opacity)
