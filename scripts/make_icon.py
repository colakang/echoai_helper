#!/usr/bin/env python3
"""
scripts/make_icon.py

Build the application icon from the EchoAI brand mark.

    .venv/bin/python scripts/make_icon.py [path/to/logo.png]

Writes src/resources/images/icon.icns, which launcher_macos.py copies into the
bundle it creates. Before this existed the launcher looked for that file, did
not find it, and the app appeared in Launchpad blank.

Two things the brand artwork cannot do unaltered, and this script fixes:

**The wordmark has to go.** The source logo sets "echoAi 365" beneath the
headset. At the sizes macOS actually draws an app icon -- 16 and 32 pixels in
Finder lists, the menu bar and the window switcher -- that text is two or three
pixels tall and becomes a grey smear. Apple's guidance is blunt about this and
it is right. So the mark is cropped out and the type left behind.

**The frame has to become a squircle.** The artwork is a full-bleed square. A
square icon among rounded ones reads as broken, so the mark is composited onto
a rounded rectangle at the corner radius macOS uses for its own.

**Detail is dropped at the small sizes rather than scaled.** The circuit
filigree inside the headset is lovely at 512 and turns to mud at 16, so the
smallest slices get the headset alone, recoloured bright enough to separate
from the plate. That is not a compromise: an .icns is a set of images, not one
image resized, and building it that way is how Apple's own icons work.

Every claim above was checked by decomposing the finished .icns with
`iconutil -c iconset` and looking at the actual 16x16 file. Resizing the icns
with `sips` does not show you that -- it downsamples the largest slice and
hides exactly the problem you are looking for.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    sys.exit("This needs Pillow:  uv pip install --python .venv/bin/python pillow\n"
             "It is a build-time tool only; the app itself does not import it.")

SIZE = 1024

# Sampled from the artwork, so the plate and the mark are the same navy.
BACKGROUND = (9, 14, 45)

CORNER = 0.2237                 # macOS uses this fraction of the icon's width

# Where the mark sits in a 1024x1024 source, and how much of the plate it fills.
MARK_BOX = (262, 175, 762, 645)
MARK_SCALE = 0.76
# The small mark has no detail to carry it, so it leans on size instead.
SMALL_MARK_SCALE = 0.90

# Below this pixel size the filigree is mud; use the simplified mark instead.
SIMPLIFY_BELOW = 48


def rounded_plate(size: int, radius_frac: float, colour) -> Image.Image:
    """A macOS-shaped rounded square, antialiased by rendering large."""
    scale = 4
    big = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    draw.rounded_rectangle(
        [(0, 0), (size * scale - 1, size * scale - 1)],
        radius=int(radius_frac * size * scale),
        fill=colour + (255,))
    return big.resize((size, size), Image.LANCZOS)


def extract_mark(logo: Image.Image) -> Image.Image:
    """
    The headset, cut out of its background and without the wordmark.

    The artwork is fully opaque -- a navy rectangle with the mark painted on
    it -- so cropping alone yields a navy tile, not a mark. Pasting that over a
    navy plate happens to look right, which is exactly the kind of accident
    that survives until someone changes the plate colour or brightens the mark
    and the whole tile lights up.

    So the background is keyed out by luminance. The field is very dark and
    every part of the mark is not, which makes the separation clean and needs
    no mask to be drawn by hand.
    """
    mark = logo.crop(MARK_BOX).convert("RGBA")
    luma = mark.convert("L")
    # Below FLOOR is background, above CEILING is solidly mark; between them
    # is the antialiased edge, and keeping that ramp is what stops the cut-out
    # looking like it was done with scissors.
    floor, ceiling = 26, 90
    alpha = luma.point(
        lambda v: 0 if v <= floor else
        (255 if v >= ceiling else int((v - floor) * 255 / (ceiling - floor))))
    r, g, b, _ = mark.split()
    return Image.merge("RGBA", (r, g, b, alpha))


def simplify(mark: Image.Image) -> Image.Image:
    """
    Make the mark survive being drawn at 16 pixels.

    Two separate problems, and the second is the one that matters. Blurring
    away the circuit filigree is easy. The hard part is that this artwork is
    dark-on-dark by design -- a mid-blue headset on a navy field -- which reads
    beautifully at 512 where the detail carries it, and at 16 collapses into a
    single dark smudge because there is almost no figure-to-ground contrast
    left once the detail is gone.

    So the small mark is lifted towards the brand's own cyan until it separates
    from the plate. It is the same shape and the same hue family; it is simply
    bright enough to be seen at a size where nothing else is.
    """
    blurred = mark.filter(ImageFilter.GaussianBlur(radius=mark.width * 0.018))
    r, g, b, a = blurred.split()
    # Firm up what the blur softened, so the headset comes back solid while the
    # filigree -- which the blur has already thinned to nothing -- stays gone.
    a = a.point(lambda v: 0 if v < 90 else min(255, int((v - 90) * 3.0)))
    # Recolour to the logo's own highlight cyan. The shape is unchanged; it is
    # simply bright enough to separate from the plate at a size where detail
    # cannot do that job.
    r = r.point(lambda _: 122)
    g = g.point(lambda _: 232)
    b = b.point(lambda _: 216)
    return Image.merge("RGBA", (r, g, b, a))


def compose(mark: Image.Image, size: int, scale: float = MARK_SCALE) -> Image.Image:
    plate = rounded_plate(size, CORNER, BACKGROUND)
    target = int(size * scale)
    scaled = mark.resize((target, target), Image.LANCZOS)
    offset = ((size - target) // 2, (size - target) // 2)
    plate.alpha_composite(scaled, offset)
    return plate


def main() -> int:
    if sys.platform != "darwin":
        print("iconutil is macOS-only; the .icns is committed, so this only "
              "needs running when the artwork changes.")
        return 0

    root = Path(__file__).resolve().parent.parent
    default_logo = root / "docs" / "images" / "logo.png"
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else default_logo
    if not source.exists():
        return print(f"No artwork at {source}") or 1

    out = root / "src" / "resources" / "images" / "icon.icns"
    logo = Image.open(source).convert("RGBA")
    if logo.size != (SIZE, SIZE):
        logo = logo.resize((SIZE, SIZE), Image.LANCZOS)

    mark = extract_mark(logo)
    small_mark = simplify(mark)

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            for retina, suffix in ((1, ""), (2, "@2x")):
                pixels = base * retina
                small = pixels < SIMPLIFY_BELOW
                compose(small_mark if small else mark, pixels,
                        SMALL_MARK_SCALE if small else MARK_SCALE).save(
                    iconset / f"icon_{base}x{base}{suffix}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
                       check=True, capture_output=True)

    print(f"Wrote {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
