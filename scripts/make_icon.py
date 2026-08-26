#!/usr/bin/env python3
"""
scripts/make_icon.py

Generate the application icon.

The icon is code rather than a checked-in binary so it can be read, changed and
regenerated. It also removes a dependency: rendering is done here with plain
zlib and arithmetic, so building it needs nothing beyond a Python install and
the `iconutil` that ships with macOS.

    python scripts/make_icon.py

Writes src/resources/images/icon.icns, which launcher_macos.py copies into the
bundle it creates. Before this existed the launcher looked for that file, did
not find it, and the app showed up in Launchpad with a blank icon.

Design notes, such as they are: a level meter, because that is what the app is
doing and it survives being drawn at 16 pixels, which is where Finder lists and
the menu bar will show it.

Three bars, not the five or seven a meter would naturally have. Five was tried
and fails at the smallest size: the gaps are under a pixel once downscaled, the
bars bleed together, and the icon reads as a pale blob. Legibility at 16px
decides this, not how it looks at 1024.

Antialiasing evaluates a signed distance per pixel rather than supersampling --
sharper, and about ten times quicker in pure Python.
"""

import math
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

SIZE = 1024

# Deep blue to violet. Dark enough that the white meter carries the shape, and
# distinguishable from the blue-grey macOS fills it will sit next to.
TOP = (37, 74, 194)
BOTTOM = (98, 44, 176)

CORNER = 0.2237 * SIZE          # what macOS uses for its own rounded squares

# Bar geometry, as fractions of the canvas.
BAR_COUNT = 3
BAR_W = 0.150
BAR_GAP = 0.105
BAR_HEIGHTS = (0.42, 0.74, 0.42)


def rounded_rect_distance(x, y, cx, cy, half_w, half_h, radius):
    """Signed distance to a rounded rectangle; negative inside."""
    dx = abs(x - cx) - (half_w - radius)
    dy = abs(y - cy) - (half_h - radius)
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside - radius


def coverage(distance, softness=1.0):
    """Antialiased 0..1 coverage from a signed distance."""
    return min(1.0, max(0.0, 0.5 - distance / softness))


def render():
    half = SIZE / 2
    bar_w = BAR_W * SIZE
    gap = BAR_GAP * SIZE
    span = BAR_COUNT * bar_w + (BAR_COUNT - 1) * gap
    first_cx = half - span / 2 + bar_w / 2

    bars = []
    for i, height_frac in enumerate(BAR_HEIGHTS):
        bars.append((first_cx + i * (bar_w + gap),
                     height_frac * SIZE / 2))

    rows = []
    for y in range(SIZE):
        row = bytearray()
        # Vertical gradient, computed once per row.
        t = y / (SIZE - 1)
        base = tuple(int(round(TOP[c] + (BOTTOM[c] - TOP[c]) * t))
                     for c in range(3))
        for x in range(SIZE):
            px, py = x + 0.5, y + 0.5

            outside = 1.0 - coverage(
                rounded_rect_distance(px, py, half, half, half, half, CORNER))
            if outside >= 1.0:
                row += b"\x00\x00\x00\x00"          # fully outside the squircle
                continue

            r, g, b = base
            # The meter, in white, over the gradient.
            for cx, half_h in bars:
                d = rounded_rect_distance(px, py, cx, half, bar_w / 2,
                                          half_h, bar_w / 2)
                ink = coverage(d)
                if ink > 0.0:
                    r = int(round(r + (255 - r) * ink))
                    g = int(round(g + (255 - g) * ink))
                    b = int(round(b + (255 - b) * ink))

            alpha = int(round((1.0 - outside) * 255))
            row += bytes((r, g, b, alpha))
        rows.append(bytes(row))
    return rows


def write_png(path: Path, rows) -> None:
    """Minimal RGBA PNG. No dependency worth adding for this."""
    raw = b"".join(b"\x00" + row for row in rows)

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def build_icns(master: Path, out: Path) -> None:
    """Downscale into an iconset and let macOS assemble the .icns."""
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale, suffix in ((1, ""), (2, "@2x")):
                pixels = size * scale
                target = iconset / f"icon_{size}x{size}{suffix}.png"
                subprocess.run(
                    ["sips", "-z", str(pixels), str(pixels), str(master),
                     "--out", str(target)],
                    check=True, capture_output=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset),
                        "-o", str(out)], check=True, capture_output=True)


def main() -> int:
    if sys.platform != "darwin":
        print("iconutil is macOS-only; the .icns is committed, so this only "
              "needs running when the design changes.")
        return 0

    root = Path(__file__).resolve().parent.parent
    out = root / "src" / "resources" / "images" / "icon.icns"

    print(f"Rendering {SIZE}x{SIZE}...")
    rows = render()

    with tempfile.TemporaryDirectory() as tmp:
        master = Path(tmp) / "icon.png"
        write_png(master, rows)
        print("Building .icns...")
        build_icns(master, out)

    print(f"Wrote {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
