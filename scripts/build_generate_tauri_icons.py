#!/usr/bin/env python3
"""Generate Tauri app and tray icons from a single logo source."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / ".archive" / "informity-logo-white.png"
ICONS_DIR = ROOT / "src/frontend" / "src-tauri" / "icons"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tauri icon assets.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the source logo image (default: .archive/informity-logo.png).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=40,
        help="Luma threshold used to extract black logo pixels from source.",
    )
    parser.add_argument(
        "--app-logo-scale",
        type=float,
        default=0.76,
        help="Relative logo size for app icon glyph (0.0-1.0).",
    )
    parser.add_argument(
        "--tray-scale",
        type=float,
        default=0.88,
        help="Relative logo size for trayTemplate.png (0.0-1.0).",
    )
    parser.add_argument(
        "--skip-tray",
        action="store_true",
        help="Do not regenerate trayTemplate.png.",
    )
    parser.add_argument(
        "--dmg-corner-radius",
        type=int,
        default=220,
        help="Corner radius for DMG volume icon mask at 1024px.",
    )
    return parser.parse_args()


def logo_mask_from_source(source_path: Path, threshold: int) -> Image.Image:
    source = Image.open(source_path).convert("RGB")
    width, height = source.size
    # If source includes wordmark text (very wide image), use the left square logo mark.
    if width > int(height * 1.2):
        source = source.crop((0, 0, height, height))
    gray = source.convert("L")
    mask = gray.point(lambda p: 255 if p < threshold else 0, mode="L")
    # If strict threshold yields empty output (e.g., bright colored logo),
    # fallback to a permissive cutoff so generation still succeeds.
    if mask.getbbox() is None:
        mask = gray.point(lambda p: 255 if p < 220 else 0, mode="L")
    bbox = mask.getbbox()
    if bbox is None:
        raise RuntimeError(f"No logo shape found in {source_path}")
    return mask.crop(bbox)


def resize_binary(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    resized = mask.resize(size, Image.Resampling.LANCZOS)
    return resized.point(lambda p: 255 if p >= 128 else 0, mode="L")


def build_app_icon(
    logo_mask: Image.Image,
    size: int,
    logo_scale: float = 0.76,
) -> Image.Image:
    logo_size = max(1, round(size * logo_scale))
    # Full-bleed square artboard; let macOS apply final corner shaping.
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    symbol_mask = resize_binary(logo_mask, (logo_size, logo_size))
    symbol = Image.new("RGBA", (logo_size, logo_size), (0, 0, 0, 255))
    offset = ((size - logo_size) // 2, (size - logo_size) // 2)
    image.paste(symbol, offset, symbol_mask)
    return image


def build_tray_icon(logo_mask: Image.Image, size: int = 64, scale: float = 0.88) -> Image.Image:
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"Invalid tray scale {scale}; expected 0.0 < scale <= 1.0")
    logo_size = max(1, round(size * scale))
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    symbol_mask = logo_mask.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
    symbol = Image.new("RGBA", (logo_size, logo_size), (0, 0, 0, 255))
    offset = ((size - logo_size) // 2, (size - logo_size) // 2)
    image.paste(symbol, offset, symbol_mask)
    return image


def build_dmg_volume_icon(app_icon_1024: Image.Image, corner_radius: int = 220) -> Image.Image:
    if app_icon_1024.size != (1024, 1024):
        raise ValueError("DMG volume icon expects a 1024x1024 app icon input")
    if corner_radius <= 0:
        raise ValueError(f"Invalid DMG corner radius {corner_radius}; expected > 0")

    mask = Image.new("L", (1024, 1024), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, 1023, 1023), radius=corner_radius, fill=255)

    rounded = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    rounded.paste(app_icon_1024, (0, 0), mask)
    return rounded


def write_png(icon: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    icon.save(path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Logo source not found: {source_path}")

    logo_mask = logo_mask_from_source(source_path, threshold=args.threshold)

    app_sizes = {
        "icon-1024.png": 1024,
        "icon.png": 512,
        "32x32.png": 32,
        "64x64.png": 64,
        "128x128.png": 128,
        "128x128@2x.png": 256,
        "Square30x30Logo.png": 30,
        "Square44x44Logo.png": 44,
        "Square71x71Logo.png": 71,
        "Square89x89Logo.png": 89,
        "Square107x107Logo.png": 107,
        "Square142x142Logo.png": 142,
        "Square150x150Logo.png": 150,
        "Square284x284Logo.png": 284,
        "Square310x310Logo.png": 310,
        "StoreLogo.png": 50,
    }

    iconset_sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    required_sizes = set(app_sizes.values()) | set(iconset_sizes.values()) | {24, 48}
    rendered: dict[int, Image.Image] = {}
    for size in sorted(required_sizes):
        rendered[size] = build_app_icon(logo_mask, size=size, logo_scale=args.app_logo_scale)

    for filename, size in app_sizes.items():
        write_png(rendered[size], ICONS_DIR / filename)

    iconset_dir = ICONS_DIR / "icon.iconset"
    for filename, size in iconset_sizes.items():
        write_png(rendered[size], iconset_dir / filename)

    if not args.skip_tray:
        tray_icon = build_tray_icon(logo_mask, size=64, scale=args.tray_scale)
        write_png(tray_icon, ICONS_DIR / "trayTemplate.png")

    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    rendered[256].save(ICONS_DIR / "icon.ico", format="ICO", sizes=ico_sizes)

    rendered[1024].save(
        ICONS_DIR / "icon.icns",
        format="ICNS",
        sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
    )

    dmg_icon_1024 = build_dmg_volume_icon(
        rendered[1024],
        corner_radius=args.dmg_corner_radius,
    )
    write_png(dmg_icon_1024, ICONS_DIR / "dmg-volume-icon-1024.png")
    dmg_icon_1024.save(
        ICONS_DIR / "dmg-volume-icon.icns",
        format="ICNS",
        sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
    )

    print(f"Generated Tauri icons from: {source_path}")
    print(f"Updated app logo scale: {args.app_logo_scale:.2f}")
    print(f"Updated DMG icon corner radius: {args.dmg_corner_radius}")
    if args.skip_tray:
        print("Tray icon preserved (not regenerated).")
    else:
        print(f"Updated tray icon scale: {args.tray_scale:.2f}")


if __name__ == "__main__":
    main()
