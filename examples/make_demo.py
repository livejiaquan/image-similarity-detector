"""Generate synthetic review fixtures; no company images or model weights required."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_demo(destination: Path) -> None:
    # Refuse existing files so a demo command cannot overwrite a user's dataset.
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    train, test = destination / "train", destination / "test"
    train.mkdir(parents=True)
    test.mkdir()
    for index in range(4):
        image = Image.new("RGB", (640, 400), (222, 231, 241))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 260, 640, 400), fill=(164, 183, 201))
        draw.line((0, 260, 640, 260), fill=(113, 133, 157), width=3)
        rng = np.random.default_rng(100 + index)
        colors = [(67, 100, 171), (205, 137, 59), (63, 141, 125), (143, 100, 170)]
        for _ in range(5):
            x = int(rng.integers(40, 500))
            y = int(rng.integers(45, 210))
            width = int(rng.integers(50, 120))
            height = int(rng.integers(45, 120))
            draw.rounded_rectangle(
                (x, y, x + width, y + height),
                radius=6,
                fill=colors[index],
                outline=(44, 61, 85),
                width=3,
            )
            draw.line((x + 8, y + 12, x + width - 8, y + 12), fill=(238, 244, 250), width=3)
        draw.text((24, 18), f"SYNTHETIC INSPECTION FRAME / {index + 1:02}", fill=(37, 55, 80))
        image.save(train / f"frame-{index + 1:02}.png")
        if index < 2:
            shutil.copyfile(train / f"frame-{index + 1:02}.png", test / f"copy-{index + 1:02}.png")
            image.save(test / f"compressed-{index + 1:02}.jpg", quality=82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo-data"))
    create_demo(parser.parse_args().output)
