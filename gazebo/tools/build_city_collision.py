#!/usr/bin/env python3
"""Build a deterministic Bullet-compatible city terrain collision mesh."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from PIL import Image


GAZEBO = Path(__file__).resolve().parents[1]
HEIGHTMAP = GAZEBO / "worlds/city_map/mesh/height_map_city_500m.png"
OUTPUT = GAZEBO / "worlds/city_map/mesh/city_terrain_collision.obj"
EXPECTED_HEIGHT_SHA256 = (
    "5a84adc1f45dcffe507fa77d2642cd672c622c225e60ae41f98280bdcf9b24cf"
)
SOURCE_PIXELS = 257
STRIDE = 2
GRID_PIXELS = 129
SIZE_X = 500.0
SIZE_Y = 500.0
SIZE_Z = 26.6
POSITION_Z = -15.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: float) -> str:
    if abs(value) < 5e-13:
        return "0"
    return f"{value:.9f}".rstrip("0").rstrip(".")


def main() -> None:
    if sha256(HEIGHTMAP) != EXPECTED_HEIGHT_SHA256:
        raise SystemExit("city heightmap hash changed")
    with Image.open(HEIGHTMAP) as source:
        if source.mode != "L" or source.size != (SOURCE_PIXELS, SOURCE_PIXELS):
            raise SystemExit(f"unexpected city heightmap: {source.mode} {source.size}")
        values = list(source.getdata())

    lines = [
        "# Deterministic 129 x 129 collision grid for Apple Park city",
        "# x/y = -250..250 m; z = -15 + uint8/255 * 26.6 m",
        "o city_terrain_collision",
    ]
    for row in range(GRID_PIXELS):
        source_row = row * STRIDE
        y = SIZE_Y / 2.0 - row / (GRID_PIXELS - 1) * SIZE_Y
        for column in range(GRID_PIXELS):
            source_column = column * STRIDE
            x = -SIZE_X / 2.0 + column / (GRID_PIXELS - 1) * SIZE_X
            pixel = values[source_row * SOURCE_PIXELS + source_column]
            z = POSITION_Z + pixel / 255.0 * SIZE_Z
            lines.append(f"v {number(x)} {number(y)} {number(z)}")

    lines.append("s off")
    for row in range(GRID_PIXELS - 1):
        for column in range(GRID_PIXELS - 1):
            top_left = row * GRID_PIXELS + column + 1
            top_right = top_left + 1
            bottom_left = top_left + GRID_PIXELS
            bottom_right = bottom_left + 1
            lines.append(f"f {top_left} {bottom_left} {top_right}")
            lines.append(f"f {top_right} {bottom_left} {bottom_right}")
    content = "\n".join(lines) + "\n"

    descriptor, name = tempfile.mkstemp(
        dir=OUTPUT.parent, prefix=f".{OUTPUT.name}.", suffix=".tmp", text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"output={OUTPUT}")
    print(f"vertices={GRID_PIXELS * GRID_PIXELS}")
    print(f"triangles={(GRID_PIXELS - 1) * (GRID_PIXELS - 1) * 2}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
