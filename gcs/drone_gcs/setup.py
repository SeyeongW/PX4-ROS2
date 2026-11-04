from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_NAME = "drone_gcs"


def map_pack_data_files():
    """Install every baked map pack under share/, preserving its directory.

    Packs are data the operator swaps, so they are installed whole rather than
    listed file by file — dropping a new `maps/<name>/` in and rebuilding is all
    it takes to add a map.
    """
    out = []
    for pack in sorted(Path("maps").glob("*/map.yaml")):
        files = [str(p) for p in sorted(pack.parent.iterdir()) if p.is_file()]
        out.append((f"share/{PACKAGE_NAME}/maps/{pack.parent.name}", files))
    return out


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ]
    + map_pack_data_files(),
    install_requires=["setuptools", "numpy", "Pillow", "PyYAML"],
    zip_safe=True,
    maintainer="seyeong",
    maintainer_email="seyeong186@gmail.com",
    description=(
        "Ground-station GUI for the city obstacle-avoidance and moving-trailer "
        "precision-landing missions: swappable map packs, live planner path "
        "overlay, camera panel."
    ),
    license="BSD-3-Clause",
    tests_require=["pytest"],
)
