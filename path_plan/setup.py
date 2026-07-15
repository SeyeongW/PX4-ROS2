from setuptools import find_packages, setup


PACKAGE_NAME = "path_plan"
setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch",
         ["launch/path_plan.launch.py", "launch/px4_mavros.launch.py"]),
        (f"share/{PACKAGE_NAME}/config", ["config/city_uav.yaml"]),
        (f"share/{PACKAGE_NAME}/rviz", ["rviz/path_plan.rviz"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "PyYAML"],
    zip_safe=True,
    maintainer="seyeong",
    maintainer_email="seyeong186@gmail.com",
    description=(
        "Global (A*/SFC/B-spline) and local (MPC + depth-camera) path "
        "generation for PX4 city UAV flight in the 20-30 m cruise band."
    ),
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "astar_node = path_plan.astar_node:main",
            "sfc_node = path_plan.sfc_node:main",
            "bspline_node = path_plan.bspline_node:main",
            "mpc_node = path_plan.mpc_node:main",
            "mavros_static_path = path_plan.mavros_static_path:main",
            "trailer_loop_driver = path_plan.trailer_loop_driver:main",
            "moving_land_node = path_plan.moving_land_node:main",
            "flight_logger = path_plan.flight_logger:main",
            "building_markers = path_plan.building_markers:main",
        ],
    },
)
