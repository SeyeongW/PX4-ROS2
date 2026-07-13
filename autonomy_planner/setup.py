from setuptools import find_packages, setup


PACKAGE_NAME = "autonomy_planner"
setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/autonomy_planner"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (
            f"share/{PACKAGE_NAME}/config",
            ["config/city_mission.yaml", "config/city_uav_1m_mission.yaml", "config/trailer_motion.yaml"],
        ),
        (
            f"share/{PACKAGE_NAME}/launch",
            ["launch/city_autonomy.launch.py", "launch/trailer_motion.launch.py"],
        ),
        (
            f"share/{PACKAGE_NAME}/maps",
            [
                "../gazebo/maps/city_coordinates.yaml",
                "../gazebo/maps/city_coordinates_uav.yaml",
            ],
        ),
        (
            f"share/{PACKAGE_NAME}/docs",
            [
                "../docs/px4_l1_bspline_cross_validation.md",
                "../docs/uav_global_local_planning_architecture.md",
            ],
        ),
    ],
    install_requires=["setuptools", "numpy", "scipy", "PyYAML"],
    # pymavlink is deliberately optional: only the mutually-exclusive native
    # PX4 AUTO.MISSION experiment needs it.  The normal B-spline/MPC ROS 2 node
    # must remain installable and runnable without a second guidance transport.
    extras_require={"px4_l1": ["pymavlink>=2.4.40"]},
    zip_safe=True,
    maintainer="seyeong",
    maintainer_email="seyeong186@gmail.com",
    description=(
        "Map-aware PX4 city mission using A*, a safe flight corridor, "
        "B-spline/local MPC, plus a mutually-exclusive native PX4 "
        "AUTO.MISSION L1-style cross-validation adapter."
    ),
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "city_autonomy = autonomy_planner.mission_node:main",
            "mission_cli = autonomy_planner.mission_cli:main",
            "plan_city_path = autonomy_planner.offline_plan:main",
            "screen_city_plan_3d = autonomy_planner.planning_screening:main",
            "export_px4_l1_plan = autonomy_planner.px4_l1_export:main",
            "px4_l1_mission = autonomy_planner.px4_l1_mission:main",
            "trailer_motion = autonomy_planner.trailer_motion_node:main",
        ],
    },
)
