from setuptools import setup


package_name = "experiment_logger"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="seyeongW",
    maintainer_email="seyeong186@gmail.com",
    description="Read-only common experiment metrics logger for JO flights.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "experiment_logger_node = "
            "experiment_logger.experiment_logger_node:main",
        ],
    },
)
