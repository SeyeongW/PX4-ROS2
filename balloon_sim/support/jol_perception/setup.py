# ROS 패키지 설치 파일과 ros2 run 실행 이름을 등록한다.
from glob import glob
from setuptools import setup

setup(
    name='jol_perception', version='0.1.0', packages=['jol_perception'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/jol_perception']),
        ('share/jol_perception', ['package.xml']),
        ('share/jol_perception/launch', ['launch/monitor.launch.py']),
        ('share/jol_perception/config', glob('config/*')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='xogus', maintainer_email='xogus@example.com',
    description='LiDAR multi-object monitoring with calibrated camera fusion',
    license='Apache-2.0',
    # This package is the bundled support library; balloon_locator owns entrypoints.
    entry_points={},
)
