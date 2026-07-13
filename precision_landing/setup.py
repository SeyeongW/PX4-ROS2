import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'precision_landing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seyeong',
    maintainer_email='seyeong186@gmail.com',
    description='ArUco precision landing controller (ArduPilot/MAVROS, GUIDED).',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'precision_landing_node = precision_landing.precision_landing_node:main',
            'follow_precland_node = precision_landing.follow_precland_node:main',
            'gimbal_down_node = precision_landing.gimbal_down_node:main',
            'precland_hw_node = precision_landing.precland_hw_node:main',
            'moving_marker_node = precision_landing.moving_marker_node:main',
            'track_follower_node = precision_landing.track_follower_node:main',
            'mission_manager_node = precision_landing.mission_manager_node:main',
        ],
    },
)
