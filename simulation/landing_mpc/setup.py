import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'landing_mpc'

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
    description='Relative-coordinate MPC for precision landing on a moving '
                'ArUco target (PX4 SITL Offboard).',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'aruco_detector_node = landing_mpc.aruco_detector_node:main',
            'marker_tf_node = landing_mpc.marker_tf_node:main',
            'marker_kf_node = landing_mpc.marker_kf_node:main',
            'trailer_cue_node = landing_mpc.trailer_cue_node:main',
            'gimbal_control_node = landing_mpc.gimbal_control_node:main',
            'mission_manager_node = landing_mpc.mission_manager_node:main',
        ],
    },
)
