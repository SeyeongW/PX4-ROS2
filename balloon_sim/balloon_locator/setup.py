from glob import glob

from setuptools import find_packages, setup

package_name = 'balloon_locator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.json')),
        ('share/' + package_name + '/meshes/stl_airframe', glob('meshes/stl_airframe/*.stl')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seyeongw',
    maintainer_email='seyeong186@gmail.com',
    description='SIYI A8 mini 짐벌 카메라 기반 정지 풍선 탐지 및 NED 좌표 추정',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stl_static_flight = balloon_locator.stl_runtime:main_mission',
            'stl_mavros_adapter = balloon_locator.stl_runtime:main_adapter',
            'stl_experiment_ui = balloon_locator.stl_runtime:main_ui',
            'hardware_check = balloon_locator.hardware_check:main',
            'gazebo_bridge = balloon_locator.gazebo_bridge:main',
            'siyi_gimbal_node = balloon_locator.siyi_gimbal_node:main',
            'siyi_camera_node = balloon_locator.siyi_camera_node:main',
            'balloon_detector_node = balloon_locator.balloon_detector_node:main',
            'gimbal_tracker_node = balloon_locator.gimbal_tracker_node:main',
            'balloon_locator_node = balloon_locator.balloon_locator_node:main',
            'mavlink_bridge_node = balloon_locator.mavlink_bridge_node:main',
        ],
    },
)
