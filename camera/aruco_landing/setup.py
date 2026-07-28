import os
from glob import glob

from setuptools import setup

package_name = 'aruco_landing'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seyeongW',
    maintainer_email='seyeong186@gmail.com',
    description='Real-vehicle ArUco perception for precision landing.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_pose_node = aruco_landing.aruco_pose_node:main',
        ],
    },
)
