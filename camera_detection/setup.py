import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'camera_detection'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_processor_node = camera_detection.yolo_processor_node:main',
            'yolo_processor_sim_node = camera_detection.yolo_processor_sim_node:main',
            'commander = camera_detection.commander:main'
        ],
    },
)