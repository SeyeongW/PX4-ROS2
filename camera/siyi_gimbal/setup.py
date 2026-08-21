import os
from glob import glob

from setuptools import setup

package_name = 'siyi_gimbal'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seyeongW',
    maintainer_email='seyeong186@gmail.com',
    description='SIYI A8 mini gimbal: nadir on arm.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'siyi_gimbal_node = siyi_gimbal.siyi_gimbal_node:main',
            'gimbal_monitor = siyi_gimbal.gimbal_monitor:main',
            'gimbal_teleop = siyi_gimbal.gimbal_teleop:main',
        ],
    },
)
