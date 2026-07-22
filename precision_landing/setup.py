import os
from setuptools import find_packages, setup

package_name = 'precision_landing'
console_scripts = [
    'mpc_precision_landing_node = '
    'precision_landing.mpc_precision_landing_node:main',
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), [
            'launch/mpc_precision_landing.launch.py',
        ]),
        (os.path.join('share', package_name, 'config'), [
            'config/mpc_precision_landing.yaml',
            'config/mpc_precision_landing_real.yaml',
        ]),
    ],
    install_requires=[
        'setuptools',
        'osqp>=1.0,<2.0',
    ],
    zip_safe=True,
    maintainer='seyeong',
    maintainer_email='seyeong186@gmail.com',
    description=(
        'PX4/MAVROS ArUco landing with fail-closed descent and '
        'confirmed touchdown.'
    ),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': console_scripts,
    },
)
