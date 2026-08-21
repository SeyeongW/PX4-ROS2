from setuptools import setup

package_name = 'trailer_link'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seyeongW',
    maintainer_email='seyeong186@gmail.com',
    description='Trailer coordinate pipeline: MAVLink radio -> drone-local ENU '
                'relative target for the landing MPC.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trailer_gps_node = trailer_link.trailer_gps_node:main',
            'trailer_target_node = trailer_link.trailer_target_node:main',
            'radio_probe = trailer_link.radio_probe:main',
        ],
    },
)
