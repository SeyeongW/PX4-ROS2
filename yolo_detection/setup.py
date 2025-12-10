from setuptools import find_packages, setup

package_name = 'yolo_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seyeong',
    maintainer_email='seyeong186@gmail.com',
    description='YOLO Detection and SIYI Gimbal Control',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera = yolo_detection.siyi_camera:main',
            'tracker = yolo_detection.yolo_tracker:main',
        ],
    },
)