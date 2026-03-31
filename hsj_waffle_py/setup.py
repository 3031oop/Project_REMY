from setuptools import setup
import os
from glob import glob

package_name = 'hsj_waffle_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='remi',
    description='YOLOv8 NCNN(Python)',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node = hsj_waffle_py.yolo_state_node:main'
        ],
    },
)
