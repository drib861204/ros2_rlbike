from setuptools import setup
import os
from glob import glob

package_name = 'rlbike'
submodules = 'rlbike/files'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*_launch.py')) 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='drib861204@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'train = rlbike.train:main',
		        'motor_bridge = rlbike.motor_bridge:main',
		        'imu_bridge = rlbike.imu_bridge:main'
        ],
    },
)
