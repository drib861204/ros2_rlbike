from setuptools import setup

package_name = 'rlbike'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
                'run = rlbike.run:main',
		'talker = rlbike.publisher_member_function:main',
		'listener = rlbike.subscriber_member_function:main',
		'motor_bridge = rlbike.motor_bridge:main'
        ],
    },
)
