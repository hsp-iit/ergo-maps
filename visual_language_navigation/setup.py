from setuptools import find_packages, setup

package_name = 'visual_language_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, package_name + '.*']),
    #packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/build_map.launch.py',
            'launch/semantic_map_server.launch.py',
            'launch/rviz_vlnav.launch.py'
        ]),
        ('share/' + package_name + '/params', [
            'params/mapping_params.yaml',
            'params/semantic_map_server_params.yaml'
        ]),
        ('share/' + package_name + '/params/rviz', [
            'params/rviz/vlnav.rviz'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Simone Micheletti',
    maintainer_email='simone_micheletti@outlook.it',
    description='TODO: Package description',
    license='BSD 3-Clause License',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'semantic_map_builder = visual_language_navigation.map.semantic_map_builder:main',
            'semantic_map_server = visual_language_navigation.map.semantic_map_server:main',
        ],
    },
)