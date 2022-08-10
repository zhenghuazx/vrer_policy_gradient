from setuptools import find_packages, setup

install_requires = open('requirements.txt').read().splitlines()

setup(
    name='PG-VRER-tf2',
    version='1.0.1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Hua Zheng',
    description='Implementations of variance reduction based policy optimizations algorithms in tensorflow 2.x',
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'rlalgorithms-tf2=rlalgorithms_tf2.cli:execute',
        ],
    },
)
