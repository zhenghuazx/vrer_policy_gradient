from setuptools import find_packages, setup

install_requires = open('requirements.txt').read().splitlines()

setup(
    name='vrer-pg',
    version='1.1.1',
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
            'vrer-pg=vrer_policy_gradient.cli:execute',
        ],
    },
)
