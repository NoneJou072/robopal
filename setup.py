from setuptools import setup, find_packages

with open("README.md", "r", encoding='UTF-8') as fh:
    long_description = fh.read()

setup(
    name='robopal',
    packages=[package for package in find_packages() if package.startswith("robopal")],
    version='0.2.1',
    author="Haoran Zhou, Yichao Huang, Zongdao Li, Yuhan Zhao, Yang Lu",
    author_email="jou072@126.com",
    description="robopal: A Simulation Framework Based Mujoco",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NoneJou072/robopal",
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        'mujoco>=2.3.7',
        'pin'
    ],
    extras_require={
        'gym': ['gymnasium'],
        'traj': ['ruckig~=0.9.2'],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)
