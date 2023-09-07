from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='robopal',
    packages=[package for package in find_packages() if package.startswith("robopal")],
    version='0.1.0',
    author="Haoran Zhou, Yichao Huang, Zongdao Li, Yuhan Zhao, Yang Lu",
    author_email="jou072@126.com",
    description="robopal: A Simulation Framework Based Mujoco",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NoneJou072/robopal",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        'mujoco==2.3.7',
        'ruckig~=0.9.2',
        'pin'
    ],
    extras_require={
        'interactive': ['gymnasium']
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)
