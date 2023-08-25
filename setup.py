from setuptools import setup, find_packages

setup(
    name='robopal',
    packages=[package for package in find_packages() if package.startswith("robopal")],
    version='0.0.1',
    author="Haoran Zhou, Yichao Huang, Yuhan Zhao, Yang Lu",
    author_email="jou072@126.com, 15897778219@163.com, zyhzyh9426@163.com, yanglu202202@163.com",
    description="A simulation framework based mujoco",
    url="https://github.com/NoneJou072/robopal",
    python_requires=">=3",
    install_requires=[
        'mujoco==2.3.7',
        'ruckig~=0.9.2',
    ],
    extras_require={
        'interactive': ['gym~=0.21.0']
    },
)
