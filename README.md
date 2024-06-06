
<div style="text-align: center;">

<img title="Logo" src="https://github.com/NoneJou072/robopal/blob/main/docs/source/_static/logo.png?raw=true" width = 80%/>

![License](https://img.shields.io/badge/license-Apache2.0-yellow?style=flat-square) 
![GitHub Repo stars](https://img.shields.io/github/stars/NoneJou072/robopal?style=flat-square&logo=github)
![Language](https://img.shields.io/badge/language-python-brightgreen?style=flat-square)
[![Documentation](https://img.shields.io/badge/documentation-yes-brightgreen?style=flat-square)](https://robopal.readthedocs.io/zh/latest/index.html)
![PyPI - Version](https://img.shields.io/pypi/v/robopal?style=flat-square)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11078757.svg)](https://doi.org/10.5281/zenodo.11078757)

</div>

**robopal** is a multi-platform, modular robot simulation framework based on [MuJoCo](http://mujoco.org/) physics engine, which is mainly used for reinforcement learning training and control algorithm implementation of robotic arms. Please check the [Documentation](https://robopal.readthedocs.io/) for more information.

**robopal** 是一个基于 [MuJoCo](http://mujoco.org/) 物理引擎搭建的多平台的，模块化的机器人仿真框架，主要用于机械臂的强化学习训练与控制算法实施。

robopal 为您提供了：
* 采用 [Mujoco](http://mujoco.org/) 原生 API 计算机械臂动力学与[运动学]()，无需额外安装扩展库，提高运行帧数
* 简洁的代码结构，没有复杂的嵌套关系，方便快速上手学习和使用
* 具备 [Gymnasium]() 风格的单臂环境与 [PettingZoo]() 风格的双臂环境，方便集成大部分的单/多智能体强化学习算法库(eg. [stable-baselines3]()，[MARL]())
* 提供多种基础控制方案，如关节空间/笛卡尔空间的位置控制、速度控制、阻抗控制，并提供了遥操作接口
* 提供丰富的任务环境示例，如 ConveyorBelt，PickAndPlace, Drawer, Cabinet，VisualServo等
* 模块化定制 MJCF 描述的机器人场景模型，可自由组合搭配场景，基座，机械臂，末端执行器和物体

请查看[文档](https://robopal.readthedocs.io/)以获取更多信息 (更新中)

---
## Getting Started  

### Preparation

* **Windows** / **Linux**
* [MuJoCo-3.1.5](http://mujoco.org/)
* Python 3.8 +

### Install from pip
> You are advised to **Install from source** to obtain the latest version

```commandline
$ pip install robopal
```

### Install from source
  
   ```python
   # Clone robopal
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   # Install robopal and its requirements.
   $ pip install -r requirements.txt
   ```

### Run a demo

```bash
python -m robopal.demos.demo_controllers
```

## Contribute
robopal currently has many shortcomings. Welcome to raise questions or leave suggestions in [Issue](), and also welcome to [Pull Request]() to improve this project together.

## Citation
Please cite robopal if you find useful in this work:
```bibtex
@software{Zhou_robopal_A_Simulation_2024,
author = {Zhou, Haoran and Huang, Yichao and Zhao, Yuhan and Lu, Yang},
doi = {10.5281/zenodo.11078757},
month = apr,
title = {{robopal: A Simulation Framework based Mujoco}},
url = {https://github.com/NoneJou072/robopal},
version = {0.3.1},
year = {2024}
}
```
