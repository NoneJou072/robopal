
<div style="text-align: center;">

# **ROBOPAL**

![License](https://img.shields.io/badge/license-Apache2.0-yellow?style=flat-square) 
![GitHub Repo stars](https://img.shields.io/github/stars/NoneJou072/robopal?style=flat-square&logo=github)
![Language](https://img.shields.io/badge/language-python-brightgreen?style=flat-square)
[![Documentation](https://img.shields.io/badge/documentation-yes-brightgreen?style=flat-square)](https://robopal.readthedocs.io/zh/latest/index.html)
![PyPI - Version](https://img.shields.io/pypi/v/robopal?style=flat-square)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11078757.svg)](https://doi.org/10.5281/zenodo.11078757)
</div>

**robopal** 是一个基于 [MuJoCo](http://mujoco.org/) 动力学引擎搭建的多平台开源机器人仿真框架，主要用于机械臂的深度强化学习训练与控制算法验证。框架内提供了多种控制方案与底层环境，
具有以下优点：
* 采用 Mujoco 原生 API 计算机械臂运动学与动力学
* 简洁的代码结构，没有复杂的嵌套关系，方便快速上手学习和使用
* 环境遵循最新版 OpenAI Gymnasium 接口规范，方便集成主流强化学习算法库(eg. SB3)
* 提供多种基础控制方案，如关节空间/笛卡尔空间的位置控制、速度控制、阻抗控制
* 提供丰富的任务环境，如桌面操作，视觉伺服等

请[查看文档](https://robopal.readthedocs.io/)以获取更多信息 (更新中)

---
## Getting Started  

### 环境要求

* **Windows** / **Linux**
* [MuJoCo-3.1.4+](http://mujoco.org/)
* Python 3.8 +

### 二进制安装
> 当前 PyPi 上的 robopal 版本过低，暂不推荐使用. **建议从源安装最新版本**

```commandline
$ pip install robopal
```

### Build from source
  
   ```python
   # Clone robopal
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   # Install robopal and its requirements.
   $ pip install -r requirements.txt
   ```

### Run a demo

```bash
python -m robopal.demos.demo_controller
```

## Contribute
robopal 目前存在很多不足之处，欢迎大家在 issue 中提出问题或留下宝贵的建议，欢迎对这个项目有兴趣的一起来完善。

## Citation

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
