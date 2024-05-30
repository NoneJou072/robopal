# Introduction

**robopal** 是一个基于 [MuJoCo](http://mujoco.org/) 动力学引擎搭建的多平台开源机器人仿真框架，主要用于机械臂的深度强化学习训练与控制算法验证。框架内提供了多种控制方案与底层环境，
具有以下优点：
* 采用 Mujoco 原生 API 计算机械臂运动学与动力学
* 简洁的代码结构，没有复杂的嵌套关系，方便快速上手学习和使用
* 环境遵循最新版 OpenAI Gymnasium 接口规范，方便与其他算法库(e.g. SB3)集成
* 提供多种基础控制方案，如关节空间/笛卡尔空间的位置控制、速度控制、阻抗控制等
* 提供丰富的任务环境，如桌面操作，视觉伺服等

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
