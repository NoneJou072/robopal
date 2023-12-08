# Introduction

**robopal** 是一个基于 [MuJoCo](http://mujoco.org/) 动力学引擎与 [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 机器人动力学库搭建的多平台开源机器人仿真框架，主要用于机械臂的深度强化学习训练与控制算法验证。框架内提供了多种控制方案与底层环境，
具有以下优点：
* 采用 Pinocchio 动力学库计算机械臂运动学与动力学，方便验证控制算法或将算法向实物迁移
* 简洁的代码结构，没有复杂的嵌套关系，方便快速上手学习和使用
* 环境遵循最新版 OpenAI Gymnasium 接口规范，方便与其他算法库(e.g. SB3)集成
* 提供多种基础控制方案，如关节空间/笛卡尔空间的位置控制、速度控制、阻抗控制等
* 提供丰富的任务环境，如桌面操作，视觉伺服等

## Citation

```bibtex
@misc{robopal2023,
    author = {Haoran Zhou},
    title = {robopal: A Modular Robotic Simulation Framework for Reinforcement Learning},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/NoneJou072/robopal}},
}
```
