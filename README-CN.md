<!-- Author: Haoran Zhou-->
<!-- date: 04.01.2023 -->
> [English](README.md) | 中文文档
---
<p>
  <a href="https://codeup.teambition.com/62219d81e4c44077bd46bffe/RoboIMI/tree/master" alt="GitHub">
    <img src="https://img.shields.io/github/actions/workflow/status/deepmind/mujoco/build.yml?branch=main">
  </a>
  <a href="https://mujoco.readthedocs.io/" alt="Documentation">
    <img src="https://readthedocs.org/projects/mujoco/badge/?version=latest">
  </a>
  <a href="https://codeup.teambition.com/62219d81e4c44077bd46bffe/RoboIMI/tree/master" alt="License">
    <img src="https://img.shields.io/github/license/deepmind/mujoco">
  </a>
</p>

**robopal** 是一个基于 [MuJoCo](http://mujoco.org/) 的开源机器人仿真框架，主要用于机械臂的深度强化学习训练与
控制算法验证。框架内提供了多种控制方案与示例环境，方便使用者进行进行开发。
截至目前的开发版本，主要包含下述功能：
* 底层关节位置/扭矩控制器  
  使用 Ruckig 规划关节位置、速度、加速度，实现平滑运动, 并基于 pd 控制与动力学计算扭矩
* 中间笛卡尔位置控制器  
  基于 pd 控制与运动学实现笛卡尔空间下的位置/旋转控制
* 上层控制器  
  对机器人进行上层控制

在程序编写过程中，我们对每个环境进行了抽象与层级封装，规范化代码写法并撰写代码文档，
方便后续的开发与维护。本框架与其他框架相比，具有如下优点：
* 具有更高的运动精度，更符合真实环境中的运动
* 高度可移植性，代码更简洁，方便学习与使用
* 丰富的任务环境， 如柔顺控制，动作模仿，视觉伺服等

---
## Installation  

### Environments

* Ubuntu 20.04/22.04
* [MuJoCo-2.3.6](http://mujoco.org/)
* Python 3.9+

### Quick Install  

   ```commandline
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   $ pip install -e .
   ```
### Build From Source

1. 项目部署：

   ```commandline
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   $ pip3 install -e .
   ```
2. PyKDL部署：
   https://github.com/orocos/orocos_kinematics_dynamics  
   如果是conda环境，可能会出现冲突问题，解决方法如下：
   https://blog.csdn.net/qq_43557907/article/details/127818837

---
