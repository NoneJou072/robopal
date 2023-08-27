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

**robopal** 是一个基于 [MuJoCo](http://mujoco.org/) 动力学引擎与 [pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 机器人动力学库搭建的多平台开源机器人仿真框架，主要用于机械臂的深度强化学习训练与控制算法验证。框架内提供了多种控制方案与示例环境，

本框架与其他框架相比，具有如下优点：
* 更高的控制精度，更贴合真实的机械臂运动情况
* 高度可移植性，代码更简洁，方便学习与使用
* 丰富的任务环境， 如柔顺控制，动作模仿，视觉伺服等

---
## 安装  

### 环境

* **Windows** (10+) / Linux
* [MuJoCo-2.3.7](http://mujoco.org/)
* Python 3.9+
* [pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 2.6.20 

### 快速安装

   ```commandline
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   $ pip install -e .
   ```
## 控制器
目前包含了下述内容：
* 关节空间控制器  
  使用 Ruckig 规划关节位置、速度、加速度，实现平滑运动, 使用阻抗控制计算扭矩
* 笛卡尔空间控制器  
  基于 pd 控制与运动学实现笛卡尔空间下的位置和姿态控制
