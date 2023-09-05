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
并具有：
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

### 从源码安装

   ```commandline
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   $ pip install -e .
   ```
## 控制器
* 关节空间控制器  
  使用阻抗控制计算关节扭矩，并可以使用 Ruckig 进行轨迹规划（可选） 
* 笛卡尔空间控制器  
  使用逆运动学实现笛卡尔空间下的位置和姿态控制，并可以使用 pd 控制进行调节（可选）

## 渲染
* mujoco 官方提供的渲染交互界面，基于 OpenGL 实现
* 相机视角渲染，使用 OpenCV 实现（可选）
* Unity 渲染（可选）
