<!-- Author: Haoran Zhou-->
<!-- date: 04.01.2023 -->
> [English](README.md) | 中文文档

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

## 简介
**robopal** 是一个基于 [MuJoCo](http://mujoco.org/) 动力学引擎与 [pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 机器人动力学库搭建的多平台开源机器人仿真框架，主要用于机械臂的深度强化学习训练与控制算法验证。框架内提供了多种控制方案与底层环境，
并具有：
* 更高的控制精度，方便 sim2real
* 高度可移植性，代码简洁，方便学习与使用
* 丰富的任务环境，如物体抓取，视觉伺服等

---
## 安装  

### 环境要求

* **Windows** 10/11 / Linux
* [MuJoCo-2.3.7](http://mujoco.org/)
* Python 3.9+
* [pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 2.6.20 

### 二进制安装
  如果您是 Linux 系统，可以直接使用以下命令安装：
   ```commandline
   $ pip install robopal
   ```

### 从源码安装
  
   ```commandline
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   $ pip install -r requirements.txt
   ```

## 模块
### 控制器
1. **关节空间控制器**  
* `IMPEDENCE`: 使用关节空间阻抗控制计算关节扭矩，可以选用 Ruckig 进行轨迹规划（可选） 

2. **任务（笛卡尔）空间控制器**  
* `IK`: 使用逆运动学实现笛卡尔空间下的位姿控制，可以选用 pd 控制进行规划（可选）
* `IMPEDENCE`: 使用任务空间阻抗控制计算关节扭矩
  
### 渲染方式
* mujoco 官方提供的渲染交互界面，基于 OpenGL 实现
* 相机视角渲染，使用 OpenCV 实现（可选）
* ~~Unity 渲染~~（调试中）

## 未来计划
- [ ] 多种单协作臂强化学习任务场景
- [ ] 双协作臂多智能体强化学习环境 
- [ ] 大语言模型接口
- [ ] 丰富资源库

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