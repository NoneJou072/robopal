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

更多了解，请查看文档 [[Documentations(正在维护中)]](https://robopal.readthedocs.io/)

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