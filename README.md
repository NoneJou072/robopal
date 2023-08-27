<!-- Author: Haoran Zhou-->
<!-- date: 04.01.2023 -->
> English | [中文文档](README-CN.md)
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

**robopal** is an open source robot simulation framework based on [MuJoCo](http://mujoco.org/), mainly used for deep reinforcement learning training and
Control algorithm verification. A variety of control schemes and example environments are provided within the framework.  

Compared with other frameworks, ours has the following advantages:
* Higher motion accuracy, it is more in line with the motion in the real environment
* High portability, more concise code, easy to learn and use
* Rich task environment, such as compliant control, motion imitation, visual servoing, etc.

## Installation  

### Required environments

* **Windows** (10+) / Linux
* Python 3.9+
* [MuJoCo-2.3.7](http://mujoco.org/)
* [pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 2.6.20 

### Install from source

   ```commandline
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   $ pip install -e .
   ```

## Controller
The development version mainly includes the following controllers:
* Joint Space Controller  
  Use Ruckig to plan joint positions, velocities, accelerations for smooth motion, and calculate torques based on pd control and dynamics
* Cartesian Space Controller  
  Position/rotation control in Cartesian space based on pd control and kinematics
