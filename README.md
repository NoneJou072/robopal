
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

**robopal** 是一个基于 [MuJoCo](http://mujoco.org/) 动力学引擎与 [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 机器人动力学库搭建的多平台开源机器人仿真框架，主要用于机械臂的深度强化学习训练与控制算法验证。框架内提供了多种控制方案与底层环境，
具有以下优点：
* 采用 Pinocchio 动力学库计算机械臂运动学与动力学，方便验证控制算法或将算法向实物迁移
* 简洁的代码结构，没有复杂的嵌套关系，方便快速上手学习和使用
* 环境遵循最新版 OpenAI Gymnasium 接口规范，方便与其他算法库(e.g. SB3)集成
* 提供多种基础控制方案，如关节空间/笛卡尔空间的位置控制、速度控制、阻抗控制等
* 提供丰富的任务环境，如桌面操作，视觉伺服等

更多了解，请查看文档 [[Documentations(正在撰写中)]](https://robopal.readthedocs.io/)

---
## 安装  

### 环境要求

* **Windows** 10/11 / **Linux** (recommended)
* [MuJoCo-3.0.0](http://mujoco.org/)
* Python 3.9+
* [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 2.6.20 

### 二进制安装

```commandline
$ pip install robopal
```

此外，需要自行安装 Pinocchio 库，如果您是 Linux 系统，可以直接安装：
```commandline
$ pip install pin
```

如果您是 Windows 系统，可以从 conda 安装：
```commandline
$ conda install pinocchio
```

### Build from source
  
   ```commandline
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   $ pip install -r requirements.txt
   ```

## Contribute
这个项目是我在学习过程中所搭建的仿真框架，目前还有很多不足之处，欢迎大家提出建议和意见，也欢迎对这个项目有兴趣的一起来完善。
目前对该框架的近期规划是，加入双臂控制，并充分利用 Mujoco 的各种特性，加入更多新颖的任务环境。

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
