
# Installation  

### 环境要求

* **Windows** / **Linux** (recommended)
* [MuJoCo-3.0.1](http://mujoco.org/)
* Python 3.9 +
* [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 2.6.20 +

### 二进制安装
> 当前PyPi上的版本是 ![PyPI - Version](https://img.shields.io/pypi/v/robopal?style=flat-square)，Repo上的版本是 `0.3.0`，建议安装最新版本

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
