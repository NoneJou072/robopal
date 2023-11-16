
# Installation  

## Prerequisite

* **Windows**  / **Linux**
* [MuJoCo-3.0.0](http://mujoco.org/) (latest version)
* Python 3.9+
* [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/index.html) 2.6.20 (Python binding)

## 二进制安装

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

## Build from source
  
   ```commandline
   $ git clone https://github.com/NoneJou072/robopal
   $ cd robopal
   $ pip install -r requirements.txt
   ```
