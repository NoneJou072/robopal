# Quick Start

## Gymnasium Style Environment

我们的环境遵循 OpenAI Gymnasium 接口规范，可以方便地搭建强化学习训练环境。

下面是一个简单的示例，演示了使用 [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) 训练 *Pick-and-Place* 任务。
其中，使用的环境来自于 `robopal/demos/single_task_manipulation` 目录下的 `PickAndPlaceEnv` 环境。

```python
