# Robot

## 1. 概念介绍
Mujoco 使用 mjcf 模型语言来描述机器人模型文件。
在 robopal 中，mjcf 模型由四个部分组成, 分别定义为
* `manipulators`
* `grippers`
* `mounts`
* `scenes`

这些模块对应的模型分别存放在 `robopal.assets.models.*` 中。利用 `commons.xml_splice.py` 自动化脚本可以将这四个部分拼接成统一的 MJCF 模型。

## 2. 模型组合

下面是一个简单的示例，演示了如何在 robopal 中创建一个模型。

### 2.1 初始化模型 
示例中的 `DianaMed` 位于 `robopal.robots.diana_med.py`。

```python
class DianaMed(BaseArm):
    """ DianaMed robot base class. """
    def __init__(self,
                 scene='default',
                 manipulator='DianaMed',
                 gripper=None,
                 mount=None
                 ):
        super().__init__(
            name="diana_med",
            scene=scene,
            mount=mount,
            manipulator=manipulator,
            gripper=gripper,
            attached_body='0_link7',
        )
        self.arm_joint_names = [['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6', '0_j7']]
        self.arm_actuator_names = [['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6', '0_a7']]

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])
```

### 2.2 写入配置文件

### 2.3 使用自定义模型
除了内置的几种模型外，robopal 还支持使用自定义的模型文件, 下面以命名为 `CustomArm.xml` 的模型文件为例介绍如何导入自定义模型。

1. step1：将模型打包好，放入 `robopal.assets.models.manipulators.CustomArm`, 注意 `.xml` 文件的名称与存放该文件的文件夹名称一致，以将机械臂模型设置成可被 robopal 自动识别的内置模型。在放入 manipulators 文件夹后，最好先测试下 xml 文件是否可以导入 mujoco 中。
2. step2：仿照 `robopal.robots` 文件夹中提供内置模型对应的配置文件，创建一个新的配置文件，如 `robopal.robots.custom_arm.py`：

```python
import os

import numpy as np
from robopal.robots.base import BaseRobot

ASSET_DIR = os.path.join(os.path.dirname(__file__), '../assets')

class CustomArm(BaseRobot):
    """ CustomArm robot base class. """
    def __init__(self,
                 scene='default',
                 manipulator='CustomArm',  # 与 xml 文件名一致
                 gripper=None,
                 mount=None
                 ):
        super().__init__(
            scene=scene,
            mount=mount,
            manipulator=manipulator,
            gripper=gripper,
            attached_body='0_attachment',
        )
        self.arm_joint_names = {self.agents[0]: ['0_joint1', '0_joint2', '0_joint3', '0_joint4', '0_joint5', '0_joint6', '0_joint7']}
        self.arm_actuator_names = {self.agents[0]: ['0_actuator1', '0_actuator2', '0_actuator3', '0_actuator4', '0_actuator5', '0_actuator6', '0_actuator7']}
        self.base_link_name = {self.agents[0]: '0_link0'}
        self.end_name = {self.agents[0]: '0_attachment'}

        self.pos_max_bound = np.array([0.6, 0.2, 0.37])
        self.pos_min_bound = np.array([0.3, -0.2, 0.02])

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([-0.61,  -0.84,  0.47, -2.54,  0.35,  1.75, 0.44])}
```

3. step3：在 `robopal.robots.__init__.py` 中导入自定义模型

```python
from .custom_arm import *
```

上述操作完成后，你的机械臂已经可以被 robopal 识别到了，接下来可以在 `robopal.tests.test_controller.py` 中引用你的机械臂进行测试。

```python
from robopal.robots import CustomArm
from robopal.envs import RobotEnv

env = RobotEnv(
    robot=CustomArm,
    render_mode='human',
    control_freq=200,
    is_interpolate=False,
    controller=options['ctrl'],
)
```

4. step4：假如测试时有发现机械臂动作执行不正确的情况，需要调整机械臂控制器的参数。例如，如果您使用了 JNTIMP 控制器，可以通过下面的方式调整机械臂的参数：

```python
env.controller.set_jnt_params(
    b=20.0 * np.ones(self.dofs),
    k=80.0 * np.ones(self.dofs),
)
```

### 2.4 写入末端配置

### 2.5 向场景中添加物体

### 2.6 多智能体（双臂）构建
