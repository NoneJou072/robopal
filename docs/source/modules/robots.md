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
我们也可以使用自定义的模型文件, 例如下面的示例使用了自定义模型 `CustomArm.xml`。
```python
class CustomArm(BaseArm):
    def __init__(self):
        super().__init__(scene='grasping',
                         manipulator='/path/to/CustomArm/CustomArm.xml',
                         gripper='RethinkGripper',
                         mount='top_point')
```

* Q：如何将自己的机械臂模型设置成可被 robopal 自动识别的内置模型？
* A：将模型打包好，放入 `robopal.assets.models.manipulators`, 注意 `.xml` 文件的名称与存放该文件的文件夹名称一致

### 2.4 写入末端配置

### 2.5 向场景中添加物体

### 2.6 多智能体（双臂）构建
