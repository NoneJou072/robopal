
# Robot

robopal 将模型分为四个部分, 分别是 `manipulators`,`grippers`,`mounts`,`scenes`. 通过程序进行模型的拼接.

下面是一个简单的示例，
该文件位于 `robopal/robots/single_task_manipulation` 目录下的 `PickAndPlaceEnv` 环境。

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
            chassis=mount,
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

我们也可以使用自定义的模型文件, 例如下面的代码使用了自定义的模型文件 `DianaMed.xml`。
```python
class DianaDrawer(DianaMed):
    def __init__(self):
        super().__init__(scene='grasping',
                         manipulator='/home/mhming/zhr/robopal/robopal/assets/models/manipulators/DianaMed/DianaMed.xml',
                         gripper='rethink_gripper',
                         mount='top_point')

    def add_assets(self):
        # add cupboard
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cupboard/cupboard.xml')

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([-0.51198529, -0.44737435, -0.50879166, 2.3063219, 0.46514545, -0.48916244, -0.37233289])
```
