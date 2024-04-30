Controllers
==========

1. **关节空间控制器**

* ``IMPEDENCE``: 使用关节空间阻抗控制计算关节扭矩，可以选用 Ruckig 进行轨迹规划（可选）

2. **任务（笛卡尔）空间控制器**

* ``IK``: 使用逆运动学实现笛卡尔空间下的位姿控制，可以选用 pd 控制进行规划（可选）
* ``IMPEDENCE``: 使用任务空间阻抗控制计算关节扭矩


Joint Position Controller
-------------------------

.. autoclass:: robopal.controllers.jnt_imp_controller.JointImpedanceController

  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits



Joint Velocity Controller
-------------------------

.. autoclass:: robopal.controllers.jnt_vel_controller.JointVelocityController

  .. automethod:: set_goal
  .. automethod:: reset_goal


Operation Space Controller
--------------------------

.. autoclass:: robopal.controllers.task_imp_controller.CartesianImpedanceController

  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits