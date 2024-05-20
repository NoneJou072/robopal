Controllers
==========

Joint Torque Controller
-------------------------

``JNTTAU``: 直接输入关节扭矩

.. autoclass:: robopal.controllers.jnt_torque_controller.JointTorqueController

  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits

Joint Position Controller
-------------------------

``JNTIMP``: 使用关节空间阻抗控制计算关节扭矩，可以选用 Ruckig 进行轨迹规划（可选）

.. autoclass:: robopal.controllers.jnt_imp_controller.JointImpedanceController

  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits

Joint Velocity Controller
-------------------------
``JNTVEL``: 使用关节空间阻抗控制计算关节扭矩，可以选用 Ruckig 进行轨迹规划（可选）

.. autoclass:: robopal.controllers.jnt_vel_controller.JointVelocityController

  .. automethod:: set_goal
  .. automethod:: reset_goal

Cartesian Space Controller
--------------------------
``CARTIMP``: 使用任务空间阻抗控制计算关节扭矩

.. autoclass:: robopal.controllers.task_imp_controller.CartesianImpedanceController

  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits

Cartesian Space Controller
--------------------------
``CARTIK``: 使用逆运动学实现笛卡尔空间下的位姿控制，可以选用 pd 控制进行规划（可选）

.. autoclass:: robopal.controllers.task_ik_controller.CartesianIKController

  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits
