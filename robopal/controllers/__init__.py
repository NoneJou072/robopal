from .base_controller import BaseController
from .jnt_imp_controller import JointImpedanceController
from .jnt_vel_controller import JointVelocityController
from .task_imp_controller import CartesianImpedanceController
from .task_ik_controller import CartesianIKController
from .jnt_torque_controller import JointTorqueController

# controller mapping
controllers = {
    'JNTTAU': JointTorqueController,
    'JNTIMP': JointImpedanceController,
    'JNTVEL': JointVelocityController,
    'CARTIMP': CartesianImpedanceController,
    'CARTIK': CartesianIKController
}

__all__ = [
    'controllers',
]
