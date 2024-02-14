from .jnt_imp_controller import JntImpedance
from .jnt_vel_controller import JntVelController
from .task_imp_controller import CartImpedance

# controller mapping
controllers = {
    'JNTIMP': JntImpedance,
    'JNTVEL': JntVelController,
    'CARTIMP': CartImpedance,
}

__all__ = [
    'controllers',
]
