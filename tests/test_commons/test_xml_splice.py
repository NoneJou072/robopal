from robopal.robots.base import *
from mujoco import viewer



if __name__ == "__main__":
    robot = BaseRobot(
        scene='default',
        manipulator='DianaMed',
        gripper='PandaHand',
        mount=None,
        attached_body='0_attachment',
    )

    with viewer.launch_passive(robot.robot_model, robot.robot_data) as viewer:

        viewer.sync()

        while viewer.is_running():

            # Step the physics.
            mujoco.mj_step(robot.robot_model, robot.robot_data)

            viewer.sync()
    