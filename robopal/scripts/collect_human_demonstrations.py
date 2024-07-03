# Refer to [robosuite](
# https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py#L231
# ) in this part.

import robopal
from robopal.wrappers import HumanDemonstrationWrapper
from robopal.devices import Keyboard, Gamepad


if __name__ == "__main__":

    env = robopal.make(
        "PickAndPlace-v1",
        robot="PandaPickAndPlace",
        render_mode='human',
        control_freq=20,
        controller='CARTIK',
        camera_in_window="frontview",
        is_render_camera_offscreen=True,
        is_randomize_end=False,
        is_randomize_object=True,
    )
    
    env = HumanDemonstrationWrapper(
        env, 
        device=Gamepad,
        collect_freq = 4,
        saved_action_type="position",
        is_render_actions=True,
    )
    env.device.start()
    
    env.reset()

    for t in range(int(1e6)):
        
        action = env.get_action()

        next_obs, reward, termination, truncation, info = env.step(action)

        # Also break if we complete the task
        if env.task_completion_hold_count == 0 or env.device._reset_flag:
            env.reset()

        # state machine to check for having a success for 10 consecutive timesteps
        if info["is_success"]:
            if env.task_completion_hold_count > 0:
                env.task_completion_hold_count -= 1  # latched state, decrement count
            else:
                env.task_completion_hold_count = 10  # reset count on first success timestep
        else:
            env.task_completion_hold_count = -1  # null the counter if there's no success

    env.close()
