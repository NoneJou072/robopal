# Refer to [robosuite](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py#L231) in this part.

from robopal.demos.manipulation_tasks.demo_pick_place import PickAndPlaceEnv
from robopal.wrappers import HumanDemonstrationWrapper


if __name__ == "__main__":

    from robopal.robots.panda import PandaPickAndPlace

    env = PickAndPlaceEnv(
        robot=PandaPickAndPlace,
        render_mode='human',
        control_freq=20,
        controller='CARTIK',
    )
    
    env = HumanDemonstrationWrapper(env)
    env.keyboard_recoder.start()

    env.reset()
    
    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal

    for t in range(int(1e6)):
        
        action = env.get_action()

        next_obs, reward, termination, truncation, info = env.step(action)

        # Also break if we complete the task
        if task_completion_hold_count == 0 or env.keyboard_recoder._reset_flag:
            env.reset()

        # state machine to check for having a success for 10 consecutive timesteps
        if info["is_success"]:
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    env.close()
