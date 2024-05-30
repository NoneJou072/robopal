import robopal
import robopal.envs
from robopal.wrappers import GymWrapper
import logging

if __name__ == "__main__":

    logging.info("avaliable envs:")
    for env_name in robopal.envs.REGISTERED_ENVS:
        logging.info(env_name)

    env = robopal.make(
        "Drawer-v1", 
        render_mode="human"
    )
    env = GymWrapper(env)

    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        env.step(action)
    env.close()
