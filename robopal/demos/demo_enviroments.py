import robopal
import robopal.envs
from robopal.wrappers import GymWrapper
import logging

if __name__ == "__main__":

    logging.info("avaliable envs:")
    for env_id, env_name in enumerate(robopal.envs.REGISTERED_ENVS.keys()):
        logging.info(f"{env_id}. {env_name}")
    env_id = input("Choose the enviroment:")
    
    env_name = list(robopal.envs.REGISTERED_ENVS.keys())[int(env_id)]
    env = robopal.make(
        env_name, 
        render_mode="human"
    )
    env = GymWrapper(env)

    env.reset()
    for _ in range(400):
        action = env.action_space.sample()
        env.step(action)
    env.close()
