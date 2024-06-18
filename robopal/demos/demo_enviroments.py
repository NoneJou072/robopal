import robopal
import robopal.envs
from robopal.wrappers import GymWrapper, PettingStyleWrapper
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

    if len(env.agents) == 1:
        env = GymWrapper(env)
    else:
        env = PettingStyleWrapper(env)

    env.reset()
    for _ in range(400):
        if len(env.agents) == 1:
            action = env.action_space.sample()
        else:
            action = {
                agent: env.action_space(agent).sample() for agent in env.agents
            }
        env.step(action)
    env.close()
