import os
from typing import Dict, List, Any
import logging
import time
from dataclasses import dataclass, field
import inspect
from copy import deepcopy

import numpy as np
import h5py

import robopal
from robopal.envs import RobotEnv
from robopal.plugins.devices.keyboard import KeyboardIO
import robopal.commons.transform as T

ROBOPAL_PATH = os.path.dirname(inspect.getfile(robopal))
COLLECTIONS_DIR_NAME = 'collections/collections_' + str(time.time()).replace(".", "_")
DEFAULT_DATA_DIR_PATH = os.path.join(ROBOPAL_PATH, COLLECTIONS_DIR_NAME)


@dataclass
class Collection:
    num_samples: int = 0
    model_file: str = None
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[np.ndarray] = field(default_factory=list)
    dones: List[np.ndarray] = field(default_factory=list)
    obs: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"observations": []})
    next_obs: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"observations": []})


class HumanDemonstrationWrapper(object):

    keyboard_recoder = KeyboardIO()

    def __init__(
            self, 
            env: RobotEnv, 
            collections_dir: str = DEFAULT_DATA_DIR_PATH,
            collect_freq = 1,
            max_collect_horizon = 100,
            is_drop_unsuccess_exp = True,
        ):
        
        self.env = env
        self.collections_dir = collections_dir
        self.collect_freq = collect_freq
        self.max_collect_horizon = max_collect_horizon
        self.is_drop_unsuccess_exp = is_drop_unsuccess_exp

        if not os.path.exists(self.collections_dir):
            logging.info("HumanDemonstrationWrapper: making new directory at {}".format(self.collections_dir))
            os.makedirs(self.collections_dir)

        hdf5_path = os.path.join(self.collections_dir, "demo.hdf5")
        self.f = h5py.File(hdf5_path, "w")
        # store some metadata in the attributes of one group
        self.root_group = self.f.create_group("data")

        env_args = {
            "env_name": self.env.get_configs("env_name"),
            "env_type": 2,  # GYM_TYPE
            "env_kwargs": 0
        }
        # store env config as an attribute
        for key, value in env_args.items():
            self.root_group.attrs[key] = value

        # choose one agent
        self.agent = "arm0"

        # memorize the init pose
        self.env.robot.end[self.agent].open()

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        self.collection: Collection = None
        self.num_collects  = 0

        # disable the viewer keyboard
        self.env.renderer.enable_viewer_keyboard = False
    
    def get_action(self):
        """ compute next actions based on the keyboard input
        """
        action = np.zeros(4)

        # normalize the action to [-1, 1]
        action[:3] = self.keyboard_recoder.get_end_pos_offset()
        action[:3] *= 20
        
        # action[3:7] = T.mat_2_quat(T.quat_2_mat(action[3:7]).dot(self.keyboard_recoder.get_end_rot_offset()))

        # normalized end action, since the end action from the keyboard is a binary value (0 or 1)
        action[3] = int(self.keyboard_recoder._gripper_flag)
        action[3] = action[3] * 2 - 1

        return action.copy()
    
    def step(self, action: np.ndarray | Dict[str, np.ndarray]):
        """ Input actions should be normalized, since the action 
        will un-normalize before applying to the environment.
        """
        # collect current observation
        obs = self.env._get_obs()

        next_obs, reward, termination, truncation, info = self.env.step(action)

        if not self.has_interaction:
            self.has_interaction = True

        if self.env.cur_timestep % self.collect_freq == 0:
            self.collection.num_samples += 1
            self.collection.actions.append(action)
            self.collection.obs["observations"].append(obs)
            self.collection.next_obs["observations"].append(next_obs)
            self.collection.dones.append(termination)
            self.collection.rewards.append(reward)
            self.env.save_state()
            self.collection.states.append(self.env.get_state())

        if self.keyboard_recoder._exit_flag:
            self.close()

        return next_obs, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        # save the data if any interactions have happened in this episode
        if self.has_interaction:
            self.save_collection()

        self.keyboard_recoder._reset_flag = False
        self.has_interaction = False

        ret = self.env.reset(seed, options)

        # delete old and make new collection
        del self.collection
        self.collection = Collection(
            model_file=self.env.get_configs("model_file")
        )

        return ret
    
    def save_collection(self):
        # drop unsuccessful episode
        if self.is_drop_unsuccess_exp:
            if not self.collection.infos[-1]["is_success"]:
                logging.info("Demonstration is unsuccessful and has NOT been saved")
                return
        
        # check the collection
        if len(self.collection.observations) == 0:
            return
        assert len(self.collection.observations) == len(self.collection.actions)

        self.num_collects += 1
        ep_data_grp = self.root_group.create_group("demo_{}".format(self.num_collects))

        # add attrs
        ep_data_grp.attrs["num_samples"] = self.collection.num_samples
        ep_data_grp.attrs["model_file"] = self.collection.model_file

        # write datasets
        ep_data_grp.create_dataset("states", data=np.array(self.collection.states, dtype=np.float64))
        ep_data_grp.create_dataset("observations", data=np.array(self.collection.observations, dtype=np.float32))
        ep_data_grp.create_dataset("actions", data=np.array(self.collection.actions, dtype=np.float32))
        ep_data_grp.create_dataset("rewards", data=np.array(self.collection.actions, dtype=np.float32))
        ep_data_grp.create_dataset("dones", data=np.array(self.collection.actions, dtype=np.float32))

        # write obs
        obs_group = ep_data_grp.create_group("obs")
        for key, value in self.collection.obs.items():
            obs_group.create_dataset(key, value)
        next_obs_group = ep_data_grp.create_group("next_obs")
        for key, value in self.collection.next_obs.items():
            next_obs_group.create_dataset(key, value)

        logging.info("Demonstration is successful and has been saved")

    def close(self):
        if self.has_interaction:
            self.save_collection()

        self.f.close()
        logging.info("HumanDemonstrationWrapper: closed hdf5 file")

        self.env.close()
