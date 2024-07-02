import os
from typing import Dict, List, Any, Union
import logging
import time
from dataclasses import dataclass, field
import inspect
import json

import numpy as np
import h5py

import robopal
from robopal.envs.manipulation_tasks.robot_manipulate import ManipulateEnv
import robopal.commons.transform as T
from robopal.devices import BaseDevice

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
        default_factory=lambda: {"low_dim": []})
    next_obs: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"low_dim": []})


class HumanDemonstrationWrapper(object):

    def __init__(
            self, 
            env: ManipulateEnv,
            device: BaseDevice = None,
            collections_dir: str = DEFAULT_DATA_DIR_PATH,
            collect_freq = 1,
            max_collect_horizon = 100,
            is_drop_unsuccess_exp = True,
            is_drop_invalid_action = True,
            saved_action_type = "velocity"
        ):

        self.env = env
        self.device: BaseDevice = device()
        self.collections_dir = collections_dir
        self.collect_freq = collect_freq
        self.max_collect_horizon = max_collect_horizon
        self.is_drop_unsuccess_exp = is_drop_unsuccess_exp
        self.is_drop_invalid_action = is_drop_invalid_action
        self.saved_action_type = saved_action_type
        if self.saved_action_type == "position":
            logging.warn("HumanDemonstrationWrapper: saved action type is set to position, note the action is unnormalized.")

        self._last_action = np.zeros(4)

        if not os.path.exists(self.collections_dir):
            logging.info("HumanDemonstrationWrapper: making new directory at {}".format(self.collections_dir))
            os.makedirs(self.collections_dir)

        hdf5_path = os.path.join(self.collections_dir, "demo.hdf5")
        self.f = h5py.File(hdf5_path, "w")
        # store some metadata in the attributes of one group
        self.root_group = self.f.create_group("data")

        env_args = {
            "env_name": self.env.get_configs("env_name"),
            "type": 4,  # ROBOPAL_TYPE in robomimic4pal
            # pass to the env constructor
            "env_kwargs": {
                "robot": self.env.get_configs("robot"),
                "control_freq" : self.env.control_freq,
                "controller" : self.env.controller.name,
                "is_render_camera_offscreen": self.env.is_render_camera_offscreen,
                "is_randomize_end" : self.env.is_randomize_end,
                "is_randomize_object" : self.env.is_randomize_object,
                "action_type" : self.saved_action_type,
            }
        }
        # store env config as an attribute
        self.root_group.attrs["env_args"] = json.dumps(env_args)

        # choose one agent
        self.agent = "arm0"

        # memorize the init pose
        self.env.robot.end[self.agent].open()

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        self.collection: Collection = None
        self.num_collects  = 0
        self.total = 0
        self.task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal

        # disable the viewer keyboard
        self.env.renderer.enable_viewer_keyboard = False
    
    def get_action(self):
        """ compute next actions based on the keyboard input
        """
        action = np.zeros(4)

        # normalize the action to [-1, 1]
        action[:3] = self.device.get_outputs()[0] * 40
        action[:3] = action[:3].clip(-1, 1)
        
        # action[3:7] = T.mat_2_quat(T.quat_2_mat(action[3:7]).dot(self.device.get_end_rot_offset()))

        # normalized end action, since the end action from the keyboard is a binary value (0 or 1)
        action[3] = int(self.device._gripper_flag)
        action[3] = action[3] * 2 - 1

        return action.copy()
    
    def step(self, action: Union[np.ndarray, Dict[str, np.ndarray]]):
        """ Input actions should be normalized, since the action 
        will un-normalize before applying to the environment.
        """
        # collect current observation
        obs = self.env._get_obs()
        self.env.save_state()
        state = self.env.get_state()

        next_obs, reward, termination, truncation, info = self.env.step(action)

        if not self.has_interaction and action[:3].any() != 0:
            self.has_interaction = True
            logging.info("HumanDemonstrationWrapper: interaction has started")

        if self.saved_action_type == "velocity":
            pass
        elif self.saved_action_type == "position":
            action[:3] = self.env.desired_position
        else:
            raise ValueError(f"Invalid action type: {self.saved_action_type}")

        if self.has_interaction and self.env.cur_timestep % self.collect_freq == 0:
            collect_flag = True
            if self.is_drop_invalid_action and self.task_completion_hold_count < 0:
                collect_flag = self._check_action(action, obs, next_obs)
            if collect_flag:
                # collect the data
                self.collection.num_samples += 1
                self.collection.actions.append(action)
                self.collection.obs["low_dim"].append(obs)
                self.collection.next_obs["low_dim"].append(next_obs)
                self.collection.dones.append(termination)
                self.collection.rewards.append(reward)
                self.collection.states.append(state)

        if self.device._exit_flag:
            self.close()

        return next_obs, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        # save the data if any interactions have happened in this episode
        if self.has_interaction:
            self.save_collection()

        self.device._reset_flag = False
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
            if not self.env._get_info()["is_success"]:
                logging.info("Demonstration is unsuccessful and has NOT been saved")
                return
        
        # check the collection
        if len(self.collection.actions) == 0:
            return

        ep_data_grp = self.root_group.create_group("demo_{}".format(self.num_collects))
        self.num_collects += 1

        # add attrs
        self.total += self.collection.num_samples
        ep_data_grp.attrs["num_samples"] = self.collection.num_samples
        ep_data_grp.attrs["model_file"] = self.collection.model_file

        # write datasets
        ep_data_grp.create_dataset("states", data=np.array(self.collection.states, dtype=np.float32))
        ep_data_grp.create_dataset("actions", data=np.array(self.collection.actions, dtype=np.float32))
        ep_data_grp.create_dataset("rewards", data=np.array(self.collection.actions, dtype=np.float32))
        ep_data_grp.create_dataset("dones", data=np.array(self.collection.actions, dtype=np.float32))

        # write obs
        obs_group = ep_data_grp.create_group("obs")
        for key, value in self.collection.obs.items():
            obs_group.create_dataset(key, data=np.array(value, dtype=np.float32))
        next_obs_group = ep_data_grp.create_group("next_obs")
        for key, value in self.collection.next_obs.items():
            next_obs_group.create_dataset(key, data=np.array(value, dtype=np.float32))

        logging.info("Demonstration is successful and has been saved in demo_{}".format(self.num_collects))

    def close(self):
        if self.has_interaction:
            self.save_collection()

        # store the total number of samples
        self.root_group.attrs["total"] = self.total
        self.f.close()
        logging.info("HumanDemonstrationWrapper: closed hdf5 file")

        self.env.close()

    def _check_action(self, action: np.ndarray, obs: np.ndarray, next_obs: np.ndarray):
        """ check if the action is valid
        """
        valid = True
        if np.linalg.norm(self._last_action - action) < 1e-3:
            if np.linalg.norm(obs - next_obs) < 1e-2:
                valid = False
        self._last_action = action
        return valid
    