import os
import inspect
from glob import glob
import logging
import json

import h5py
import numpy as np
import robopal

ROBOPAL_PATH = os.path.dirname(inspect.getfile(robopal))
COLLECTIONS_DIR_NAME = 'collections/collections_*'
DEFAULT_DATA_DIR_PATH = os.path.join(ROBOPAL_PATH, COLLECTIONS_DIR_NAME)


def play_demonstrations():
    for state_file in sorted(glob(DEFAULT_DATA_DIR_PATH)):

        # read .hdf5 files
        file = h5py.File(state_file + '/demo.hdf5','r')   
        logging.info("Reading from {}".format(state_file + '/demo.hdf5'))

        env_args = json.loads(file["data"].attrs["env_args"])
        env_name = env_args["env_name"]
        env_meta = env_args["env_kwargs"]
        logging.info(f"env name: {env_name}")
        logging.info(f"env meta: {env_meta}")

        env = robopal.make(
            env_name,
            **env_meta
        )

        demos = list(file["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        # playback each episode
        for episode in demos:
            logging.info("\n>Reading episode: {}".format(episode))

            for key in file["data"][episode].attrs:
                if key != "model_file":
                    logging.info("{}: {}".format(key, file["data"][episode].attrs[key]))

            env.load_model_from_string(file["data"][episode].attrs["model_file"])

            first_state = file["data"][episode]["states"][0]

            env.load_state(first_state)
            env.forward()
            env.update_init_pose_to_current()

            for action in file["data"][episode]["actions"]:
                env.step(action)
                    
        env.renderer.close()

if __name__ == "__main__":
    play_demonstrations()
