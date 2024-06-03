import os
import inspect
from glob import glob
import logging
import ast

import h5py
import robopal

ROBOPAL_PATH = os.path.dirname(inspect.getfile(robopal))
COLLECTIONS_DIR_NAME = 'collections/collections_*'
DEFAULT_DATA_DIR_PATH = os.path.join(ROBOPAL_PATH, COLLECTIONS_DIR_NAME)


def play_demonstrations():
    for state_file in sorted(glob(DEFAULT_DATA_DIR_PATH)):

        # read .hdf5 files
        file = h5py.File(state_file + '/demo.hdf5','r')   
        logging.info("Reading from {}".format(state_file + '/demo.hdf5'))

        for group in file.keys():

            env_name = file["data"].attrs["env_name"]
            env_meta = ast.literal_eval(file["data"].attrs["env_kwargs"])
            logging.info(f"env meta: {env_meta}")

            env = robopal.make(
                env_name,
                **env_meta
            )

            for episode in file[group].keys():
                logging.info("\n>Reading group: {}, episode: {}".format(group, episode))

                # for key in file[group][episode].attrs:
                #     logging.info("{}: {}".format(key, file[group][episode].attrs[key]))

                env.load_model_from_string(file[group][episode].attrs["model_file"])

                first_state = file[group][episode]["states"][0]

                env.load_state(first_state)
                env.forward()

                env.update_init_pose_to_current()

                for action in file[group][episode]["actions"]:
                    env.step(action)
                    
            env.renderer.close()

if __name__ == "__main__":
    play_demonstrations()
