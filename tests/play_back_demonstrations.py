import os
import inspect
from glob import glob
import logging

import h5py
import robopal
from robopal.demos.manipulation_tasks.demo_pick_place import PickAndPlaceEnv

ROBOPAL_PATH = os.path.dirname(inspect.getfile(robopal))
COLLECTIONS_DIR_NAME = 'collections/collections_*'
DEFAULT_DATA_DIR_PATH = os.path.join(ROBOPAL_PATH, COLLECTIONS_DIR_NAME)


for state_file in sorted(glob(DEFAULT_DATA_DIR_PATH)):

    # read .hdf5 files
    file_object = h5py.File(state_file + '/demo.hdf5','r')   
    logging.info("Reading from {}".format(state_file + '/demo.hdf5'))

    for group in file_object.keys():

        from robopal.robots.panda import PandaPickAndPlace

        env = PickAndPlaceEnv(
            robot=PandaPickAndPlace,
            render_mode="human",
            control_freq=20,
            controller='CARTIK',
        )

        for episode in file_object[group].keys():

            logging.info("\n>Reading group: {}, episode: {}".format(group, episode))

            for key in file_object[group][episode].attrs:
                logging.info("{}: {}".format(key, file_object[group][episode].attrs[key]))


            env.reset(seed=file_object[group][episode].attrs["seed"])
            env.load_model_from_string(file_object[group][episode].attrs["mjcf"])

            first_state = file_object[group][episode]["states"][0]
            env.load_state(first_state)

            for action in file_object[group][episode]["actions"]:
                env.step(action)
