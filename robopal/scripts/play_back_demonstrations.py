import os
import inspect
from glob import glob
import logging
import json

import h5py
import numpy as np
import robopal
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
	MofNCompleteColumn,
    TimeRemainingColumn,
)

ROBOPAL_PATH = os.path.dirname(inspect.getfile(robopal))
COLLECTIONS_DIR_NAME = 'collections/collections_*'
DEFAULT_DATA_DIR_PATH = os.path.join(ROBOPAL_PATH, COLLECTIONS_DIR_NAME)


def play_all():
    for state_dir in sorted(glob(DEFAULT_DATA_DIR_PATH)):
        play_demonstrations(state_dir)

def play_demonstrations(demo_dir=None):

    # read .hdf5 files
    file = h5py.File(demo_dir + '/demo.hdf5','r')   
    logging.info("Reading from {}".format(demo_dir + '/demo.hdf5'))

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

    progress = Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            MofNCompleteColumn(),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeRemainingColumn(),
    )

    # playback each episode
    with progress:
        for id in progress.track(range(len(demos)), description="Reading episode:"):
            episode = demos[id]

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
    play_demonstrations("robopal/collections/collections_1719978428_4136665")
