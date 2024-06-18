import importlib.util
import robopal.commons.rich_logger

if importlib.util.find_spec("mujoco") is None:
    raise ImportError("Mujoco not installed. Please install it using `pip install mujoco`")

if importlib.util.find_spec("pettingzoo") is None:
    raise ImportError("PettingZoo not installed. Please install it using `pip install pettingzoo`")

if importlib.util.find_spec("gymnasium") is None:
    raise ImportError("Gym not installed. Please install it using `pip install gymnasium`")

from robopal.envs.base import make

__version__ = "0.4.1"
