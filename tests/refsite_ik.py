import mujoco
from mujoco import viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(filename='diana_ik.xml', assets=None)
data = mujoco.MjData(model)
renderer = viewer.launch_passive(model, data)

mujoco.mj_resetDataKeyframe(model, data, 0)

while True:
    data.ctrl[0] += 0.0001
    mujoco.mj_step(model, data, nstep=20)
    renderer.sync()
