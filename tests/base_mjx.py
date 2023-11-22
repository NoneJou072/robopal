import jax
import mujoco
from mujoco import mjx

XML = r"""
<mujoco>
<worldbody>
<body>
<freejoint/>
<geom size=".15" mass="1" type="sphere"/>
</body>
</worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.device_put(model)


@jax.vmap
def batched_step(vel):
    mjx_data = mjx.make_data(mjx_model)
    qvel = mjx_data.qvel.at[0].set(vel)
    mjx_data = mjx_data.replace(qvel=qvel)
    pos = mjx.step(mjx_model, mjx_data).qpos[0]
    return pos


vel = jax.numpy.arange(0.0, 1.0, 0.01)
pos = jax.jit(batched_step)(vel)
print(pos)
