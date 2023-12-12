Renderers
==========

我们提供了便携的渲染方式，这些渲染方式同样完全遵循**gymnasium**的接口规范！

1. **human**
`render_mode="human"`

2. **rgb_array**
`render_mode="rgb_array"`

3. **depth**
`render_mode="depth"`

3. **unity(开发中)**
`render_mode="unity"`


Mujoco Renderer
-------------------------

.. autoclass:: robopal.commons.MjRenderer

  .. automethod:: key_callback
  .. automethod:: render
  .. automethod:: close
  .. automethod:: set_renderer_config
  .. automethod:: add_visual_point
  .. automethod:: render_pixels_from_camera
