# Behavior Cloning

## 收集 | Collect human demonstrarions

## 回放 | Playback the demonstations

## 训练 | [Robomimic](https://robomimic.github.io/)

## Dataset Structure

**robopal** 的 HDF5 数据集文件遵循 [**robomimic**](https://robomimic.github.io/docs/datasets/overview.html#dataset-structure) 的数据集结构，无需处理即可用于 robomimic 的训练。
需要注意，robopal 无法使用来自 **robosuite/robomimic** 的数据集，在基于 robopal 的环境中进行训练，只能使用来自 robopal 的数据集。

A single dataset is a single HDF5 file with the following structure:

<details>
  <summary><b>HDF5 Structure <span style="color:red;">(click to expand)</span></b></summary>
<p>

- **`data`** (group)

  - **`total`** (attribute) - number of state-action samples in the dataset

  - **`env_args`** (attribute) - a json string that contains metadata on the environment and relevant arguments used for collecting data. Three keys: `env_name`, the name of the environment or task to create, `env_type`, one of robomimic's supported [environment types](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_base.py#L9), and `env_kwargs`, a dictionary of keyword-arguments to be passed into the environment of type `env_name`.

  - **`demo_0`** (group) - group for the first trajectory (every trajectory has a group)

    - **`num_samples`** (attribute) - the number of state-action samples in this trajectory

    - **`model_file`** (attribute) - the xml string corresponding to the MJCF MuJoCo model.

    - **`states`** (dataset) - flattened raw MuJoCo states, ordered by time. Shape (N, D) where N is the length of the trajectory, and D is the dimension of the state vector.

    - **`actions`** (dataset) - environment actions, ordered by time. Shape (N, A) where N is the length of the trajectory, and A is the action space dimension

    - **`rewards`** (dataset) - environment rewards, ordered by time. Shape (N,) where N is the length of the trajectory.

    - **`dones`** (dataset) - done signal, equal to 1 if playing the corresponding action in the state should terminate the episode. Shape (N,) where N is the length of the trajectory.

    - **`obs`** (group) - group for the observation keys. Each key is stored as a dataset.

      - **`<obs_key_1>`** (dataset) - the first observation key. Note that the name of this dataset and shape will vary. As an example, the name could be "agentview_image", and the shape could be (N, 84, 84, 3). 

        ...

    - **`next_obs`** (group) - group for the next observations.

      - **`<obs_key_1>`** (dataset) - the first observation key.

        ...

  - **`demo_1`** (group) - group for the second trajectory

    ...
    
- **`mask`** (group) - this group will exist in hdf5 datasets that contain filter keys

  - **`<filter_key_1>`** (dataset) - the first filter key. Note that the name of this dataset and length will vary. As an example, this could be the "valid" filter key, and contain the list ["demo_0", "demo_19", "demo_35"], corresponding to 3 validation trajectories.

    ...

</p>
</details>
