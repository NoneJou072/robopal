import numpy as np
import time
from robopal.utils.KDL_utils import KDL_utils
from robopal.utils.gym_wrapper import GymWrapper
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import TimeLimit

URDF_PATH = "/home/zhouhr/mujoco_diana/roboIMI/roboimi/assets/models/manipulators/DianaMed/DianaMed.urdf"


class Lma_hcm(object):
    def __init__(self, verbose=False):
        self.kin = KDL_utils(URDF_PATH)
        self.verbose = verbose
        # robot parameters
        self.Dof = 7
        self.init_qpos = np.zeros(self.Dof)
        self.current_qpos = np.zeros(self.Dof)
        self.target_qpos = np.zeros(self.Dof)

        # hyper-params
        self.max_ep_steps = 100
        self.eps_phi = 1e-6
        self.lamb = 0.01  # init
        self.v_d = 1.2  # default
        self.v_m = 1.2

        # lma params
        self.iter_times = 0
        self.prev_phi = None

        # training params
        self.action_dim = (2,)
        self.obs_dim = (2,)
        self.action_low = np.array([-1.0, 1.0])
        self.action_high = np.array([1.0, 1.0])

    def setInitQpos(self, qpos: np.array):
        self.init_qpos = qpos
        self.current_qpos = qpos

    def setTargetQpos(self, qpos: np.array):
        self.target_qpos = qpos

    def computeError(self, current_joints):
        """ Phi means the error between current pose and target pose, eg. e = Pd - P """
        current_T = self.kin.getEEtf(self.kin.setJntArray(current_joints))
        current_pos = np.array([current_T.p[0], current_T.p[1], current_T.p[2]])
        current_rot = current_T.M

        target_T = self.kin.getEEtf(self.kin.setJntArray(self.target_qpos))
        target_pos = np.array([target_T.p[0], target_T.p[1], target_T.p[2]])
        target_rot = target_T.M

        current_pos_err = target_pos - current_pos
        current_rot_err = (target_rot * (current_rot.Inverse())).GetRot()

        err = np.array([current_pos_err[0], current_pos_err[1], current_pos_err[2],
                        current_rot_err[0], current_rot_err[1], current_rot_err[2]])

        return err

    def computePhi(self, err):
        return err.T @ np.eye(6) @ err

    def computeDelta(self, lamb, qpos, err):
        jac = self.kin.setNumpyMat(self.kin.getJac(self.kin.setJntArray(qpos)))
        mat = jac.T.dot(jac) + lamb * np.eye(7)
        if np.linalg.det(mat) == 0:
            raise np.linalg.LinAlgError
        inv_mat = np.linalg.inv(mat)
        return inv_mat @ jac.T @ err

    def updatePhi(self, lamb):
        current_qpos = self.current_qpos.copy()
        err = self.computeError(current_qpos)
        delta_q = self.computeDelta(lamb, current_qpos, err)
        phi = self.computePhi(err)
        new_qpos = current_qpos + delta_q
        return phi, lamb, new_qpos

    def updateLambda(self):
        # phi, lamb, current_qpos = self.update(self.lamb * np.exp(-self.v))
        phi, lamb, qpos = self.updatePhi(self.lamb / (self.v_d + 1e-5))
        if phi - self.prev_phi < self.eps_phi:
            return lamb, qpos, phi, True

        phi, lamb, qpos = self.updatePhi(self.lamb)
        if phi - self.prev_phi < self.eps_phi:
            return lamb, qpos, phi, True

        # phi, lamb, qpos = self.update(self.lamb * np.exp(self.v))
        phi, lamb, qpos = self.updatePhi(self.lamb * self.v_m)
        if phi - self.prev_phi < self.eps_phi:
            return lamb, qpos, phi, True
        # success = False
        # while phi - self.prev_phi >= 0:
        #     _phi, _lamb, _qpos = self.updatePhi(lamb * self.v_m)
        #     if phi - self.prev_phi > 0:
        #         print("lamb:", _lamb)
        #         break
        #     phi = _phi
        #     lamb = _lamb
        #     qpos = _qpos
        # Failure
        return self.lamb, self.current_qpos, self.prev_phi, False

    def step(self, action):
        """ LMA computation """
        self.iter_times += 1
        self.v_d = (1 / 1 + np.exp(action[0])) * 100
        self.v_m = (1 / 1 + np.exp(action[1])) * 100

        done = False
        reward = 0.0

        step_start_time = time.time()
        self.lamb, self.current_qpos, self.prev_phi, success = self.updateLambda()
        step_cost_time = time.time() - step_start_time

        print(
            f"times={self.iter_times}, phi={self.prev_phi:.7f}, cost_time={step_cost_time:.5f}. lamb:{self.lamb}")

        # reward += 0.3 * (1 - 1e3 * step_cost_time) if (1 - 1e3 * step_cost_time) >= 0 else 0.0
        reward -= 0.3 * self.iter_times
        if self.prev_phi >= 0.1:
            reward += 0.5 * (1.0 - 1 * self.prev_phi)
        elif 0.1 > self.prev_phi >= 0.01:
            reward += 0.5 + (1.0 - 10 * self.prev_phi)
        elif 0.01 > self.prev_phi >= 0.001:
            reward += 1.5 + (1.0 - 100 * self.prev_phi)
        elif 0.001 > self.prev_phi >= 0.0001:
            reward += 2.5 + (1.0 - 1000 * self.prev_phi)
        elif 0.0001 > self.prev_phi >= 0.00001:
            reward += 3.5 + (1.0 - 10000 * self.prev_phi)
        elif 0.00001 > self.prev_phi >= 0.000001:
            reward += 4.5 + (1.0 - 10000 * self.prev_phi)
        else:
            # if self.prev_phi <= self.eps_phi:
            done = True
            reward += 10.0

        if success is False:
            reward = 0
        return self.get_obs(), reward, done, dict()

    def reset(self):
        self.setInitQpos(3.14 * np.random.uniform(low=-1.0, high=1.0, size=(7,)))
        self.setTargetQpos(3.14 * np.random.uniform(low=-1.0, high=1.0, size=(7,)))
        err = self.computeError(self.init_qpos)
        self.prev_phi = self.computePhi(err)
        self.iter_times = 0
        self.lamb = 0.01
        print("=====================================new episode==================================")
        if self.verbose:
            print("init phi =", self.prev_phi)
            print("init qpos =", self.init_qpos)
            print("goal qpos =", self.target_qpos)
            result = self.kin.getEEtf(self.kin.setJntArray(self.current_qpos))
            target = self.kin.getEEtf(self.kin.setJntArray(self.target_qpos))
            print("init T =", result)
            print("goal T =", target)
        return self.get_obs()

    def get_obs(self):
        obs = np.zeros(self.obs_dim)
        obs[0] = np.array([self.prev_phi], dtype=np.float32)
        obs[1] = np.array([self.lamb], dtype=np.float32)
        # obs[1:8] = self.current_qpos
        # obs[8:15] = self.target_qpos
        return obs


def run_test():
    env = Lma_hcm(verbose=False)
    env.reset()
    for n_steps in range(int(1e6)):
        a = np.array([0.5, 0.1])
        s_, r, d, _ = env.step(a)
        # print(r, d)
        if d or env.iter_times % env.max_ep_steps == 0:
            if env.verbose:
                result = env.kin.getEEtf(env.kin.setJntArray(env.current_qpos))
                target = env.kin.getEEtf(env.kin.setJntArray(env.target_qpos))
                print("compute_results  =", result)
                print("target_T = ", target)
            env.reset()


def run_train():
    env = Lma_hcm()
    env = GymWrapper(env)
    env = TimeLimit(env, max_episode_steps=env.env.max_ep_steps)

    # check_env(env)

    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, log_dir, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)
            self.log_dir = log_dir

        def _on_step(self) -> bool:
            if self.n_calls % 51200 == 0:
                print("Saving new best model")
                self.model.save(self.log_dir + f"/model_saved/PPO/LMA_{self.n_calls}")
            return True

    log_dir = "log/"

    model = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=log_dir,
        device="cuda",
    )
    model.learn(total_timesteps=int(1e7), callback=TensorboardCallback(log_dir=log_dir))
    model.save("lma_rl")


def run_play():
    env = Lma_hcm(verbose=False)
    model = PPO.load("log/model_saved/PPO_2/LMA_3788800.zip")
    obs = env.reset()
    for i in range(int(1e6)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print("a", action)
        if done:
            env.reset()


if __name__ == '__main__':
    # run_test()
    run_train()
    # run_play()
