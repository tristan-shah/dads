import collections
import os

import numpy as np
import gym
from gym import spaces

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.utils import rewards
from dm_control.utils import io as resources

_STAND_HEIGHT = 1.2

# Fixed initial state — agent starts lying on its back, no random noise.
# First 13 values: qpos. Next 13 values: qvel.
_INIT_STATE = np.array([
    -5.69669133e-03, -1.40173909e+00,  3.15282207e+00, -3.09313693e+00,
    -2.32511620e-01, -3.11384200e+00, -1.17470462e-01, -3.13217174e+00,
     6.28050424e+00, -3.14893727e+00, -3.13217174e+00,  6.28050424e+00,
    -3.14893727e+00,  6.99103214e-02, -6.08517271e-04, -1.43865156e-01,
    -2.64162545e-02,  6.20225465e-02, -1.94878478e-02,  3.07025207e-02,
    -1.04211144e-01,  9.53556737e-03,  9.83598035e-02, -1.04211144e-01,
     9.53556737e-03,  9.83598035e-02,
])


def _get_model_xml():
    xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'humulum.xml')
    return resources.GetResource(xml_path)


class _Physics(mujoco.Physics):
    def head_height(self):
        return self.named.data.xpos['head', 'z']

    def torso_upright(self):
        return self.named.data.xmat['torso', 'zz']


class _HumulumTask(base.Task):
    def initialize_episode(self, physics):
        nq = physics.model.nq
        physics.data.qpos[:] = _INIT_STATE[:nq]
        physics.data.qvel[:] = _INIT_STATE[nq:]
        physics.data.time = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[1:].copy()  # skip root_x
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        standing = rewards.tolerance(
            physics.head_height(),
            bounds=(_STAND_HEIGHT, float('inf')),
            margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        return (3 * standing + upright) / 4


class HumulumEnv(gym.Env):
    """Gym wrapper around the dm-control Humulum task.

    Uses dm-control + the new mujoco pip package — no manual binary install.
    Always resets from the fixed lying-down initial state; never adds noise.

    Observation (25-dim flat):
        position  — qpos[1:]  (12 dims, root_x omitted)
        velocity  — qvel      (13 dims)

    Action (9-dim):
        torques for waist, shoulders, elbows, hips, knees, ankles in [-1, 1]
    """

    metadata = {'render.modes': []}

    def __init__(self):
        physics = _Physics.from_xml_string(_get_model_xml(), common.ASSETS)
        task = _HumulumTask()
        # time_limit=inf: DADS wrap_env handles episode length via max_env_steps
        self._env = control.Environment(
            physics, task, time_limit=float('inf'),
            control_timestep=0.01)

        obs_spec = self._env.observation_spec()
        obs_size = sum(v.shape[0] for v in obs_spec.values())
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        action_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32)

    def _flat_obs(self, time_step):
        return np.concatenate(
            [v.ravel() for v in time_step.observation.values()]
        ).astype(np.float32)

    def head_height(self):
        return float(self._env.physics.head_height())

    def reset(self):
        time_step = self._env.reset()
        return self._flat_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._flat_obs(time_step)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        done = False  # episode length is controlled by DADS wrap_env (max_env_steps)
        return obs, reward, done, {'head_height': self.head_height()}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
