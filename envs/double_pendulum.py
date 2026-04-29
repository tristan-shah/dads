"""Double-Pendulum gym wrapper for DADS.

Matches url_benchmark/custom_dmc_tasks/double_pendulum.py exactly:
  - Resets to hanging-down state (qpos=[0,0], qvel=[0,0]).
  - Observation: [sin(θ1), sin(θ2), cos(θ1), cos(θ2), vel1, vel2]  (6-dim)
  - Tip height: geom_xpos['bob2','z']
    (consistent with val/scripts/cip/plots/plot_combined_height.py)
"""

import collections
import os

import numpy as np
import gym
from gym import spaces

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.utils import io as resources


_INIT_QPOS = np.zeros(2)   # [hinge1=0, hinge2=0] → both rods hang straight down
_INIT_QVEL = np.zeros(2)


def _get_model_xml():
    xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'double_pendulum.xml')
    return resources.GetResource(xml_path)


class _Physics(mujoco.Physics):
    def tip_height(self):
        """World z of bob2 geom centre (matches url_benchmark Physics.tip_height)."""
        return self.named.data.geom_xpos['bob2', 'z']


class _DoublePendulumTask(base.Task):
    def initialize_episode(self, physics):
        physics.data.qpos[:] = _INIT_QPOS
        physics.data.qvel[:] = _INIT_QVEL
        physics.data.time = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        angles = physics.data.qpos.copy()
        obs['position'] = np.concatenate(
            [np.sin(angles), np.cos(angles)]
        ).astype(np.float32)
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        return float(physics.tip_height() > 3.5)


class DoublePendulumEnv(gym.Env):
    """Gym wrapper for the double-pendulum swing-up task.

    Observation (6-dim flat):
        position — [sin(θ1), sin(θ2), cos(θ1), cos(θ2)]
        velocity — [vel1, vel2]

    Action (1-dim):
        torque on hinge1 in [-1, 1]
    """

    metadata = {'render.modes': []}

    def __init__(self):
        physics = _Physics.from_xml_string(_get_model_xml(), common.ASSETS)
        task = _DoublePendulumTask()
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

    def tip_height(self):
        return float(self._env.physics.tip_height())

    def reset(self):
        time_step = self._env.reset()
        return self._flat_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._flat_obs(time_step)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        done = False
        return obs, reward, done, {'tip_height': self.tip_height()}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
