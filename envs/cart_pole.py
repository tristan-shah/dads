"""Cart-Pole gym wrapper for DADS.

Matches url_benchmark/custom_dmc_tasks/cart_pole.py exactly:
  - Resets to hanging-down state (qpos=[0,0], qvel=[0,0]).
  - Observation: [cart_x, sin(θ), cos(θ), vel_cart, vel_pole]  (5-dim)
  - pole tip height: xpos['pole','z'] − xmat['pole','zz']
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


_INIT_QPOS = np.zeros(2)   # [cart_slide=0, pole_hinge=0] → pole hangs down
_INIT_QVEL = np.zeros(2)


def _get_model_xml():
    xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'cart_pole.xml')
    return resources.GetResource(xml_path)


class _Physics(mujoco.Physics):
    def pole_tip_height(self):
        """World z of the pole tip (matches url_benchmark Physics.pole_tip_height)."""
        return (self.named.data.xpos['pole', 'z'] -
                self.named.data.xmat['pole', 'zz'])


class _CartPoleTask(base.Task):
    def initialize_episode(self, physics):
        physics.data.qpos[:] = _INIT_QPOS
        physics.data.qvel[:] = _INIT_QVEL
        physics.data.time = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        pole_angle = physics.data.qpos[1]
        obs['position'] = np.array([
            physics.data.qpos[0],   # cart x
            np.sin(pole_angle),
            np.cos(pole_angle),
        ], dtype=np.float32)
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        return float(physics.pole_tip_height() > 2.0)


class CartPoleEnv(gym.Env):
    """Gym wrapper for the cart-pole swing-up task.

    Observation (5-dim flat):
        position — [cart_x, sin(θ), cos(θ)]
        velocity — [vel_cart, vel_pole]

    Action (1-dim):
        force on cart in [-1, 1]
    """

    metadata = {'render.modes': []}

    def __init__(self):
        physics = _Physics.from_xml_string(_get_model_xml(), common.ASSETS)
        task = _CartPoleTask()
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

    def pole_tip_height(self):
        return float(self._env.physics.pole_tip_height())

    def reset(self):
        time_step = self._env.reset()
        return self._flat_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._flat_obs(time_step)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        done = False
        return obs, reward, done, {'pole_tip_height': self.pole_tip_height()}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
