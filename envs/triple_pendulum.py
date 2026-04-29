"""Triple-Pendulum gym wrapper for DADS.

Matches url_benchmark/custom_dmc_tasks/triple_pendulum.py exactly:
  - Resets to hanging-down state (qpos=[π,0,0], qvel=[0,0,0]).
    foot_joint=π causes the whole chain to hang downward.
  - Observation: [sin(q1),sin(q2),sin(q3), cos(q1),cos(q2),cos(q3), v1,v2,v3]  (9-dim)
  - Tip height: (xpos['torso'] + R @ [0,0,0.4])[z]
    = torso body origin + tip of the torso geom (fromto="0 0 0 0 0 0.4").
    This is what val/scripts/cip/plots/plot_combined_height.py measures.
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


# foot_joint=π → leg hangs below foot pivot, thigh below leg, torso below thigh.
_INIT_QPOS = np.array([np.pi, 0.0, 0.0])
_INIT_QVEL = np.zeros(3)

# Local offset to torso geom tip in torso body frame (fromto "0 0 0 0 0 0.4").
_TORSO_TIP_LOCAL = np.array([0., 0., 0.4])


def _get_model_xml():
    xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'triple_pendulum.xml')
    return resources.GetResource(xml_path)


class _Physics(mujoco.Physics):
    def tip_height(self):
        """World z of the top of the torso geom.

        Computed as torso body origin + R @ [0,0,0.4], matching
        val's triple_pendulum_tip_heights / triple_pendulum_max_tip_height.
        Note: url_benchmark uses xpos['torso','z'] only (body origin, no offset).
        We use the tip here for consistency with val's normalisation baseline.
        """
        xpos = self.named.data.xpos['torso'].copy()
        xmat = self.named.data.xmat['torso'].copy().reshape(3, 3)
        return float((xpos + xmat @ _TORSO_TIP_LOCAL)[2])


class _TriplePendulumTask(base.Task):
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
        return float(physics.tip_height() > 2.0)


class TriplePendulumEnv(gym.Env):
    """Gym wrapper for the triple-pendulum swing-up task.

    Observation (9-dim flat):
        position — [sin(q1), sin(q2), sin(q3), cos(q1), cos(q2), cos(q3)]
        velocity — [vel1, vel2, vel3]

    Action (3-dim):
        torques for thigh, leg, foot joints in [-1, 1]
    """

    metadata = {'render.modes': []}

    def __init__(self):
        physics = _Physics.from_xml_string(_get_model_xml(), common.ASSETS)
        task = _TriplePendulumTask()
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
