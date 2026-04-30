"""Evaluate DADS skills on the TriplePendulum environment.

Height metric: (xpos['torso'] + R @ [0,0,0.4])[z]
= tip of the torso geom (fromto="0 0 0 0 0 0.4") in world frame.
This matches val/scripts/cip/plots/plot_combined_height.py triple_pendulum_tip_heights.

Note: url_benchmark uses xpos['torso','z'] (body origin only, no tip offset).
We use the full tip here to be consistent with val's normalisation baseline.

Usage:
    python unsupervised_skill_learning/eval_triple_pendulum.py \\
        --logdir=/path/to/logdir \\
        --flagfile=configs/triple_pendulum_offpolicy.txt \\
        [--num_eval_skills=20] \\
        [--num_seeds=8] \\
        [--out_dir=./eval_triple_pendulum]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('./'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import app, flags, logging

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

import dads_agent

from envs import skill_wrapper
from envs import triple_pendulum as tp_env
from lib import py_tf_policy

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', '~/tmp/dads', 'Directory where training data was saved')
flags.DEFINE_string('environment', 'TriplePendulum', 'Environment name (must match training)')
flags.DEFINE_integer('max_env_steps', 2000, 'Episode length in gym steps')
flags.DEFINE_integer('reduced_observation', 0, 'Reduced observation flag (must match training)')
flags.DEFINE_integer('min_steps_before_resample', 5000, 'Unused in eval; must match training graph')
flags.DEFINE_float('resample_prob', 0., 'Unused in eval; must match training graph')
flags.DEFINE_integer('num_skills', 4, 'Number of skills (must match training)')
flags.DEFINE_string('skill_type', 'cont_uniform', 'Skill prior type (must match training)')
flags.DEFINE_integer('random_skills', 50, 'Must match training graph')
flags.DEFINE_integer('hidden_layer_size', 512, 'Network size (must match training)')
flags.DEFINE_string('save_model', 'dads', 'Model save name used during training')
flags.DEFINE_integer('save_freq', 50, 'Unused in eval')
flags.DEFINE_integer('record_freq', 100, 'Unused in eval')
flags.DEFINE_string('vid_name', None, 'Unused in eval')
flags.DEFINE_integer('run_eval', 0, 'Unused in eval')
flags.DEFINE_integer('num_evals', 0, 'Unused in eval')
flags.DEFINE_integer('deterministic_eval', 0, 'Unused in eval')
flags.DEFINE_integer('run_train', 0, 'Unused in eval')
flags.DEFINE_integer('num_epochs', 0, 'Unused in eval')
flags.DEFINE_integer('replay_buffer_capacity', int(1e6), 'Unused in eval')
flags.DEFINE_integer('clear_buffer_every_iter', 0, 'Unused in eval')
flags.DEFINE_integer('initial_collect_steps', 2000, 'Unused in eval')
flags.DEFINE_integer('collect_steps', 200, 'Unused in eval')
flags.DEFINE_string('agent_relabel_type', None, 'Must match training graph')
flags.DEFINE_integer('train_skill_dynamics_on_policy', 0, 'Must match training graph')
flags.DEFINE_string('skill_dynamics_relabel_type', None, 'Must match training graph')
flags.DEFINE_integer('num_samples_for_relabelling', 100, 'Must match training graph')
flags.DEFINE_float('is_clip_eps', 0., 'Must match training graph')
flags.DEFINE_float('action_clipping', 1., 'Clip actions during eval')
flags.DEFINE_integer('debug_skill_relabelling', 0, 'Unused in eval')
flags.DEFINE_integer('skill_dyn_train_steps', 8, 'Must match training graph')
flags.DEFINE_float('skill_dynamics_lr', 3e-4, 'Must match training graph')
flags.DEFINE_integer('skill_dyn_batch_size', 256, 'Must match training graph')
flags.DEFINE_integer('agent_batch_size', 256, 'Must match training graph')
flags.DEFINE_integer('agent_train_steps', 128, 'Must match training graph')
flags.DEFINE_float('agent_lr', 3e-4, 'Must match training graph')
flags.DEFINE_float('agent_entropy', 0.1, 'Must match training graph')
flags.DEFINE_float('agent_gamma', 0.99, 'Must match training graph')
flags.DEFINE_string('collect_policy', 'default', 'Must match training graph')
flags.DEFINE_string('graph_type', 'default', 'Must match training graph')
flags.DEFINE_integer('num_components', 4, 'Must match training graph')
flags.DEFINE_integer('fix_variance', 1, 'Must match training graph')
flags.DEFINE_integer('normalize_data', 1, 'Must match training graph')
flags.DEFINE_integer('debug', 0, 'Debug flag')
flags.DEFINE_integer('expose_last_action', 1, 'Unused for TriplePendulum')
flags.DEFINE_integer('expose_upright', 1, 'Unused for TriplePendulum')
flags.DEFINE_float('upright_threshold', 0.9, 'Unused for TriplePendulum')
flags.DEFINE_float('robot_noise_ratio', 0.05, 'Unused for TriplePendulum')
flags.DEFINE_float('root_noise_ratio', 0.002, 'Unused for TriplePendulum')
flags.DEFINE_float('scale_root_position', 1, 'Unused for TriplePendulum')
flags.DEFINE_integer('run_on_hardware', 0, 'Unused for TriplePendulum')
flags.DEFINE_float('randomize_hfield', 0.0, 'Unused for TriplePendulum')
flags.DEFINE_integer('observation_omission_size', 0, 'Unused for TriplePendulum')
flags.DEFINE_integer('randomized_initial_distribution', 0, 'Unused for TriplePendulum')
flags.DEFINE_float('horizontal_wrist_constraint', 1.0, 'Unused for TriplePendulum')
flags.DEFINE_float('vertical_wrist_constraint', 1.0, 'Unused for TriplePendulum')
flags.DEFINE_integer('planning_horizon', 1, 'Unused in eval')
flags.DEFINE_integer('primitive_horizon', 1, 'Unused in eval')
flags.DEFINE_integer('num_candidate_sequences', 50, 'Unused in eval')
flags.DEFINE_integer('refine_steps', 10, 'Unused in eval')
flags.DEFINE_float('mppi_gamma', 10.0, 'Unused in eval')
flags.DEFINE_string('prior_type', 'normal', 'Unused in eval')
flags.DEFINE_float('smoothing_beta', 0.9, 'Unused in eval')
flags.DEFINE_integer('top_primitives', 5, 'Unused in eval')

flags.DEFINE_integer('num_eval_skills', 20, 'Number of random skills to evaluate')
flags.DEFINE_integer('num_seeds', 8, 'Number of stochastic seeds for the best-skill plot')
flags.DEFINE_string('out_dir', './eval_triple_pendulum', 'Directory to save output plots and npy files')
flags.DEFINE_integer('eval_seed', 42, 'RNG seed for sampling eval skills')


def _normal_projection_net(action_spec, init_means_output_factor=0.1):
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


def _build_eval_env(preset_skill, inner_env=None):
    if inner_env is None:
        inner_env = tp_env.TriplePendulumEnv()
    wrapped = skill_wrapper.SkillWrapper(
        inner_env,
        num_latent_skills=FLAGS.num_skills,
        skill_type=FLAGS.skill_type,
        preset_skill=preset_skill,
        min_steps_before_resample=FLAGS.min_steps_before_resample,
        resample_prob=0.0)
    return wrap_env(wrapped, max_episode_steps=FLAGS.max_env_steps), inner_env


def _build_agent(py_env, save_dir, global_step):
    py_action_spec = py_env.action_spec()
    tf_action_spec = tensor_spec.from_spec(py_action_spec)
    env_obs_spec = py_env.observation_spec()
    agent_obs_spec = env_obs_spec
    py_agent_time_step_spec = ts.time_step_spec(agent_obs_spec)
    tf_agent_time_step_spec = tensor_spec.from_spec(py_agent_time_step_spec)

    skill_dynamics_observation_size = (
        env_obs_spec.shape[0] - FLAGS.num_skills)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_agent_time_step_spec.observation,
        tf_action_spec,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        continuous_projection_net=_normal_projection_net)

    critic_net = critic_network.CriticNetwork(
        (tf_agent_time_step_spec.observation, tf_action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=(FLAGS.hidden_layer_size,) * 2)

    reweigh_batches_flag = (
        FLAGS.skill_dynamics_relabel_type is not None
        and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type
        and FLAGS.is_clip_eps > 1.0)

    agent = dads_agent.DADSAgent(
        save_dir,
        skill_dynamics_observation_size,
        observation_modify_fn=lambda obs: obs,
        restrict_input_size=0,
        latent_size=FLAGS.num_skills,
        latent_prior=FLAGS.skill_type,
        prior_samples=FLAGS.random_skills,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        normalize_observations=FLAGS.normalize_data,
        network_type=FLAGS.graph_type,
        num_mixture_components=FLAGS.num_components,
        fix_variance=FLAGS.fix_variance,
        reweigh_batches=reweigh_batches_flag,
        skill_dynamics_learning_rate=FLAGS.skill_dynamics_lr,
        time_step_spec=tf_agent_time_step_spec,
        action_spec=tf_action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        target_update_tau=0.005,
        target_update_period=1,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.agent_lr),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.agent_lr),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.agent_lr),
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=FLAGS.agent_gamma,
        reward_scale_factor=1. / (FLAGS.agent_entropy + 1e-12),
        gradient_clipping=None,
        debug_summaries=FLAGS.debug,
        train_step_counter=global_step)

    return agent


def _run_episode(py_env, policy, inner_env):
    """Run one episode; return array of torso tip heights at each step."""
    time_step = py_env.reset()
    heights = []
    while not time_step.is_last():
        heights.append(inner_env.tip_height())
        action_step = policy.action(time_step)
        if FLAGS.action_clipping < 1.:
            action_step = action_step._replace(
                action=np.clip(action_step.action,
                               -FLAGS.action_clipping, FLAGS.action_clipping))
        time_step = py_env.step(action_step.action)
    return np.array(heights)


def main(_):
    tf.compat.v1.enable_resource_variables()
    tf.compat.v1.disable_eager_execution()
    logging.set_verbosity(logging.INFO)

    root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
    log_dir = os.path.join(root_dir, FLAGS.environment)
    save_dir = os.path.join(log_dir, 'models')
    out_dir = os.path.abspath(FLAGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dummy_inner = tp_env.TriplePendulumEnv()
    dummy_env, _ = _build_eval_env(
        preset_skill=np.zeros(FLAGS.num_skills, dtype=np.float32),
        inner_env=dummy_inner)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = _build_agent(dummy_env, save_dir, global_step)
    agent.build_agent_graph()
    agent.build_skill_dynamics_graph()
    agent.create_savers()

    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'agent'),
        agent=agent,
        global_step=global_step)

    eval_policy = py_tf_policy.PyTFPolicy(agent.policy)
    collect_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)

    with tf.compat.v1.Session().as_default() as sess:
        train_checkpointer.initialize_or_restore(sess)
        agent.set_sessions(initialize_or_restore_skill_dynamics=True, session=sess)

        step = sess.run(global_step)
        print(f'Loaded checkpoint at step {step}')

        rng = np.random.RandomState(FLAGS.eval_seed)
        eval_skills = rng.uniform(-1.0, 1.0,
                                  size=(FLAGS.num_eval_skills, FLAGS.num_skills)).astype(np.float32)

        print(f'\nRunning {FLAGS.num_eval_skills} deterministic skill episodes...')
        skill_heights = {}
        for idx, skill in enumerate(eval_skills):
            py_env, inner_env = _build_eval_env(preset_skill=skill)
            heights = _run_episode(py_env, eval_policy, inner_env)
            skill_heights[idx] = heights
            print(f'  skill {idx:2d}: {len(heights)} steps, '
                  f'mean={heights.mean():.3f}m, max={heights.max():.3f}m')

        best_idx = max(skill_heights, key=lambda i: skill_heights[i].mean())
        best_heights = skill_heights[best_idx]
        best_skill = eval_skills[best_idx]
        print(f'\nBest skill index: {best_idx} (mean={best_heights.mean():.3f}m)')

        print(f'\n{"Skill":>6}  {"Mean":>8}  {"Max":>8}  {"Final":>8}')
        for i, h in sorted(skill_heights.items(), key=lambda x: -x[1].mean()):
            marker = ' <-- best' if i == best_idx else ''
            print(f'{i:>6}  {h.mean():>8.3f}  {h.max():>8.3f}  {h[-1]:>8.3f}{marker}')

        # best-skill stochastic seeds — shape (num_seeds, num_steps)
        print(f'\nRunning {FLAGS.num_seeds} stochastic seeds for best skill...')
        trajectories = []
        for seed in range(FLAGS.num_seeds):
            np.random.seed(seed)
            py_env, inner_env = _build_eval_env(preset_skill=best_skill)
            heights = _run_episode(py_env, collect_policy, inner_env)
            trajectories.append(heights)
            print(f'  seed {seed}: {len(heights)} steps, mean={heights.mean():.3f}m')

        arr = np.array(trajectories)
        bs_npy = os.path.join(out_dir, 'best_skill_seeds.npy')
        np.save(bs_npy, arr)
        print(f'Saved trajectories {arr.shape} → {bs_npy}')

        max_len = max(len(h) for h in skill_heights.values())
        padded = [np.pad(h, (0, max_len - len(h)), constant_values=h[-1])
                  for h in skill_heights.values()]
        padded_arr = np.array(padded)
        mean_heights = padded_arr.mean(axis=0)
        std_heights = padded_arr.std(axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f'DADS TriplePendulum — step {step}')
        ax1.plot(best_heights, color='tab:blue')
        ax1.set_title(f'Best skill (idx {best_idx}, mean={best_heights.mean():.3f} m)')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Torso tip height (m)')
        steps = np.arange(max_len)
        ax2.plot(steps, mean_heights, color='tab:orange', label='mean')
        ax2.fill_between(steps, mean_heights - std_heights,
                         mean_heights + std_heights,
                         color='tab:orange', alpha=0.3, label='±1 std')
        ax2.set_title(f'Average over {FLAGS.num_eval_skills} skills')
        ax2.set_xlabel('Step')
        ax2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'tip_height.png'), dpi=150)
        np.save(os.path.join(out_dir, 'tip_height.npy'), best_heights)
        print(f'Saved tip-height plot → {out_dir}/tip_height.png')
        plt.close(fig)

        mean_h = arr.mean(axis=0)
        std_h = arr.std(axis=0)
        ep_steps = np.arange(arr.shape[1])
        fig, ax = plt.subplots(figsize=(9, 5))
        cmap = plt.get_cmap('tab10')
        for i, h in enumerate(trajectories):
            ax.plot(np.arange(len(h)), h, color=cmap(i), alpha=0.7, label=f'seed {i}')
        ax.plot(ep_steps, mean_h, color='k', linewidth=2, label='mean')
        ax.fill_between(ep_steps, mean_h - std_h, mean_h + std_h,
                        color='k', alpha=0.15, label='±1 std')
        ax.set_title(f'DADS TriplePendulum step {step} — best skill {best_idx} '
                     f'({FLAGS.num_seeds} stochastic seeds)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Torso tip height (m)')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'best_skill_seeds.png'), dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    app.run(main)
