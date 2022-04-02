import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MatrixRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(MatrixRunner, self).__init__(config)

    def run(self, fix_action):
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        eval_step_rewards = []
        for episode in range(episodes):

            eval_episode = 0
            one_step_rewards = []
            eval_obs, eval_share_obs, eval_available_actions = self.envs.reset()

            eval_rnn_states = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32)
            eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            # print('eval_obs = {}'.format(eval_obs.shape))

            for step in range(self.episode_length):
                eval_actions, eval_rnn_states = \
                    self.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True,
                                        fix_action=fix_action)
                # print(eval_obs, eval_actions)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.envs.step(
                    eval_actions)


                one_step_rewards.append(eval_rewards)

                eval_dones_env = np.all(eval_dones, axis=1)
                # print('eval_dones = {}  eval_dones_env = {}'.format(eval_dones, eval_dones_env))
                eval_rnn_states[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                    dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                              dtype=np.float32)

            for eval_i in range(self.n_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_step_rewards.append(np.mean(one_step_rewards, axis=0))

        mean_step_rewards = np.mean(eval_step_rewards)
        print(mean_step_rewards)
        return mean_step_rewards