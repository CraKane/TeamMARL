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

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            self.restore()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % len(self.pers) == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.map_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won'] - last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game'] - last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(
                        incre_battles_game) > 0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x * y, list(
                    self.buffer.active_masks.shape))

                self.envs.eval_traverse()
                self.log_train(train_infos, total_num_steps, self.envs.traverse, self.envs.abs_traverse, self.envs.relative_traverse)

            # eval
            if episode % self.eval_interval == 0 and self.check_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        # self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),)
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks,
                           available_actions)

    def log_train(self, train_infos, total_num_steps, cover_rate, max_traverse, relative_cover_rate):
        # print('shape = {}'.format(np.shape(self.buffer.rewards[self.buffer.step])))
        if self.all_args.env_name == 'matrix_game':
            train_infos["average_step_rewards"] = np.sum(self.buffer.rewards[self.buffer.step])
            train_infos["cover_rate"] = cover_rate
            train_infos["max_traverse"] = max_traverse
            train_infos["relative_cover_rate"] = relative_cover_rate
        elif self.all_args.env_name == 'DiffGame':
            train_infos["average_step_rewards"] = np.mean(self.buffer.rewards[self.buffer.step])

        # train_infos["last_episode_sum"] = np.sum(self.buffer.rewards[self.buffer.step])
        print('last_episode_num = {}'.format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        from itertools import combinations
        pers = list(combinations(range(self.num_agents + 1), self.num_agents))
        eval_episode = 0

        eval_step_rewards = []
        eval_cover_rate = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        for per in pers:
            one_step_rewards = []
            

            eval_rnn_states = np.zeros(
                (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            # print('eval_obs = {}'.format(eval_obs.shape))
            self.restore(eval=True, team=per)
            for step in range(self.episode_length):
                eval_actions, eval_rnn_states = \
                    self.evl_policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                    eval_actions)

                one_step_rewards.append(eval_rewards)

                eval_dones_env = np.all(eval_dones, axis=1)
                # print('eval_dones = {}  eval_dones_env = {}'.format(eval_dones, eval_dones_env))
                eval_rnn_states[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                    dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                              dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_step_rewards.append(np.mean(one_step_rewards, axis=0))

            self.eval_envs.eval_traverse()
            eval_cover_rate.append(self.eval_envs.traverse)

        team_average_step_rewards = np.mean(eval_step_rewards)*self.num_agents
        team_average_cover_rate = np.mean(eval_cover_rate)
        eval_env_infos = {'team_average_step_rewards': team_average_step_rewards,
                          'team_average_cover_rate': team_average_cover_rate}
        self.log_env(eval_env_infos, total_num_steps)

    def log_env(self, eval_env_infos, total_num_steps):
        print('team_step_reward = {}'.format(eval_env_infos["team_average_step_rewards"]))
        for k, v in eval_env_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
