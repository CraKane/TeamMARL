import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            self.warmup()
            self.restore()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, \
                coach_values, coach_actions, coach_action_log_probs, coach_rnn_states, coach_rnn_states_critic,\
                coach_rnn_states_obs = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                # print('obs = {} share_obs = {} rewards = {} dones = {} available_actions = {}'.format(obs.shape,share_obs.shape, rewards.shape, dones.shape,  available_actions.shape))
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, coach_values, coach_actions, coach_action_log_probs, \
                       coach_rnn_states, coach_rnn_states_critic, coach_rnn_states_obs
                
                # insert data into buffer
                self.insert(data)

            if (episode % 6 == 0 or episode == episodes - 1):
                for i in range(len(self.pers)):
                    self.restore()
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


            # eval
            if episode % self.eval_interval == 0:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic, \
        coach_value, coach_actions, coach_action_log_prob, coach_rnn_state, coach_rnn_state_critic, \
        rnn_states_obs = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              self.buffer.coach_rnn_states[step],
                                              self.buffer.coach_rnn_states_critic[step],
                                              np.concatenate(self.buffer.rnn_states_obs[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              available_actions=np.concatenate(self.buffer.available_actions[step]))

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        coach_values = np.array(np.split(_t2n(coach_value), self.n_rollout_threads))
        coach_action_log_probs = np.array(np.split(_t2n(coach_action_log_prob), self.n_rollout_threads))
        coach_rnn_states = _t2n(coach_rnn_state)
        coach_rnn_states_critic = _t2n(coach_rnn_state_critic)
        rnn_states_obs = np.array(np.split(_t2n(rnn_states_obs), self.n_rollout_threads))
        # print(values.shape, actions.shape, rnn_states.shape, rnn_states_critic.shape, coach_values.shape, coach_rnn_states.shape, coach_rnn_states_critic.shape, action_log_probs.shape, coach_action_log_probs.shape, coach_actions.shape)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, \
        coach_values, coach_actions, coach_action_log_probs, coach_rnn_states, coach_rnn_states_critic,\
        rnn_states_obs

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, \
        coach_values, coach_actions, coach_action_log_probs, coach_rnn_states, \
        coach_rnn_states_critic, rnn_states_obs = data

        dones_env = np.all(dones, axis=1)

        # If all env end, then rnn states need 0
        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        ## coach state
        coach_rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.hidden_size), dtype=np.float32)
        coach_rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.hidden_size), dtype=np.float32)

        rnn_states_obs[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        coach_masks = np.ones((self.n_rollout_threads, 1, 1), dtype=np.float32)
        coach_masks[dones_env == True] = np.zeros(((dones_env == True).sum(), 1, 1), dtype=np.float32)
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, coach_values, coach_actions, coach_action_log_probs, 
                           coach_rnn_states, coach_rnn_states_critic, rnn_states_obs, coach_masks, available_actions=available_actions)
    

    @torch.no_grad()
    def eval(self, total_num_steps):
        from itertools import combinations
        pers = list(combinations(range(self.num_agents + 2), self.num_agents))

        eval_episode_rewards = []
        eval_episode_win_rate = []

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for per in pers:

            episode_rewards = 0
            eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                
                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
                episode_rewards += eval_rewards

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            battles_won = []
            battles_game = []
            incre_battles_won = []
            incre_battles_game = [] 
            for i, info in enumerate(eval_infos):
                if 'battles_won' in info[0].keys():
                    battles_won.append(info[0]['battles_won'])
                    incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                if 'battles_game' in info[0].keys():
                    battles_game.append(info[0]['battles_game'])
                    incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

            incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
            eval_episode_win_rate.append(incre_win_rate)
            
            last_battles_game = battles_game
            last_battles_won = battles_won
     
            eval_episode_rewards.append(episode_rewards)


        eval_env_infos = {}
        eval_average_step_reward = np.mean(eval_episode_rewards) / self.episode_length
        eval_env_infos['eval_average_step_reward'] = np.array(eval_episode_rewards) / self.episode_length
        eval_win_rate = np.mean(eval_episode_win_rate)
        eval_env_infos["eval_win_rate"] = eval_episode_win_rate
        print("eval average step reward of agent: " + str(eval_average_step_reward))
        print("eval win rate is {}.".format(eval_win_rate))
        self.log_env(eval_env_infos, total_num_steps)