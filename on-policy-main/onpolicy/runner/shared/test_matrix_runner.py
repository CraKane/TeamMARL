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

    def run(self, fix_action, test_policy, test_run):

        if fix_action is not None:
            my_policy_lists = [(0,1)]
            test_policy_lists = [(2,)]
        else:
            my_policy_lists = [(1,2), (0,2), (0,1)]
            test_policy_lists = [(0,), (1,), (2,)]

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        eval_step_rewards = []
        
        for my_policy_list, test_policy_list in zip(my_policy_lists,test_policy_lists):
            print(my_policy_list, test_policy_list)
            self.restore(test_policy, my_policy_list, test_policy_list) 
            eval_obs = self.envs.reset()    
            for episode in range(episodes):

                eval_episode = 0
                one_step_rewards = []
                eval_obs, eval_share_obs, eval_available_actions = self.envs.reset()

                eval_rnn_states = np.zeros(
                    (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
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
                        ((eval_dones_env == True).sum(), self.num_agents, self.hidden_size),
                        dtype=np.float32)

                    eval_masks = np.ones((self.all_args.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                                  dtype=np.float32)

                for eval_i in range(self.n_rollout_threads):
                    if eval_dones_env[eval_i]:
                        eval_episode += 1
                        eval_step_rewards.append(np.mean(one_step_rewards, axis=0))

        mean_step_rewards = np.mean(eval_step_rewards)
        # print(mean_step_rewards)
        return mean_step_rewards
    
    def restore(test_model, my_policy_list, test_policy_list):
        # agent model loading
        self.actor = [R_Actor(self.all_args, self.obs_space, self.act_space, self.device) for _ in range(self.all_args.my_policy)]
        rest_num = self.num_agents - self.all_args.my_policy
        if '4-3-' in test_model:
            obs_4agents_3targets = gym.spaces.box.Box(-np.inf,np.inf, (22,))      
            for i in range(rest_num):
                self.actor.insert(test_policy_list[i], R_Actor_old(self.all_args, obs_4agents_3targets, self.act_space, self.device))
        elif test_model == '4-4-mappo':
            for i in range(rest_num):
                self.actor.insert(test_policy_list[i], R_Actor_old(self.all_args, self.obs_space, self.act_space, self.device))
        else:
            for i in range(rest_num):
                self.actor.insert(test_policy_list[i], R_Actor(self.all_args, self.obs_space, self.act_space, self.device))


        # model data loading
        for my_policy in my_policy_list:
            print("my policy, " + str(self.model_dir) + "/actor_{}.pt".format(my_policy))
            actor_state_dict = torch.load(str(self.model_dir) + "/actor_{}.pt".format(my_policy))
            # print(policy_actor_state_dict)
            self.actor[my_policy].load_state_dict(actor_state_dict)
        if self.eval_dir is not None:
            rest_num = len(test_policy_list)
            for i in range(rest_num):
                print("other policy, " + self.eval_dir + "/actor_{}.pt".format(self.eval_num[i]))
                actor_state_dict = torch.load(self.eval_dir + "/actor_{}.pt".format(self.eval_num[i]))
                self.actor[test_policy_list[i]].load_state_dict(actor_state_dict)


