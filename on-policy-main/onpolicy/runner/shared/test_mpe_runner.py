import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import matplotlib
import os
import gym
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Actor_old, R_Actor_role_3

def _t2n(x):
    return x.detach().cpu().numpy()

def idv_merge(x):
    if len(x[0].shape) == 0:
        return torch.tensor(x).to(x[0].device)
    x = torch.stack(x, dim=1)
    init_shape = x.shape
    x = x.reshape([-1, *init_shape[2:]])
    return x


class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

        self.obs_space = self.envs.observation_space[0]
        self.act_space = self.envs.action_space[0]

    def run(self, test_policy, test_run):

        # create combination
        from itertools import combinations

        # pers = list(combinations(range(self.num_agents+2), self.all_args.my_policy))
        # 3-1
        if self.all_args.my_policy == 3:
            my_policy_lists = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)]
            test_policy_lists = [(0,), (1,), (2,), (3,)]

        else:
            # 2-2
            my_policy_lists = [(2,3), (1,3), (1,2), (0,3), (0,2), (0,1)]
            test_policy_lists = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

        eval_episode_rewards = []


        for my_policy_list, test_policy_list in zip(my_policy_lists,test_policy_lists):
            print(my_policy_list, test_policy_list)
            self.restore(test_policy, my_policy_list, test_policy_list)

            count=0
            eval_obs = self.envs.reset()

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = 0

            for eval_step in range(self.episode_length):
                # results = self.envs.render(mode='rgb_array')
                
                # result = np.squeeze(results[0])
                # print(result)
                # matplotlib.image.imsave('../results/pics/{}_{}_{}_{}.png'.format(test_policy, test_run, per, count), result)

                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.act(test_policy, test_policy_list,
                                                    np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                
                if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(self.envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[eval_actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                episode_rewards += eval_rewards

                dones_env = np.all(eval_dones, axis=1)
                eval_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                count+=1

            eval_episode_rewards.append(episode_rewards)

        eval_episode_rewards = np.mean(eval_episode_rewards)
        print("eval average episode rewards of agent: " + str(eval_episode_rewards))

        return eval_episode_rewards


    def restore(self, test_model, my_policy_list, test_policy_list):
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
                self.actor.insert(test_policy_list[i], R_Actor_role_3(self.all_args, self.obs_space, self.act_space, self.device))


        # model data loading
        for my_policy in my_policy_list:
            print("my policy, " + str(self.model_dir) + "/actor_{}.pt".format(my_policy))
            actor_state_dict = torch.load(str(self.model_dir) + "/actor_{}.pt".format(my_policy))
            # print(policy_actor_state_dict)
            self.actor[my_policy].load_state_dict(actor_state_dict)
        if self.eval_dir is not None:
            rest_num = len(test_policy_list)
            for i in range(rest_num):
                print("other policy, " + self.eval_dir + "/actor_{}.pt\n".format(self.eval_num[i]))
                actor_state_dict = torch.load(self.eval_dir + "/actor_{}.pt".format(self.eval_num[i]))
                self.actor[test_policy_list[i]].load_state_dict(actor_state_dict)


    def idv_reshape(self,x):
        # print('type = {} is_tensor = {}'.format(type(x),torch.is_tensor(x)))
        if torch.is_tensor(x):
            x = _t2n(x)
        return np.array(np.reshape(x, [-1, self.all_args.num_agents, *np.shape(x)[1:]]))

    def act(self, test_model, test_policy_list, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, fix_action=None):
        obs = self.idv_reshape(obs)

        rnn_states_actor = self.idv_reshape(rnn_states_actor)
        # print(rnn_states_actor.shape)
        masks = self.idv_reshape(masks)

        if available_actions is not None:
            available_actions = self.idv_reshape(available_actions)
        actions, action_log_probs, rnn_states_actor_list, values, rnn_states_critic = [], [], [], [], []
        for i in range(self.num_agents):
            if available_actions is not None:
                avail_input_i = available_actions[:, i, :]
            else:
                avail_input_i = None
            if '4-3-' in test_model and i in test_policy_list:
                obs_22 = np.hstack((obs[:,i,:-self.num_agents*2-2], obs[:,i,-self.num_agents*2:]))
                idv_actions, _, idv_rnn_states_actor = self.actor[i].idv_act(obs_22,rnn_states_actor[:,i, :],\
                                                                                    masks[:, i, :],\
                                                                                    avail_input_i,\
                                                                                    deterministic)
            else:
                idv_actions, _, idv_rnn_states_actor = self.actor[i].idv_act(obs[:, i, :],rnn_states_actor[:,i, :],\
                                                                                    masks[:, i, :],\
                                                                                    avail_input_i,\
                                                                                    deterministic)
            actions.append(idv_actions)
            rnn_states_actor_list.append(idv_rnn_states_actor)


        actions = idv_merge(actions)
        rnn_states_actor = idv_merge(rnn_states_actor_list)
        return actions, rnn_states_actor