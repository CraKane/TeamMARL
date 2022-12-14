#!/usr/bin/env python
import sys
import os
import pickle
# import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import pandas as pd
from onpolicy.config import get_config
from gym.spaces import Discrete
from tensorboardX import SummaryWriter
from tqdm import trange,tqdm

"""Train script for SMAC."""


def get_element(array, index):
    ret = array
    # print('array_shape = {}'.format(np.shape(array)))
    # print('index = {}'.format(index))
    for x in index:
        ret = ret[x]
        # print('x = {}, ret_shape = {}'.format(x,np.shape(ret)))
    return ret


def make_not_exist_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_mat_game_from_file(filename):
    with open(filename, 'rb') as f:
        matrix_para = pickle.load(f)
        r_mat = matrix_para['reward']

        trans_mat = matrix_para['trans_mat']
        end_state = np.zeros(np.shape(r_mat)[0])
        state_num = np.shape(r_mat)[0]
        max_episode_length = int(state_num * 1.33)
        init_state = 0
        env = MatrixGame(r_mat, trans_mat, init_state, end_state, max_episode_length, evaluate_mat=True)
        return env


class MatrixGame:
    def __init__(self, r_mat, trans_mat, init_state, end_state, max_episode_length, evaluate_mat=False):
        r_shape = np.shape(r_mat)
        self.r_mat = r_mat
        self.trans_mat = trans_mat
        self.state_num = r_shape[0]

        self.agent_num = len(r_shape) - 1
        self.action_num = r_shape[1]
        self.share_observation_space = []
        self.observation_space = []
        self.action_space = []
        for i in range(self.agent_num):
            self.action_space.append(Discrete(self.action_num))
            self.observation_space.append(self.state_num)
            self.share_observation_space.append(self.state_num)
        self.init_state = init_state
        self.now_state = init_state
        self.step_count = 0
        self.end_state = end_state
        self.max_episode_length = max_episode_length
        self.state_action_count = np.zeros_like(r_mat).reshape([self.state_num, -1])
        self.long_step_count = 0
        self.evaluate_mat = evaluate_mat
        self.traverse = 0
        self.abs_traverse = 0
        self.relative_traverse = 0

    def eval_traverse(self):
        # print('state_action_count = {}'.format(self.state_action_count))
        print('long_step_count = {}'.format(self.long_step_count))
        covered_count = (self.state_action_count > 0).sum()
        all_state_action = self.state_action_count.shape[0] * self.state_action_count.shape[1]
        traverse = covered_count / all_state_action
        max_traverse = min(self.long_step_count / all_state_action, 1)
        relative_traverse = covered_count / self.long_step_count
        print('abs_traverse = {} max_traverse = {} relative_traverse = {}'.format(traverse, max_traverse,
                                                                                  relative_traverse))
        freq_mat = self.state_action_count / self.long_step_count
        freq_mat = freq_mat.reshape(self.r_mat.shape)
        static_return = (freq_mat * self.r_mat).sum()
        print('static_return = {}'.format(static_return))
        self.traverse = traverse
        self.abs_traverse = max_traverse
        self.relative_traverse = relative_traverse
        return self.traverse, self.abs_traverse, self.relative_traverse

    def reset(self):
        self.now_state = self.init_state
        self.step_count = 0
        obs = self.get_obs()
        obs = np.expand_dims(obs, axis=0)
        share_obs = obs
        available_actions = self.get_avail_agent_actions_all()
        available_actions = np.expand_dims(available_actions, axis=0)
        return obs, share_obs, available_actions

    def reset_evaluate(self):
        self.long_step_count = 0
        self.state_action_count = np.zeros_like(self.state_action_count)

    def get_obs(self):
        obs = []
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        for i in range(self.agent_num):
            obs.append(state)
        return obs

    def get_ac_idx(self, action):
        idx = 0
        for a in action:
            idx = self.action_num * idx + a
            # print('idx = {} a = {}'.format(idx,a))
        return idx

    def get_state(self):
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        return state

    def step(self, action, evaluate=False):
        # print('step = {} action  = {}'.format(self.step_count))
        sa_index = []
        sa_index.append(self.now_state)
        # sa_index += action
        action = np.array(action).reshape(-1)
        for a in action:
            sa_index.append(a)
        # print('sa_index = {},action = {}'.format(sa_index, action))
        if not evaluate:
            ac_idx = self.get_ac_idx(action)
            self.state_action_count[self.now_state, ac_idx] += 1
            self.long_step_count += 1

        r = get_element(self.r_mat, sa_index)
        next_s_prob = get_element(self.trans_mat, sa_index)
        # print('sa_index = {} next_s_prob = {}'.format(sa_index,next_s_prob))
        next_state = np.random.choice(range(self.state_num), size=1, p=next_s_prob)[0]
        self.now_state = next_state
        self.step_count += 1

        done = self.end_state[self.now_state]
        if self.step_count >= self.max_episode_length:
            done = 1
        done_ret = np.ones([1, self.agent_num], dtype=bool) * done
        reward_ret = np.ones([1, self.agent_num, 1]) * r
        obs = self.get_obs()
        obs = np.expand_dims(obs, axis=0)
        share_obs = obs
        available_actions = self.get_avail_agent_actions_all()
        available_actions = np.expand_dims(available_actions, axis=0)
        info = [[{} for i in range(self.agent_num)]]
        return obs, share_obs, reward_ret, done_ret, info, available_actions

    def get_env_info(self):
        env_info = {}
        env_info["n_actions"] = self.action_num
        env_info["n_agents"] = self.agent_num
        env_info["state_shape"] = self.state_num
        env_info["obs_shape"] = self.state_num
        env_info["episode_limit"] = self.max_episode_length
        return env_info

    def get_avail_agent_actions_all(self):
        return np.ones([self.agent_num, self.action_num])

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.action_num)

    def get_model_info(self, state, action):
        sa_index = []
        sa_index.append(state)
        action = np.array(action)
        # print('action = {}'.format(action))
        for a in action:
            sa_index.append(a)
        r = get_element(self.r_mat, sa_index)
        next_s_prob = get_element(self.trans_mat, sa_index)
        # print('action = {} sa_index = {} self.trans_mat = {} next_s_prob = {}'.format(action,sa_index,self.trans_mat.shape, next_s_prob.shape  ))
        return r, next_s_prob

    def close(self):
        return


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]
    all_args.matrix_test = True
    all_args.env_name = 'matrix_game'
    return all_args


def main(args, seed, i, fix_action=None, test_run=None, test_model=None, test_policy=None):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = '../../results/control_idv_mappo/'
    run_dir = run_dir + 'run{}'.format(i)
    run_dir = Path(run_dir)

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        print("tensorboard log")

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # env
    all_args.map_name = 'random_matrix_game_3_symmetric'
    envs = make_mat_game_from_file('{}.pkl'.format(all_args.map_name))
    eval_envs = make_mat_game_from_file('{}.pkl'.format(all_args.map_name)) if all_args.use_eval else None
    env_info = envs.get_env_info()
    num_agents = env_info["n_agents"]
    all_args.num_agents = num_agents
    all_args.episode_length = env_info['episode_limit']
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "eval_policy_dir": '../../results/' + test_policy + '/run{}/models'.format(test_run) if test_run else None,
        "eval_policy_num": test_model if test_model is not None else None,
    }

    # run experiments
    from onpolicy.runner.shared.test_matrix_runner import MatrixRunner as Runner

    # create combination
    from itertools import combinations

    pers = list(combinations(range(num_agents), 2))
    np.random.shuffle(pers)

    mean_rewards = []
    for per in pers:
        print(per)
        config['agent_ids'] = per
        runner = Runner(config)
        mean_reward = runner.run(fix_action)
        mean_rewards.append(mean_reward)

        # post process
        envs.reset()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.reset()

        if all_args.use_wandb:
            run.finish()
        else:
            runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
            runner.writter.close()

    return np.mean(np.array(mean_rewards)) * num_agents


if __name__ == "__main__":
    seeds = [2021, 2022, 114, 2, 2021114]
    test_policy_list = ['population_idv_mappo_turn_update', 'population_idv_mappo', 'population_idv_mappo_total_update']
    for i in range(len(seeds)):
        print('seed = {}'.format(seeds[i]))
        mean_rewards = []
        for j in trange(5):
            mean_reward = main(sys.argv[1:], seeds[i], i + 1, fix_action=j)
            print('mean_episode_reward = {}, fix_action = {}'.format(mean_reward, j))
            mean_rewards.append(mean_reward)
        for test_policy in tqdm(test_policy_list):
            for j in range(5):
                num = 3 if test_policy == 'control_idv_mappo' else 4
                for k in range(num):
                    mean_reward = main(sys.argv[1:], seeds[i], i + 1, test_run=j+1, test_model=k, test_policy=test_policy)
                    print('mean_episode_reward = {}, model = {}_run{}_{}'.format(mean_reward, test_policy, j + 1, k))
                    mean_rewards.append(mean_reward)

        seed = np.ones(len(mean_rewards)) * (i + 1)
        if i == 0:
            res = pd.DataFrame({'seed': seed, 'mean_episode_reward': mean_rewards})
        else:
            tmp = pd.DataFrame({'seed': seed, 'mean_episode_reward': mean_rewards})
            res = res.append(tmp)
    print(res)
    res.to_csv('../../results/control_mappo_test_generalization_result.csv', index=False)
