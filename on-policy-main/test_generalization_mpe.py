#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for MPEs."""

def make_train_env(all_args, seed):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, seed):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='navigation_control_full', help="Which scenario to run on")
    parser.add_argument("--collision_penal", type=float, default=0)
    parser.add_argument("--vision", type=float, default=1)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args, seed, i, fix_action=None, test_run=None, test_model=None, test_policy=None):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

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

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # env init
    envs = make_train_env(all_args, seed)
    eval_envs = make_eval_env(all_args, seed) if all_args.use_eval else None
    num_agents = all_args.num_agents
    all_args.n_agents = all_args.num_agents
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
    if all_args.share_policy:
        from onpolicy.runner.shared.test_mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    mean_reward = runner.run(test_policy, test_run)
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

    return mean_reward



if __name__ == "__main__":
    seeds = [2021]#, 2022, 114]#, 2, 2021114]
    test_policy_list = ['population_idv_mappo_turn_update', 'population_idv_mappo', 'population_idv_mappo_total_update']
    mean_rewards = []
    for i in range(len(seeds)):
        for test_policy in tqdm(test_policy_list):
            for j in range(1):
                num = 4 if test_policy == 'control_idv_mappo' else 6
                from itertools import combinations
                pers = list(combinations(range(num), 2))   # for my policy
                np.random.shuffle(pers)
                for k in pers:
                    mean_reward = main(sys.argv[1:], seeds[i], i + 1, test_run=j+1, test_model=k, test_policy=test_policy)
                    print('mean_episode_reward = {}, model = {}_run{}_{}'.format(mean_reward, test_policy, j + 1, k))
                    mean_rewards.append(mean_reward)

    av_reward = np.mean(mean_rewards)

    print(av_reward)