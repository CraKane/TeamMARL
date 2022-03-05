import os
import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic



def build(run_dir, num_agents, args, obs_space, sobs_space, act_space, device):
    # create model folder
    model_dir = str(run_dir / 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    idv_para = args.idv_para

    share_obs_space = sobs_space if args.use_centralized_V else obs_space

    policy_actor = [R_Actor(args, obs_space, act_space, device) for _ in range(num_agents+1)]
    policy_critic = [R_Critic(args, share_obs_space, device) for _ in range(num_agents+1)]

    if idv_para:
        for i in range(num_agents+1):
            torch.save(policy_actor[i].state_dict(), str(model_dir) + "/actor_{}.pt".format(i))
            torch.save(policy_critic[i].state_dict(), str(model_dir) + "/critic_{}.pt".format(i))


def build_ruled_based(n_act, args, obs_space, sobs_space, act_space, device):
    model_dir = "models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    idv_para = args.idv_para

    share_obs_space = sobs_space if args.use_centralized_V else obs_space

    policy_actor = [R_Actor(args, obs_space, act_space, device) for _ in range(n_act)]
    policy_critic = [R_Critic(args, share_obs_space, device) for _ in range(n_act)]

    if idv_para:
        for i in range(n_act):
            torch.save(policy_actor[i].state_dict(), str(model_dir) + "/ruled_actor_only_{}.pt".format(i))
            torch.save(policy_critic[i].state_dict(), str(model_dir) + "/ruled_critic_only_{}.pt".format(i))
