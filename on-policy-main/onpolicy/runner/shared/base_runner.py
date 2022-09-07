import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer
# create combination
from itertools import combinations

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']  

        if self.all_args.env_name == "MPE":
            pool = self.num_agents + 2
        else:
            pool = self.num_agents + 1

        self.pers = list(combinations(range(pool), self.num_agents))     

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        # print('self.episode_length = {}'.format(self.episode_length))
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        if self.use_eval and self.all_args.env_name != "MPE":
            self.agent_ids = config['agent_ids']

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # dir
        # self.model_dir = self.all_args.model_dir
        self.model_dir = str(self.run_dir / 'models')

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        # print('self.envs.action_space[0] = {}, self.envs.action_space = {}'.format(self.envs.action_space[0],self.envs.action_space))
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        self.evl_policy = Policy(self.all_args,
                             self.eval_envs.observation_space[0],
                             share_observation_space,
                             self.eval_envs.action_space[0],
                             device=self.device)

        self.eval_dir = None
        if self.use_eval and config['eval_policy_dir'] is not None:
            self.eval_dir = config['eval_policy_dir']
            self.eval_num = config['eval_policy_num']

        # if self.model_dir is not None:
        #     self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values, coach_next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                self.buffer.coach_rnn_states_critic[-1],
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        coach_next_values = np.array(np.split(_t2n(coach_next_values), self.n_rollout_threads))
        # print(next_values.shape, coach_next_values.shape)

        # print(next_values.shape, coach_next_values.shape)
        self.buffer.compute_returns(next_values, coach_next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        policy_critic = self.trainer.policy.critic
        if self.all_args.idv_para:
            for i in range(self.num_agents):
                torch.save(policy_actor[i].state_dict(), str(self.save_dir) + "/actor_{}.pt".format(self.agent_ids[i]))
                torch.save(policy_critic[i].state_dict(), str(self.save_dir) + "/critic_{}.pt".format(self.agent_ids[i]))
        else:
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self, eval = False, team=(0, 1, 2)):
        """Restore policy's networks from a saved model."""
        # np.random.shuffle(self.pers)
        if not self.use_eval:
            self.agent_ids = self.pers[np.random.randint(len(self.pers))]
        if self.all_args.idv_para:
            if eval:
                for i in range(self.num_agents):
                    policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor_{}.pt".format(team[i]))
                    self.evl_policy.actor[i].load_state_dict(policy_actor_state_dict)
                    if not self.all_args.use_render:
                        policy_critic_state_dict = torch.load(
                            str(self.model_dir) + "/critic_{}.pt".format(team[i]))
                        self.evl_policy.critic[i].load_state_dict(policy_critic_state_dict)
            else:
                for i in range(self.num_agents):
                    # print("my policy, " + str(self.model_dir) + "/actor_{}.pt in position {}".format(self.agent_ids[i], i))
                    policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor_{}.pt".format(self.agent_ids[i]))
                    # print(policy_actor_state_dict)
                    self.policy.actor[i].load_state_dict(policy_actor_state_dict)
                    if not self.all_args.use_render:
                        policy_critic_state_dict = torch.load(str(self.model_dir) + "/critic_{}.pt".format(self.agent_ids[i]))
                        self.policy.critic[i].load_state_dict(policy_critic_state_dict)
                
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
