import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule,soft_update,hard_update
import numpy as np

def _t2n(x):
    return x.detach().cpu().numpy()
def idv_merge(x):
    if len(x[0].shape) == 0:
        return torch.tensor(x).to(x[0].device)
    x = torch.stack(x, dim=1)
    init_shape = x.shape
    x = x.reshape([-1, *init_shape[2:]])
    return x



class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        if self.args.idv_para:
            self.actor = [R_Actor(args, self.obs_space, self.act_space, self.device) for _ in range(self.args.num_agents)]
            self.critic = [R_Critic(args, self.share_obs_space, self.device) for _ in range(self.args.num_agents)]



            self.actor_optimizer = [torch.optim.Adam(self.actor[i].parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay) for i in range(self.args.num_agents)]

            self.critic_optimizer = [torch.optim.Adam(self.critic[i].parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay) for i in range(self.args.num_agents)]

            if self.args.target_dec:
                self.target_actor = [R_Actor(args, self.obs_space, self.act_space, self.device) for _ in range(self.args.num_agents)]
                self.hard_update_policy()
        else:
            self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
            self.critic = R_Critic(args, self.share_obs_space, self.device)

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay)
            if self.args.target_dec:
                self.target_actor = R_Actor(args, self.obs_space, self.act_space, self.device)
                self.hard_update_policy()

    def soft_update_policy(self,tau = 0.001):
        if self.args.idv_para:
            for i in range(self.args.num_agents):
                soft_update(self.target_actor[i],self.actor[i],tau)
        else:
            soft_update(self.target_actor,self.actor,tau)

    def hard_update_policy(self):
        if self.args.idv_para:
            for i in range(self.args.num_agents):
                hard_update(self.target_actor[i], self.actor[i])
        else:
            hard_update(self.target_actor,self.actor)
    def optim_reset(self,agent_id):
        if self.args.idv_para:
            self.actor_optimizer[agent_id] = torch.optim.Adam(self.actor[agent_id].parameters(),
                                                     lr=self.lr, eps=self.opti_eps,
                                                     weight_decay=self.weight_decay)

            self.critic_optimizer[agent_id] = torch.optim.Adam(self.critic[agent_id].parameters(),
                                                      lr=self.critic_lr,
                                                      eps=self.opti_eps,
                                                      weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        if self.args.idv_para:
            for i in range(self.args.num_agents):
                update_linear_schedule(self.actor_optimizer[i], episode, episodes, self.lr)
                update_linear_schedule(self.critic_optimizer[i], episode, episodes, self.critic_lr)
        else:
            update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
            update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def idv_reshape(self,x):
        # print('type = {} is_tensor = {}'.format(type(x),torch.is_tensor(x)))
        if torch.is_tensor(x):
            x = _t2n(x)
        return np.array(np.reshape(x, [-1, self.args.num_agents, *np.shape(x)[1:]]))
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, fix_action=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if self.args.idv_para:


            obs = self.idv_reshape(obs)
            cent_obs = self.idv_reshape(cent_obs)
            rnn_states_critic = self.idv_reshape(rnn_states_critic)

            rnn_states_actor = self.idv_reshape(rnn_states_actor)

            masks = self.idv_reshape(masks)
            # print('available_actions_prev = {}'.format(available_actions.shape))
            if available_actions is not None:
                available_actions = self.idv_reshape(available_actions)
            # print('available_actions_after = {}'.format(available_actions.shape))
            actions,action_log_probs,rnn_states_actor_list,values,rnn_states_critic_list = [],[],[],[],[]
            for i in range(self.args.num_agents):
                if available_actions is not None:
                    avail_input_i = available_actions[:, i, :]
                else:
                    avail_input_i = None
                if self.args.target_dec:
                    idv_actions, idv_action_log_probs, idv_rnn_states_actor = self.target_actor[i](obs[:,i,:],
                                                                                    rnn_states_actor[:,i,:],
                                                                                    masks[:,i,:],
                                                                                    avail_input_i,
                                                                                    deterministic)
                else:
                    idv_actions, idv_action_log_probs, idv_rnn_states_actor = self.actor[i](obs[:, i, :],
                                                                                                   rnn_states_actor[:,
                                                                                                   i, :],
                                                                                                   masks[:, i, :],
                                                                                                   avail_input_i,
                                                                                                   deterministic)
                idv_values, idv_rnn_states_critic = self.critic[i](cent_obs[:, i, :], rnn_states_critic[:, i, :], masks[:, i, :])
                if self.args.use_eval and i == self.args.num_agents - 1 and fix_action != None:
                    actions.append(torch.tensor([[fix_action]]).to('cuda:0'))
                else:
                    actions.append(idv_actions)
                action_log_probs.append(idv_action_log_probs)
                rnn_states_actor_list.append(idv_rnn_states_actor)
                values.append(idv_values)
                rnn_states_critic_list.append(idv_rnn_states_critic)


            actions = idv_merge(actions)
            action_log_probs = idv_merge(action_log_probs)
            rnn_states_actor = idv_merge(rnn_states_actor_list)
            rnn_states_critic = idv_merge(rnn_states_critic_list)
            values = idv_merge(values)
        else:
            if self.args.target_dec:
                actions, action_log_probs, rnn_states_actor = self.target_actor(obs,
                                                                         rnn_states_actor,
                                                                         masks,
                                                                         available_actions,
                                                                         deterministic)
            else:
                actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                         rnn_states_actor,
                                                                         masks,
                                                                         available_actions,
                                                                         deterministic)

            values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.args.idv_para:

            cent_obs = self.idv_reshape(cent_obs)
            rnn_states_critic = self.idv_reshape(rnn_states_critic)
            masks = self.idv_reshape(masks)
            values =  []
            for i in range(self.args.num_agents):
                idv_values,_ = self.critic[i](cent_obs[:, i, :], rnn_states_critic[:, i, :],
                                                                   masks[:, i, :])
                values.append(idv_values)

            def idv_merge(x):
                if len(x[0].shape) == 0:
                    return torch.tensor(x)
                x = torch.stack(x, dim=1)
                init_shape = x.shape
                x = x.reshape([-1, *init_shape[2:]])
                return x
            values = idv_merge(values)
        else:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.args.idv_para:


            obs = self.idv_reshape(obs)
            cent_obs = self.idv_reshape(cent_obs)
            action = self.idv_reshape(action)
            rnn_states_critic = self.idv_reshape(rnn_states_critic)

            rnn_states_actor = self.idv_reshape(rnn_states_actor)

            masks = self.idv_reshape(masks)
            # print('avail_act_prev = {}'.format(available_actions.shape))
            if available_actions is not None:
                available_actions = self.idv_reshape(available_actions)
            # print('avail_act_after = {}'.format(available_actions.shape))
            if  active_masks is not None:
                active_masks =  self.idv_reshape(active_masks)


            action_log_probs,dist_entropy, values = [], [], []
            for i in range(self.args.num_agents):
                # print('avail_act_after_for_{} = {}'.format(i,available_actions[:,i, :].shape))
                if available_actions is not None:
                    avail_input_i = available_actions[:, i, :]
                else:
                    avail_input_i = None
                if active_masks is not None:
                    active_masks_input_i = active_masks[:,i,:]
                else:
                    active_masks_input_i = None

                idv_action_log_probs, idv_dist_entropy = self.actor[i].evaluate_actions(obs[:, i, :],
                                                                                        rnn_states_actor[:,
                                                                                        i, :],
                                                                                        action[:,i,:],
                                                                                        masks[:, i, :],
                                                                                        avail_input_i,
                                                                                        active_masks_input_i)


                idv_values,_= self.critic[i](cent_obs[:, i, :], rnn_states_critic[:, i, :],
                                                                   masks[:, i, :])
                action_log_probs.append(idv_action_log_probs)
                dist_entropy.append(idv_dist_entropy)
                values.append(idv_values)


            # print('dist_entropy = {}'.format(dist_entropy))
            action_log_probs = idv_merge(action_log_probs)
            values = idv_merge(values )
            dist_entropy = idv_merge(dist_entropy)
            dist_entropy = dist_entropy.mean()
        else:
            action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                         rnn_states_actor,
                                                                         action,
                                                                         masks,
                                                                         available_actions,
                                                                         active_masks)

            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        # print('dist_entropy = {}'.format(dist_entropy))
        return values, action_log_probs, dist_entropy

    def evaluate_actions_single(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None,update_index = -1):
        action_log_probs, dist_entropy = self.actor[update_index].evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic[update_index](cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, fix_action=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        # if self.args.target_dec:
        #     actions, _, rnn_states_actor = self.target_actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        # else:
        if self.args.idv_para:

            obs = self.idv_reshape(obs)

            rnn_states_actor = self.idv_reshape(rnn_states_actor)
            masks = self.idv_reshape(masks)

            if available_actions is not None:
                available_actions = self.idv_reshape(available_actions)
            actions, action_log_probs, rnn_states_actor_list, values, rnn_states_critic = [], [], [], [], []
            for i in range(self.args.num_agents):
                if available_actions is not None:
                    avail_input_i = available_actions[:, i, :]
                else:
                    avail_input_i = None
                idv_actions, _, idv_rnn_states_actor = self.actor[i](obs[:, i, :],rnn_states_actor[:,i, :],\
                                                                                        masks[:, i, :],\
                                                                                        avail_input_i,\
                                                                                        deterministic)
                if self.args.use_eval and i == self.args.num_agents - 1 and fix_action != None:
                    actions.append(torch.tensor([[fix_action]]).to('cuda:0'))
                else:
                    actions.append(idv_actions)
                rnn_states_actor_list.append(idv_rnn_states_actor)


            actions = idv_merge(actions)
            rnn_states_actor = idv_merge(rnn_states_actor_list)
        else:
            actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
