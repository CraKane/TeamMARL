import torch
import torch.nn as nn
import numpy as np
from torch.nn import init as tinit
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer, ACTLayer_old
from onpolicy.utils.util import get_shape_from_obs_space
import torch.distributions as D
from onpolicy.algorithms.utils.distributions import DiagGaussian, StrategyGaussian

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Input and Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tinit.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    tinit.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                tinit.constant_(m.weight, 1)
                tinit.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                tinit.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    tinit.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.latent_dim = args.latent_dim
        self.var_floor = args.var_floor
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        activation_func=nn.LeakyReLU()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.act_feat = init_(nn.Linear(self.hidden_size, self.hidden_size//2))

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # role module
        # self.obs_embedding = nn.GRUCell(self.hidden_size//2, self.hidden_size)

        # orthogonal initialization
        # for name, param in self.obs_embedding.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     elif 'weight' in name:
        #         if self._use_orthogonal:
        #             nn.init.orthogonal_(param)
        #         else:
        #             nn.init.xavier_uniform_(param)

        self.role_out = StrategyGaussian(self.hidden_size//2, self.latent_dim, self._use_orthogonal, self._gain, device=device)

        self.latent_net = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_size//2),
                                        activation_func)


        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, role_dis, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param obs_traj: (np.ndarray / torch.Tensor) observation inputs with previous trajectory into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # print(rnn_states, rnn_states.shape)
        masks = check(masks).to(**self.tpdv)
        role_dis = check(role_dis).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print(actor_features.shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # print("role embedding layer: ", role_dis.shape)
        role_feature = self.latent_net(role_dis)
        

        # role_feature = role_feature.view(-1, self.hidden_size)
        # print("role feature", role_feature.shape)

        ##### role embedding end


        actions, action_log_probs = self.act(actor_features, role_feature, available_actions, deterministic)
        # print(actions.shape, action_log_probs.shape)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, role_dis, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        role_dis = check(role_dis).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print('obs = {}, actor_feature = {}, avail_action = {}'.format(obs.shape,actor_features.shape,available_actions.shape))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        # print('obs = {}, actor_feature = {}, avail_action = {}'.format(obs.shape, actor_features.shape,
        #                                                                available_actions.shape))

        # role embedding
        role_feature = self.latent_net(role_dis)
        # print("role embedding layer: ", latent.shape)

        # role_feature = role_feature.view(-1, self.hidden_size)
        # print("role feature", role_feature.shape)

        ##### role embedding end

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   role_feature,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy

    def idv_act(self, obs, rnn_states, masks, available_actions=None, deterministic=False):

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # print(rnn_states, rnn_states.shape)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print(actor_features.shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # role embedding
        # h_obs = self.obs_embedding(actor_features, rnn_states)
        # print(actor_features.shape, rnn_states.shape, h_obs.shape)


        """
        # print(self.latent.shape, self.latent)
        self.latent[:, -self.latent_dim:] = torch.clamp(torch.exp(self.latent[:, -self.latent_dim:]), min=self.var_floor)  # var
        # print(self.latent, self.latent.shape)
        latent_embed = self.latent.reshape(-1, self.latent_dim * 2)
        # print(latent_embed.shape)
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        latent = gaussian_embed.mode() if deterministic else gaussian_embed.rsample()
        # print(gaussian_embed, latent.shape)
        """
        gaussian_embed = self.role_out(actor_features)
        latent = gaussian_embed.mode() if deterministic else gaussian_embed.sample()
        # print(latent.shape)

        role_feature = self.latent_net(latent)
        # role_feature = role_feature.view(-1, self.hidden_size)
        # print("role feature", role_feature.shape)

        ##### role embedding end
        actions, action_log_probs = self.act(actor_features, role_feature, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def output_gaussian_for_update(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # print(rnn_states, rnn_states.shape)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print(actor_features.shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # role embedding
        # h_obs = self.obs_embedding(actor_features, rnn_states)
        # print(actor_features.shape, rnn_states.shape, h_obs.shape)

        """
        # print(self.latent.shape, self.latent)
        self.latent[:, -self.latent_dim:] = torch.clamp(torch.exp(self.latent[:, -self.latent_dim:]), min=self.var_floor)  # var
        # print(self.latent, self.latent.shape)
        latent_embed = self.latent.reshape(-1, self.latent_dim * 2)
        # print(latent_embed.shape)
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        """
        gaussian_embed = self.role_out(actor_features)
        return gaussian_embed

class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.act_feat = init_(nn.Linear(self.hidden_size, self.hidden_size//2))

        self.v_out = init_(nn.Linear(self.hidden_size//2, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # print("sssssssssss", cent_obs.shape)
        cent_obs = check(cent_obs).to(**self.tpdv)
        # print("aaaaaaaaaaaaaaaaaaaa")
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.act_feat(self.base(cent_obs))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states


class Coach_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(Coach_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_agents = args.num_agents
        self._gain = args.gain
        self.latent_dim = args.latent_dim
        self.var_floor = args.var_floor
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        activation_func=nn.LeakyReLU()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.act_feat = init_(nn.Linear(self.hidden_size, self.hidden_size//2))

        # meta module
        self.state_embedding = nn.GRUCell(self.hidden_size//2*self.num_agents, self.hidden_size)
        self.obs_embedding = nn.GRUCell(self.hidden_size//2, self.hidden_size)

        # orthogonal initialization
        for name, param in self.state_embedding.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        # orthogonal initialization
        for name, param in self.obs_embedding.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        # self.strategy = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                                activation_func,
        #                                nn.Linear(self.hidden_size, self.latent_dim * 2*self.num_agents))

        ### Attention module:
        ###### input: (batch_size, num_agents, hidden_size*2[128])
        ###### output: (batch_size, num_agents, hidden_size*2[128])
        self.obs_attention = ScaledDotProductAttention(d_model=self.hidden_size*2, d_k=self.hidden_size, d_v=self.hidden_size, h=3)


        self.action_out = StrategyGaussian(self.hidden_size*2, self.latent_dim, self._use_orthogonal, self._gain, device=device)

        self.add_module('action_out', self.action_out)
        self.add_module('feature_attention', self.obs_attention)

        self.to(device)

    def forward(self, obs, rnn_states_team, rnn_states_actor):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs of all agents into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states_team = check(rnn_states_team).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))

        h_obses = self.obs_embedding(actor_features[0], rnn_states_actor[0])

        actor_features = actor_features.view(actor_features.shape[0], -1)
        # print(actor_features.shape)
        h_state_team = self.state_embedding(actor_features, rnn_states_team)  # (1*hidden size )
        # print(rnn_states.shape, h_state_team.shape)


        """
        self.latent = self.strategy(h_state_team)   # (1, self.latent_dim*2*num_agents)
        # print(self.latent.shape, self.latent)
        self.latent[:, -self.latent_dim*self.num_agents:] = torch.clamp(torch.exp(self.latent[:, -self.latent_dim*self.num_agents:]), min=self.var_floor)  # var
        # print(self.latent, self.latent.shape)

        latent_embed_mean = self.latent[:, :self.latent_dim*self.num_agents]   # (1, self.latent_dim*num_agents)
        latent_embed_std = self.latent[:, -self.latent_dim*self.num_agents:]   # (1, self.latent_dim*num_agents)

        # print(latent_embed_mean.shape, latent_embed_std.shape)
        # for i in range(self.num_agents):
        #     print(latent_embed_mean[:, self.latent_dim*i:self.latent_dim*(i+1)])
        #     print(latent_embed_std[:, self.latent_dim*i:self.latent_dim*(i+1)])

        gaussian_embed = [D.Normal(latent_embed_mean[:, self.latent_dim*i:self.latent_dim*(i+1)],   # (num_agents with D)
            (latent_embed_std[:, self.latent_dim*i:self.latent_dim*(i+1)]) ** (1 / 2)) for i in range(self.num_agents)]
        """


        ## attention module
        actor_emb = torch.stack([torch.cat([h_state_team, h_obses[i].unsqueeze(0)], 1) for i in range(self.num_agents)])
        actor_emb = torch.transpose(actor_emb, 0, 1)
        actor_attention = self.obs_attention(actor_emb, actor_emb, actor_emb)


        gaussian_embed = [self.action_out(actor_emb[:,i,:]) for i in range(self.num_agents)]
        # print(self.obs_embedding.state_dict())

        actions = [gaussian_embed[i].sample() for i in range(self.num_agents)]  # (num_agents, 1, 3)
        # print(actions, gaussian_embed)
        # print([gaussian_embed[i].log_prob(actions[i]) for i in range(self.num_agents)])
        action_log_probs = torch.stack([gaussian_embed[i].log_prob(actions[i]) for i in range(self.num_agents)])
        # print("forward")
        # print(action_log_probs, actions, gaussian_embed[1].mean, gaussian_embed[1].scale)

        # print(action_log_probs.shape)
        action_log_probs = torch.sum(torch.sum(action_log_probs, 0), 1).reshape(-1, 1)
        # print(action_log_probs.shape)
        actions = torch.cat(actions)

        return actions, action_log_probs, h_state_team, h_obses

    def evaluate_actions(self, obs, rnn_states_team, rnn_states_actor, actions):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        # print(obs.shape)
        rnn_states_team = check(rnn_states_team).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))

        h_obses = [self.obs_embedding(actor_features[:,i,:], rnn_states_actor[:,i,:]) for i in range(self.num_agents)]
        actor_features = actor_features.view(actor_features.shape[0], -1)
        # print(actor_features.shape)
        h_state_team = self.state_embedding(actor_features, rnn_states_team)  # (1*hidden size )
        # print(rnn_states.shape, h_state_team.shape)

        """
        self.latent = self.strategy(h_state_team)   # (1, self.latent_dim*2*num_agents)
        # print(self.latent.shape)
        self.latent[:, -self.latent_dim*self.num_agents:] = torch.clamp(torch.exp(self.latent[:, -self.latent_dim*self.num_agents:]), min=self.var_floor)  # var
        # print(self.latent, self.latent.shape)

        latent_embed_mean = self.latent[:, :self.latent_dim*self.num_agents]   # (1, self.latent_dim*num_agents)
        latent_embed_std = self.latent[:, -self.latent_dim*self.num_agents:]   # (1, self.latent_dim*num_agents)

        # print(latent_embed_mean.shape, latent_embed_std.shape)
        # for i in range(self.num_agents):
        #     print(latent_embed_mean[:, self.latent_dim*i:self.latent_dim*(i+1)])
        #     print(latent_embed_std[:, self.latent_dim*i:self.latent_dim*(i+1)])

        gaussian_embed = [D.Normal(latent_embed_mean[:, self.latent_dim*i:self.latent_dim*(i+1)],   # (num_agents with D)
            (latent_embed_std[:, self.latent_dim*i:self.latent_dim*(i+1)]) ** (1 / 2)) for i in range(self.num_agents)]
        # print(gaussian_embed)
        """

        actor_emb = torch.stack([torch.cat([h_state_team, h_obses[i]], 1) for i in range(self.num_agents)])
        actor_emb = torch.transpose(actor_emb, 0, 1)
        actor_attention = self.obs_attention(actor_emb, actor_emb, actor_emb)

        
        gaussian_embed = [self.action_out(actor_emb[:,i,:]) for i in range(self.num_agents)]

        action_log_probs = torch.stack([gaussian_embed[i].log_prob(actions[:, i,:]) for i in range(self.num_agents)])
        # print("evaluation action")
        # print(action_log_probs, actions, gaussian_embed)

        # print(action_log_probs.shape)
        action_log_probs = torch.sum(torch.sum(action_log_probs, 0), 1).reshape(-1, 1)
        # print(action_log_probs.shape)

        dist_entropy = torch.stack([gaussian_embed[i].entropy() for i in range(self.num_agents)])
        dist_entropy = torch.sum(dist_entropy, 0).reshape(-1, 1)
        # print(dist_entropy.shape)

        return action_log_probs, dist_entropy

    def output_gaussian_for_update(self, obs, rnn_states_team, rnn_states_actor):
        obs = check(obs).to(**self.tpdv)
        rnn_states_team = check(rnn_states_team).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print(actor_features.shape)

        h_obses = [self.obs_embedding(actor_features[:,i,:], rnn_states_actor[:,i,:]) for i in range(self.num_agents)]
        actor_features = actor_features.view(actor_features.shape[0], -1)
        # print(actor_features.shape)
        h_state_team = self.state_embedding(actor_features, rnn_states_team)  # (1*hidden size )

        """
        self.latent = self.strategy(h_state_team)   # (1, self.latent_dim*2*num_agents)
        # print(self.latent.shape, self.latent)
        self.latent[:, -self.latent_dim*self.num_agents:] = torch.clamp(torch.exp(self.latent[:, -self.latent_dim*self.num_agents:]), min=self.var_floor)  # var
        # print(self.latent, self.latent.shape)

        latent_embed_mean = self.latent[:, :self.latent_dim*self.num_agents]   # (1, self.latent_dim*num_agents)
        latent_embed_std = self.latent[:, -self.latent_dim*self.num_agents:]   # (1, self.latent_dim*num_agents)

        # print(latent_embed_mean.shape, latent_embed_std.shape)
        # for i in range(self.num_agents):
        #     print(latent_embed_mean[:, self.latent_dim*i:self.latent_dim*(i+1)])
        #     print(latent_embed_std[:, self.latent_dim*i:self.latent_dim*(i+1)])

        gaussian_embed = [D.Normal(latent_embed_mean[:, self.latent_dim*i:self.latent_dim*(i+1)],   # (num_agents with D)
            (latent_embed_std[:, self.latent_dim*i:self.latent_dim*(i+1)]) ** (1 / 2)) for i in range(self.num_agents)]
        """
        actor_emb = torch.stack([torch.cat([h_state_team, h_obses[i]], 1) for i in range(self.num_agents)])
        actor_emb = torch.transpose(actor_emb, 0, 1)
        actor_attention = self.obs_attention(actor_emb, actor_emb, actor_emb)

        
        gaussian_embed = [self.action_out(actor_emb[:,i,:]) for i in range(self.num_agents)]
        return gaussian_embed


class Coach_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space)  observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(Coach_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_agents = args.num_agents
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.act_feat = init_(nn.Linear(self.hidden_size, self.hidden_size//2))

        # meta module
        self.obs_embedding = nn.GRUCell(self.hidden_size//2*self.num_agents, self.hidden_size)

        # orthogonal initialization
        for name, param in self.obs_embedding.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, obs, rnn_states):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        critic_features = self.act_feat(self.base(obs))

        critic_features = critic_features.view(critic_features.shape[0], -1)
        h_state_team = self.obs_embedding(critic_features, rnn_states)
        # print(h_state_team.shape, rnn_states.shape)
        values = self.v_out(h_state_team)
        # print(values.shape)

        return values, h_state_team



class R_Actor_old(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_old, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer_old(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        # print('obs = {}, actor_feature = {}, avail_action = {}'.format(obs.shape,actor_features.shape,available_actions.shape))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        # print('obs = {}, actor_feature = {}, avail_action = {}'.format(obs.shape, actor_features.shape,
        #                                                                available_actions.shape))

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy

    def idv_act(self, obs, rnn_states, masks, available_actions=None, deterministic=False):

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # print(rnn_states, rnn_states.shape)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

    
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

class R_Actor_role_3(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_role_3, self).__init__()
        self.hidden_size = args.hidden_size
        self.latent_dim = 3
        self.var_floor = args.var_floor
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        activation_func=nn.LeakyReLU()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.act_feat = init_(nn.Linear(self.hidden_size, self.hidden_size//2))

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # role module
        # self.obs_embedding = nn.GRUCell(self.hidden_size//2, self.hidden_size)

        # orthogonal initialization
        # for name, param in self.obs_embedding.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     elif 'weight' in name:
        #         if self._use_orthogonal:
        #             nn.init.orthogonal_(param)
        #         else:
        #             nn.init.xavier_uniform_(param)

        self.role_out = StrategyGaussian(self.hidden_size//2, self.latent_dim, self._use_orthogonal, self._gain, device=device)

        self.latent_net = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_size//2),
                                        activation_func)


        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, role_dis, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param obs_traj: (np.ndarray / torch.Tensor) observation inputs with previous trajectory into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # print(rnn_states, rnn_states.shape)
        masks = check(masks).to(**self.tpdv)
        role_dis = check(role_dis).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print(actor_features.shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # print("role embedding layer: ", role_dis.shape)
        role_feature = self.latent_net(role_dis)
        

        # role_feature = role_feature.view(-1, self.hidden_size)
        # print("role feature", role_feature.shape)

        ##### role embedding end


        actions, action_log_probs = self.act(actor_features, role_feature, available_actions, deterministic)
        # print(actions.shape, action_log_probs.shape)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, role_dis, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        role_dis = check(role_dis).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print('obs = {}, actor_feature = {}, avail_action = {}'.format(obs.shape,actor_features.shape,available_actions.shape))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        # print('obs = {}, actor_feature = {}, avail_action = {}'.format(obs.shape, actor_features.shape,
        #                                                                available_actions.shape))

        # role embedding
        role_feature = self.latent_net(role_dis)
        # print("role embedding layer: ", latent.shape)

        # role_feature = role_feature.view(-1, self.hidden_size)
        # print("role feature", role_feature.shape)

        ##### role embedding end

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   role_feature,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy

    def idv_act(self, obs, rnn_states, masks, available_actions=None, deterministic=False):

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # print(rnn_states, rnn_states.shape)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print(actor_features.shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # role embedding
        # h_obs = self.obs_embedding(actor_features, rnn_states)
        # print(actor_features.shape, rnn_states.shape, h_obs.shape)


        """
        # print(self.latent.shape, self.latent)
        self.latent[:, -self.latent_dim:] = torch.clamp(torch.exp(self.latent[:, -self.latent_dim:]), min=self.var_floor)  # var
        # print(self.latent, self.latent.shape)
        latent_embed = self.latent.reshape(-1, self.latent_dim * 2)
        # print(latent_embed.shape)
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        latent = gaussian_embed.mode() if deterministic else gaussian_embed.rsample()
        # print(gaussian_embed, latent.shape)
        """
        gaussian_embed = self.role_out(actor_features)
        latent = gaussian_embed.mode() if deterministic else gaussian_embed.sample()
        # print(latent.shape)

        role_feature = self.latent_net(latent)
        # role_feature = role_feature.view(-1, self.hidden_size)
        # print("role feature", role_feature.shape)

        ##### role embedding end
        actions, action_log_probs = self.act(actor_features, role_feature, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def output_gaussian_for_update(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # print(rnn_states, rnn_states.shape)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.act_feat(self.base(obs))
        # print(actor_features.shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # role embedding
        # h_obs = self.obs_embedding(actor_features, rnn_states)
        # print(actor_features.shape, rnn_states.shape, h_obs.shape)

        """
        # print(self.latent.shape, self.latent)
        self.latent[:, -self.latent_dim:] = torch.clamp(torch.exp(self.latent[:, -self.latent_dim:]), min=self.var_floor)  # var
        # print(self.latent, self.latent.shape)
        latent_embed = self.latent.reshape(-1, self.latent_dim * 2)
        # print(latent_embed.shape)
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        """
        gaussian_embed = self.role_out(actor_features)
        return gaussian_embed