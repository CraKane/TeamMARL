import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.popart import PopArt
from onpolicy.algorithms.utils.util import check
from torch.distributions import kl_divergence

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()   # only compute the active agents
        else:
            value_loss = value_loss.mean()

        return value_loss


    def coach_cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    
    def ppo_update(self, sample, update_actor=True,aga_update_tag = False,update_index = -1):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, coach_actions_batch, coach_old_action_log_probs_batch,\
        coach_values_batch, coach_returns_batch, coach_rnn_states_batch, coach_rnn_states_critic_batch,\
        coach_rnn_states_obs_batch, coach_masks_batch, coach_adv_targ = sample


        # print('share_obs_batch = {}'.format(share_obs_batch.shape))
        # print('obs_batch = {}'.format(share_obs_batch.shape))

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        # print('active_masks = {}'.format(active_masks_batch.shape))
        # Reshape to do in a single forward pass for all steps
        # print('aga_update_tag = {}, update_index = {}'.format(aga_update_tag,update_index))
        if self.args.idv_para and aga_update_tag and self.args.aga_tag:
            # print('case1')
            values, action_log_probs, dist_entropy,\
            coach_values, coach_action_log_probs, coach_dist_entropy = self.policy.evaluate_actions_single(share_obs_batch,
                                                                                  obs_batch,
                                                                                  rnn_states_batch,
                                                                                  rnn_states_critic_batch,
                                                                                  actions_batch,
                                                                                  masks_batch,
                                                                                  coach_rnn_states_batch, 
                                                                                  coach_rnn_states_critic_batch,
                                                                                  coach_rnn_states_obs_batch,
                                                                                  coach_actions_batch,
                                                                                  available_actions_batch,
                                                                                  active_masks_batch,update_index=update_index)
        else:
            # print('case2')
            values, action_log_probs, dist_entropy,\
            coach_values, coach_action_log_probs, coach_dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                                  obs_batch,
                                                                                  rnn_states_batch,
                                                                                  rnn_states_critic_batch,
                                                                                  actions_batch,
                                                                                  masks_batch,
                                                                                  coach_rnn_states_batch, 
                                                                                  coach_rnn_states_critic_batch,
                                                                                  coach_rnn_states_obs_batch,
                                                                                  coach_actions_batch,
                                                                                  available_actions_batch,
                                                                                  active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # print(imp_weights.shape)


        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
        # print(policy_loss, policy_loss.shape, dist_entropy, dist_entropy.shape)

        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.actor_optimizer[update_index].zero_grad()
            else:
                for i in range(self.args.num_agents):
                    self.policy.actor_optimizer[i].zero_grad()
        else:
            self.policy.actor_optimizer.zero_grad()

        if update_actor:
            # print((policy_loss - dist_entropy * self.entropy_coef), policy_loss, dist_entropy)
            (policy_loss - dist_entropy * self.entropy_coef).backward()
            # for p in self.policy.actor[0].latent_net.parameters():
            #     if p is not None:
            #         print("grad: ", p.grad)

        if self._use_max_grad_norm:
            if self.args.idv_para:
                if aga_update_tag and self.args.aga_tag:
                    actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor[update_index].parameters(), self.max_grad_norm)
                else:
                    actor_grad_norm = 0
                    for i in range(self.args.num_agents):
                        idv_actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor[i].parameters(),
                                                                   self.max_grad_norm)
                        actor_grad_norm += idv_actor_grad_norm
                    actor_grad_norm /= self.args.num_agents

            else:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            if self.args.idv_para:
                if aga_update_tag and self.args.aga_tag:
                    actor_grad_norm = get_gard_norm(self.policy.actor[update_index].parameters())
                else:
                    actor_grad_norm = 0
                    for i in range(self.args.num_agents):
                        idv_actor_grad_norm = get_gard_norm(self.policy.actor[i].parameters())
                        actor_grad_norm += idv_actor_grad_norm
                    actor_grad_norm /= self.args.num_agents

            else:
                actor_grad_norm = get_gard_norm(self.policy.actor.parameters())




        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.actor_optimizer[update_index].step()
            else:
                for i in range(self.args.num_agents):
                    self.policy.actor_optimizer[i].step()
        else:
            self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.critic_optimizer[update_index].zero_grad()
            else:
                for i in range(self.args.num_agents):
                    self.policy.critic_optimizer[i].zero_grad()
        else:
            self.policy.critic_optimizer.zero_grad()

        # print("value loss: ", value_loss)
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            if self.args.idv_para:
                if aga_update_tag and self.args.aga_tag:
                    critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic[update_index].parameters(),
                                                               self.max_grad_norm)
                else:
                    critic_grad_norm = 0
                    for i in range(self.args.num_agents):
                        idv_critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic[i].parameters(),
                                                                       self.max_grad_norm)
                        critic_grad_norm += idv_critic_grad_norm
                    critic_grad_norm /= self.args.num_agents

            else:
                critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            if self.args.idv_para:
                if aga_update_tag and self.args.aga_tag:
                    critic_grad_norm = get_gard_norm(self.policy.critic[update_index].parameters())
                else:
                    critic_grad_norm = 0
                    for i in range(self.args.num_agents):
                        idv_critic_grad_norm = get_gard_norm(self.policy.critic[i].parameters())
                        critic_grad_norm += idv_critic_grad_norm
                    critic_grad_norm /= self.args.num_agents

            else:
                critic_grad_norm = get_gard_norm(self.policy.critic.parameters())


        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.critic_optimizer[update_index].step()
            else:
                for i in range(self.args.num_agents):
                    self.policy.critic_optimizer[i].step()
        else:
            self.policy.critic_optimizer.step()



        ######################### coach Learning #############################################################


        coach_old_action_log_probs_batch = check(coach_old_action_log_probs_batch).to(**self.tpdv)
        coach_adv_targ = check(coach_adv_targ).to(**self.tpdv)
        coach_values_batch = check(coach_values_batch).to(**self.tpdv)
        coach_returns_batch = check(coach_returns_batch).to(**self.tpdv)
        
        
        # coach actor update
        coach_imp_weights = torch.exp(coach_action_log_probs - coach_old_action_log_probs_batch)
        # print("coach policy loss:")
        # print(coach_action_log_probs, coach_old_action_log_probs_batch)

        coach_surr1 = coach_imp_weights * coach_adv_targ
        coach_surr2 = torch.clamp(coach_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * coach_adv_targ
        # print(surr1.shape, coach_surr1.shape)

        
        coach_policy_action_loss = -torch.sum(torch.min(coach_surr1, coach_surr2), dim=-1, keepdim=True).mean()
        # print(coach_policy_action_loss, coach_surr1.shape, coach_surr2.shape)
        # print("entropy: ", coach_dist_entropy, self.entropy_coef)
        

        coach_policy_loss = coach_policy_action_loss
        # print("final loss: ", (coach_policy_loss - coach_dist_entropy * self.entropy_coef))
        # print()

        
        # print(self.policy.coach_actor.state_dict())
        # print(coach_action_log_probs.is_leaf, coach_dist_entropy.is_leaf)
        self.policy.coach_actor_optimizer.zero_grad()

        # print(coach_policy_loss - coach_dist_entropy * self.entropy_coef, coach_policy_loss, coach_dist_entropy)
        if update_actor:
            (coach_policy_loss - coach_dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            coach_actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.coach_actor.parameters(), self.max_grad_norm)
        else:
            coach_actor_grad_norm = get_gard_norm(self.policy.coach_actor.parameters())

        self.policy.coach_actor_optimizer.step()


        # coach critic update
        coach_value_loss = self.coach_cal_value_loss(coach_values, coach_values_batch, coach_returns_batch)

        # print("coach critic loss: ", coach_value_loss.shape, coach_value_loss)

        self.policy.coach_critic_optimizer.zero_grad()

        # print("coach value loss: ", coach_value_loss)
        (coach_value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            coach_critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.coach_critic.parameters(), self.max_grad_norm)
        else:
            coach_critic_grad_norm = get_gard_norm(self.policy.coach_critic.parameters())


        self.policy.coach_critic_optimizer.step()


        ##################################### Mutual Information Learning ########################################

        gaussian_embed, gaussian_infer = self.policy.output_gaussian_for_update(obs_batch, rnn_states_batch, \
            coach_rnn_states_batch, coach_rnn_states_obs_batch, masks_batch)
        # print(len(gaussian_infer), len(gaussian_embed))
        # print(gaussian_embed, gaussian_infer)
        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.role_optimizer[update_index].zero_grad()
            else:
                for i in range(self.args.num_agents):
                    self.policy.role_optimizer[i].zero_grad()
        else:
            self.policy.role_optimizer.zero_grad()


        kl_loss = torch.tensor(0).to(**self.tpdv)
        for i in range(self.args.num_agents):
            # print("mean: ", gaussian_embed[i].mean, gaussian_infer[i].mean)
            # print("std: ", gaussian_embed[i].scale, gaussian_infer[i].scale)
            loss = kl_divergence(gaussian_infer[i], gaussian_embed[i]).sum(dim=-1).mean()
            loss = torch.clamp(loss, max=2e3)
            kl_loss += loss
        
        # print(kl_loss)
        (kl_loss/self.args.num_agents).backward()
        

        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.role_optimizer[update_index].step()
            else:
                for i in range(self.args.num_agents):
                    self.policy.role_optimizer[i].step()
        else:
            self.policy.role_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, critic_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])   # (39, 1, 3, 1)
            # print(buffer.returns[:-1].shape, buffer.coach_returns[:-1].shape, buffer.value_preds[:-1].shape, buffer.coach_values[:-1].shape)
            coach_advantages = buffer.coach_returns[:-1] - self.value_normalizer.denormalize(buffer.coach_values[:-1])   # (39, 1, 1 ,1)
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
            coach_advantages = buffer.coach_returns[:-1] - buffer.coach_values[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        ## coach
        coach_advantages_copy = coach_advantages.copy()
        coach_mean_advantages = np.nanmean(coach_advantages_copy)
        coach_std_advantages = np.nanstd(coach_advantages_copy)
        coach_advantages = (coach_advantages - coach_mean_advantages) / (coach_std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        epoch_num = 0
        if self.args.new_period and buffer.aga_update_tag and self.args.aga_tag:
            epoch_num = self.ppo_epoch * self.args.period
        else:
            epoch_num = self.ppo_epoch
        if self.args.optim_reset and self.args.aga_tag and buffer.aga_update_tag:
            self.policy.optim_reset(buffer.update_index)
        for _ in range(epoch_num):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, coach_advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor,aga_update_tag = buffer.aga_update_tag,update_index = buffer.update_index)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        if self.args.target_dec:
            if self.args.soft_target:
                self.policy.soft_update_policy()
            else:
                if buffer.aga_update_tag and self.args.aga_tag:
                    self.policy.hard_update_policy()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        # coach
        self.policy.coach_actor.train()
        self.policy.coach_critic.train()
        if self.args.idv_para:
            for i in range(self.args.num_agents):
                self.policy.actor[i].train()
                self.policy.critic[i].train()
        else:
            self.policy.actor.train()
            self.policy.critic.train()

    def prep_rollout(self):

        # coach
        self.policy.coach_actor.eval()
        self.policy.coach_critic.eval()
        if self.args.idv_para:
            for i in range(self.args.num_agents):
                self.policy.actor[i].eval()
                self.policy.critic[i].eval()
        else:
            self.policy.actor.eval()
            self.policy.critic.eval()
