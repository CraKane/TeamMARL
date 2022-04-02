import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import matplotlib

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self, test_policy, test_run):

         # create combination
        from itertools import combinations

        # pers = list(combinations(range(self.num_agents+2), self.all_args.my_policy))
        # 3-1
        if self.all_args.my_policy == 3:
            pers = [(0,1,2,3,0),(0,1,2,3,1), (0,1,2,3,2), (0,1,2,3,3)]
        else:
            # 2-2
            pers = [(0,1,2,3,0,1), (0,1,2,3,0,2), (0,1,2,3,0,3),\
                    (0,1,2,3,1,2), (0,1,2,3,1,3), (0,1,2,3,2,3),]
        np.random.shuffle(pers)

        eval_episode_rewards = []

        for per in pers:
            print(per)
            self.agent_ids = per
            self.restore()

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
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
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

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                count+=1

            eval_episode_rewards.append(episode_rewards)

        eval_episode_rewards = np.mean(eval_episode_rewards)
        print("eval average episode rewards of agent: " + str(eval_episode_rewards))

        return eval_episode_rewards