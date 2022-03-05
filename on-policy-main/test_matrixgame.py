import pickle
import numpy as np
from gym.spaces import Discrete

def get_element(array,index):
    ret = array
    # print('array_shape = {}'.format(np.shape(array)))
    # print('index = {}'.format(index))
    for x in index:
        ret = ret[x]
        # print('x = {}, ret_shape = {}'.format(x,np.shape(ret)))
    return ret


def make_mat_game_from_file():
    with open('random_matrix_game_3.pkl','rb') as f:
        matrix_para = pickle.load(f)
        r_mat = matrix_para['reward']  # (s,a) rewards

        trans_mat = matrix_para['trans_mat']
        print(r_mat[0])
        end_state = np.zeros(np.shape(r_mat)[0])
        print(end_state)
        state_num = np.shape(r_mat)[0]
        max_episode_length = int(state_num *1.33)
        init_state = 0
        env = MatrixGame(r_mat, trans_mat, init_state, end_state, max_episode_length,evaluate_mat=True)
        return env


class MatrixGame:


    def __init__(self,r_mat,trans_mat,init_state,end_state,max_episode_length,evaluate_mat=False):

        r_shape = np.shape(r_mat)
        self.r_mat = r_mat
        self.trans_mat = trans_mat
        self.state_num = r_shape[0]

        self.agent_num = len(r_shape) - 1 # minus state
        self.action_num = r_shape[1]
        self.share_observation_space = []
        self.observation_space = []
        self.action_space = []
        for i in range(self.agent_num):
            self.action_space.append(Discrete(self.action_num ))
            self.observation_space.append(self.state_num)
            self.share_observation_space.append(self.state_num)
        # print(self.action_space)
        self.init_state = init_state
        self.now_state  = init_state
        self.step_count = 0
        self.end_state = end_state
        self.max_episode_length = max_episode_length
        self.state_action_count = np.zeros_like(r_mat).reshape([self.state_num,-1])
        print(self.state_action_count.shape)
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
        return self.traverse, self.abs_traverse,self.relative_traverse

    def reset(self):
        self.now_state = self.init_state
        self.step_count = 0
        obs = self.get_obs()
        obs = np.expand_dims(obs, axis=0)
        share_obs = obs
        available_actions = self.get_avail_agent_actions_all()
        available_actions = np.expand_dims(available_actions, axis=0)
        return obs,share_obs,available_actions

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

    def get_ac_idx(self,action):
        idx = 0
        for a in action:
            idx = self.action_num * idx + a
            # print('idx = {} a = {}'.format(idx,a))
        return idx

    def get_state(self):
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        return state

    def step(self,action,evaluate=False):
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
            self.state_action_count[self.now_state,ac_idx] += 1
            self.long_step_count += 1

        r = get_element(self.r_mat,sa_index)
        next_s_prob = get_element(self.trans_mat,sa_index)
        # print('sa_index = {} next_s_prob = {}'.format(sa_index,next_s_prob))
        next_state = np.random.choice(range(self.state_num),size = 1, p = next_s_prob)[0]
        self.now_state = next_state
        self.step_count += 1

        done = self.end_state[self.now_state]
        if self.step_count >= self.max_episode_length:
            done = 1
        done_ret = np.ones([1,self.agent_num],dtype=bool) *done
        reward_ret = np.ones([1,self.agent_num,1]) * r
        obs = self.get_obs()
        obs = np.expand_dims(obs,axis = 0)
        share_obs = obs
        available_actions = self.get_avail_agent_actions_all()
        available_actions = np.expand_dims(available_actions,axis = 0)
        info = [[{} for i in range(self.agent_num)]]
        return obs,share_obs,reward_ret,done_ret,info,available_actions

    def get_env_info(self):
        env_info = {}
        env_info["n_actions"] = self.action_num
        env_info["n_agents"] = self.agent_num
        env_info["state_shape"] = self.state_num
        env_info["obs_shape"] = self.state_num
        env_info["episode_limit"] = self.max_episode_length
        return env_info

    def get_avail_agent_actions_all(self):
        return np.ones([self.agent_num,self.action_num])

    def get_avail_agent_actions(self,agent_id):
        return np.ones(self.action_num)

    def get_model_info(self,state,action):
        sa_index = []
        sa_index.append(state)
        action = np.array(action)
        # print('action = {}'.format(action))
        for a in action:
            sa_index.append(a)
        r = get_element(self.r_mat, sa_index)
        next_s_prob = get_element(self.trans_mat, sa_index)
        # print('action = {} sa_index = {} self.trans_mat = {} next_s_prob = {}'.format(action,sa_index,self.trans_mat.shape, next_s_prob.shape  ))
        return r,next_s_prob

    def close(self):
        return


env = make_mat_game_from_file()



from itertools import combinations
num_agents = 3

pers = list(combinations(range(num_agents+1), 3))
# print(pers)


for i in pers[0]:
        print(i)