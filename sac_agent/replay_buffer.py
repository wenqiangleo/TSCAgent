import torch
import numpy as np
from collections import deque
import random

class GraphBuffer_offline:
    def __init__(self,device="cpu"):
        self._obs = []
        self._actions = []
        self._next_obs = []
        self._rewards = []
        self._terminateds = []
        self._mc_returns = []
        self._device = device

    def load_buffer(self, buffer_data):
        """
        Load data into buffer.

        Args:
            buffer_data: Data to load into buffer.
        """
        self._obs = buffer_data[0]
        self._next_obs = buffer_data[1]
        self._actions = buffer_data[2]
        self._rewards = buffer_data[3]
        self._terminateds = buffer_data[4]
        self._mc_returns = buffer_data[5]

    def get_size(self):
        return len(self._rewards)

    def sample(self, batch_size):
        # indices = np.random.randint(0, self.get_size(), size=batch_size)
        indices = np.random.choice(self.get_size(), batch_size, replace=False)
        batch_obs = {'tsc_feat':[],'phase_feat':[],'lane_feat':[],'phase-tsc':[],'lane-phase':[]}
        batch_next_obs = {'tsc_feat':[],'phase_feat':[],'lane_feat':[],'phase-tsc':[],'lane-phase':[]}
        batch_actions = []
        batch_rewards = []
        bacth_teminateds = []
        batch_mc_returns = []
        for i in indices:
            batch_obs['tsc_feat'].append(self._obs[i]['tsc_feat'])
            batch_obs['phase_feat'].append(self._obs[i]['phase_feat'])
            batch_obs['lane_feat'].append(self._obs[i]['lane_feat'])
            batch_obs['phase-tsc'].append(self._obs[i]['phase-tsc'])
            batch_obs['lane-phase'].append(self._obs[i]['lane-phase'])
            batch_next_obs['tsc_feat'].append(self._next_obs[i]['tsc_feat'])
            batch_next_obs['phase_feat'].append(self._next_obs[i]['phase_feat'])
            batch_next_obs['lane_feat'].append(self._next_obs[i]['lane_feat'])
            batch_next_obs['phase-tsc'].append(self._next_obs[i]['phase-tsc'])
            batch_next_obs['lane-phase'].append(self._next_obs[i]['lane-phase'])
            batch_actions.append(torch.tensor([self._actions[i]], dtype=torch.int64))
            batch_rewards.append(torch.tensor([self._rewards[i]], dtype=torch.float))
            bacth_teminateds.append(torch.tensor([self._terminateds[i]], dtype=torch.float))
            batch_mc_returns.append(torch.tensor([self._mc_returns[i]], dtype=torch.float))
        '''
        batch_obs['tsc_feat'] = torch.stack(batch_obs['tsc_feat'])
        batch_obs['phase_feat'] = torch.stack(batch_obs['phase_feat'])
        batch_obs['lane_feat'] = torch.stack(batch_obs['lane_feat'])
        batch_obs['phase-tsc'] = torch.stack(batch_obs['phase-tsc'])
        batch_obs['lane-phase'] = torch.stack(batch_obs['lane-phase'])
        batch_next_obs['tsc_feat'] = torch.stack(batch_next_obs['tsc_feat'])
        batch_next_obs['phase_feat'] = torch.stack(batch_next_obs['phase_feat'])
        batch_next_obs['lane_feat'] = torch.stack(batch_next_obs['lane_feat'])
        batch_next_obs['phase-tsc'] = torch.stack(batch_next_obs['phase-tsc'])
        batch_next_obs['lane-phase'] = torch.stack(batch_next_obs['lane-phase'])        
        '''
        batch_actions = torch.stack(batch_actions)
        batch_rewards = torch.stack(batch_rewards)
        bacth_teminateds = torch.stack(bacth_teminateds)
        batch_mc_returns = torch.stack(batch_mc_returns)

        return [batch_obs, batch_next_obs, batch_actions, batch_rewards, bacth_teminateds, batch_mc_returns]
    
class GraphBuffer_online:
    def __init__(self,ts_ids, mean_rew, std_rew):
        self._obs = []
        self._actions = []
        self._next_obs = []
        self._rewards = []
        self._terminateds = []
        self._mc_returns = [0]*32
        self.dict_obs = {}
        self.dict_actions = {}
        self.dict_next_obs = {}
        self.dict_rewards = {}
        self.dict_terminateds = {}
        for id in ts_ids:
            self.dict_obs[id] = []
            self.dict_next_obs[id] = []
            self.dict_actions[id] = []
            self.dict_rewards[id] = []
            self.dict_terminateds[id] = []
        self.mean_rew = mean_rew
        self.std_rew = std_rew

    def get_size(self):
        return len(self._rewards)

    
    def reset(self):
        self._obs = []
        self._actions = []
        self._next_obs = []
        self._rewards = []
        self._terminateds = []     

    def sample(self, batch_size):
        # indices = np.random.randint(0, self.get_size(), size=batch_size)
        indices = np.random.choice(self.get_size(), batch_size, replace=False)
        batch_obs = {'tsc_feat':[],'phase_feat':[],'lane_feat':[],'phase-tsc':[],'lane-phase':[]}
        batch_next_obs = {'tsc_feat':[],'phase_feat':[],'lane_feat':[],'phase-tsc':[],'lane-phase':[]}
        batch_actions = []
        batch_rewards = []
        bacth_teminateds = []
        batch_mc_returns = []
        for i in indices:
            batch_obs['tsc_feat'].append(self._obs[i]['tsc_feat'])
            batch_obs['phase_feat'].append(self._obs[i]['phase_feat'])
            batch_obs['lane_feat'].append(self._obs[i]['lane_feat'])
            batch_obs['phase-tsc'].append(self._obs[i]['phase-tsc'])
            batch_obs['lane-phase'].append(self._obs[i]['lane-phase'])
            batch_next_obs['tsc_feat'].append(self._next_obs[i]['tsc_feat'])
            batch_next_obs['phase_feat'].append(self._next_obs[i]['phase_feat'])
            batch_next_obs['lane_feat'].append(self._next_obs[i]['lane_feat'])
            batch_next_obs['phase-tsc'].append(self._next_obs[i]['phase-tsc'])
            batch_next_obs['lane-phase'].append(self._next_obs[i]['lane-phase'])
            batch_actions.append(torch.tensor([self._actions[i]], dtype=torch.int64))
            batch_rewards.append(torch.tensor([self._rewards[i]], dtype=torch.float))
            bacth_teminateds.append(torch.tensor([self._terminateds[i]], dtype=torch.float))
            if i>=len(self._mc_returns):
                batch_mc_returns.append(torch.tensor([0.0], dtype=torch.float))
            else:
                batch_mc_returns.append(torch.tensor([self._mc_returns[i]], dtype=torch.float))
        '''
        batch_obs['tsc_feat'] = torch.stack(batch_obs['tsc_feat'])
        batch_obs['phase_feat'] = torch.stack(batch_obs['phase_feat'])
        batch_obs['lane_feat'] = torch.stack(batch_obs['lane_feat'])
        batch_obs['phase-tsc'] = torch.stack(batch_obs['phase-tsc'])
        batch_obs['lane-phase'] = torch.stack(batch_obs['lane-phase'])
        batch_next_obs['tsc_feat'] = torch.stack(batch_next_obs['tsc_feat'])
        batch_next_obs['phase_feat'] = torch.stack(batch_next_obs['phase_feat'])
        batch_next_obs['lane_feat'] = torch.stack(batch_next_obs['lane_feat'])
        batch_next_obs['phase-tsc'] = torch.stack(batch_next_obs['phase-tsc'])
        batch_next_obs['lane-phase'] = torch.stack(batch_next_obs['lane-phase'])        
        '''
        batch_actions = torch.stack(batch_actions)
        batch_rewards = torch.stack(batch_rewards)
        bacth_teminateds = torch.stack(bacth_teminateds)
        batch_mc_returns = torch.stack(batch_mc_returns)

        return [batch_obs, batch_next_obs, batch_actions, batch_rewards, bacth_teminateds, batch_mc_returns]

    def add_transition(self, old_obs, new_obs, action, rew, done):
        for id in old_obs.keys():
            self.dict_obs[id].append(old_obs[id])
            self.dict_next_obs[id].append(new_obs[id])
            self.dict_actions[id].append(action[id])
            # nomarlize reward!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # rew[id] = (rew[id]-self.mean_rew)/self.std_rew
            self.dict_rewards[id].append(rew[id])  # ((rew[id]-self.mean_rew)/self.std_rew) 
            self.dict_terminateds[id].append(done[id])
        self.reset()
        for id in old_obs.keys():
            self._obs+=self.dict_obs[id]
            self._next_obs+=self.dict_next_obs[id]
            self._actions+=self.dict_actions[id]
            self._rewards+=self.dict_rewards[id]
            self._terminateds+=self.dict_terminateds[id]
        '''
        if self._terminateds[-1]:
            self.cal_return_to_go()        
        '''

    def cal_return_to_go(self):
        self._mc_returns = []
        ep_len = 0
        cur_rewards = []
        terminals = []
        for i,(r,d) in enumerate(zip(self._rewards, self._terminateds)):
            cur_rewards.append(float(r))
            terminals.append(float(d))
            ep_len += 1
            if d:
                discounted_returns = [0] * ep_len
                prev_return = 0
                for e in reversed(range(ep_len)):
                    discounted_returns[e] = cur_rewards[e] + 0.99 * prev_return * (1 - terminals[e])
                    prev_return = discounted_returns[e]
                self._mc_returns += discounted_returns
                ep_len = 0
                cur_rewards = []
                terminals = []

class GraphBuffer:
    def __init__(self,device="cpu"):
        self._obs = []
        self._actions = []
        self._next_obs = []
        self._rewards = []
        self._terminateds = []
        self._device = device
        self.sample_index = 0

    def load_buffer(self, buffer_data):
        """
        Load data into buffer.

        Args:
            buffer_data: Data to load into buffer.
        """
        self._obs = buffer_data[0]
        self._next_obs = buffer_data[1]
        self._actions = buffer_data[2]
        self._rewards = buffer_data[3]
        self._terminateds = buffer_data[4]

    def generate_sample_index(self, batch_size):
        self.sample_index = []
        samples = list(range(len(self._actions)))
        random.shuffle(samples)
        self.sample_index+=samples
        complete_inedx_num = len(self.sample_index)%batch_size
        if complete_inedx_num==0:
            pass
        else:
            complete_samples = random.sample(self.sample_index, batch_size-complete_inedx_num)
            self.sample_index += complete_samples

        self.sample_index = np.array(self.sample_index).reshape(-1, batch_size).tolist()

    def get_size(self):
        return len(self.sample_index)

    def obs2list(self, obs):
        node_pt = obs['phase-tsc']
        node_lp = obs['lane-phase']
        tsc_feat = obs['tsc_feat']
        phase_feat = obs['phase_feat']
        lane_feat = obs['lane_feat']
        return node_pt, node_lp, tsc_feat, phase_feat, lane_feat

    def sample(self, batch_size, global_step):
        batch_obs = {'tsc_feat':[],'phase_feat':[],'lane_feat':[],'phase-tsc':[],'lane-phase':[]}
        batch_next_obs = {'tsc_feat':[],'phase_feat':[],'lane_feat':[],'phase-tsc':[],'lane-phase':[]}
        batch_actions = []
        batch_rewards = []
        bacth_teminateds = []
        for i in self.sample_index[global_step]:
            batch_obs['tsc_feat'].append(self._obs[i]['tsc_feat'])
            batch_obs['phase_feat'].append(self._obs[i]['phase_feat'])
            batch_obs['lane_feat'].append(self._obs[i]['lane_feat'])
            batch_obs['phase-tsc'].append(self._obs[i]['phase-tsc'])
            batch_obs['lane-phase'].append(self._obs[i]['lane-phase'])
            batch_next_obs['tsc_feat'].append(self._next_obs[i]['tsc_feat'])
            batch_next_obs['phase_feat'].append(self._next_obs[i]['phase_feat'])
            batch_next_obs['lane_feat'].append(self._next_obs[i]['lane_feat'])
            batch_next_obs['phase-tsc'].append(self._next_obs[i]['phase-tsc'])
            batch_next_obs['lane-phase'].append(self._next_obs[i]['lane-phase'])
            batch_actions.append(torch.tensor([self._actions[i]], dtype=torch.int64))
            batch_rewards.append(torch.tensor([self._rewards[i]], dtype=torch.float))
            bacth_teminateds.append(torch.tensor([self._terminateds[i]], dtype=torch.float))
        '''
        batch_obs['tsc_feat'] = torch.stack(batch_obs['tsc_feat'])
        batch_obs['phase_feat'] = torch.stack(batch_obs['phase_feat'])
        batch_obs['lane_feat'] = torch.stack(batch_obs['lane_feat'])
        batch_obs['phase-tsc'] = torch.stack(batch_obs['phase-tsc'])
        batch_obs['lane-phase'] = torch.stack(batch_obs['lane-phase'])
        batch_next_obs['tsc_feat'] = torch.stack(batch_next_obs['tsc_feat'])
        batch_next_obs['phase_feat'] = torch.stack(batch_next_obs['phase_feat'])
        batch_next_obs['lane_feat'] = torch.stack(batch_next_obs['lane_feat'])
        batch_next_obs['phase-tsc'] = torch.stack(batch_next_obs['phase-tsc'])
        batch_next_obs['lane-phase'] = torch.stack(batch_next_obs['lane-phase'])        
        '''
        batch_actions = torch.stack(batch_actions)
        batch_rewards = torch.stack(batch_rewards)
        bacth_teminateds = torch.stack(bacth_teminateds)

        return batch_obs, batch_next_obs, batch_actions, batch_rewards, bacth_teminateds
    

class GraphBuffer_SAC:
    def __init__(self,ts_ids):
        self._obs = []
        self._actions = []
        self._next_obs = []
        self._rewards = []
        self._terminateds = []
        self._mc_returns = [0]*32
        self.dict_obs = {}
        self.dict_actions = {}
        self.dict_next_obs = {}
        self.dict_rewards = {}
        self.dict_terminateds = {}
        for id in ts_ids:
            self.dict_obs[id] = []
            self.dict_next_obs[id] = []
            self.dict_actions[id] = []
            self.dict_rewards[id] = []
            self.dict_terminateds[id] = []

    def get_size(self):
        return len(self._rewards)

    
    def reset(self):
        self._obs = []
        self._actions = []
        self._next_obs = []
        self._rewards = []
        self._terminateds = []     

    def sample(self, batch_size):
        # indices = np.random.randint(0, self.get_size(), size=batch_size)
        indices = np.random.choice(self.get_size(), batch_size, replace=False)
        batch_obs = {'tsc_feat':[],'phase_feat':[],'lane_feat':[],'phase-tsc':[],'lane-phase':[]}
        batch_next_obs = {'tsc_feat':[],'phase_feat':[],'lane_feat':[],'phase-tsc':[],'lane-phase':[]}
        batch_actions = []
        batch_rewards = []
        bacth_teminateds = []
        batch_mc_returns = []
        for i in indices:
            batch_obs['tsc_feat'].append(self._obs[i]['tsc_feat'])
            batch_obs['phase_feat'].append(self._obs[i]['phase_feat'])
            batch_obs['lane_feat'].append(self._obs[i]['lane_feat'])
            batch_obs['phase-tsc'].append(self._obs[i]['phase-tsc'])
            batch_obs['lane-phase'].append(self._obs[i]['lane-phase'])
            batch_next_obs['tsc_feat'].append(self._next_obs[i]['tsc_feat'])
            batch_next_obs['phase_feat'].append(self._next_obs[i]['phase_feat'])
            batch_next_obs['lane_feat'].append(self._next_obs[i]['lane_feat'])
            batch_next_obs['phase-tsc'].append(self._next_obs[i]['phase-tsc'])
            batch_next_obs['lane-phase'].append(self._next_obs[i]['lane-phase'])
            batch_actions.append(torch.tensor([self._actions[i]], dtype=torch.int64))
            batch_rewards.append(torch.tensor([self._rewards[i]], dtype=torch.float))
            bacth_teminateds.append(torch.tensor([self._terminateds[i]], dtype=torch.float))
            if i>=len(self._mc_returns):
                batch_mc_returns.append(torch.tensor([0.0], dtype=torch.float))
            else:
                batch_mc_returns.append(torch.tensor([self._mc_returns[i]], dtype=torch.float))
        '''
        batch_obs['tsc_feat'] = torch.stack(batch_obs['tsc_feat'])
        batch_obs['phase_feat'] = torch.stack(batch_obs['phase_feat'])
        batch_obs['lane_feat'] = torch.stack(batch_obs['lane_feat'])
        batch_obs['phase-tsc'] = torch.stack(batch_obs['phase-tsc'])
        batch_obs['lane-phase'] = torch.stack(batch_obs['lane-phase'])
        batch_next_obs['tsc_feat'] = torch.stack(batch_next_obs['tsc_feat'])
        batch_next_obs['phase_feat'] = torch.stack(batch_next_obs['phase_feat'])
        batch_next_obs['lane_feat'] = torch.stack(batch_next_obs['lane_feat'])
        batch_next_obs['phase-tsc'] = torch.stack(batch_next_obs['phase-tsc'])
        batch_next_obs['lane-phase'] = torch.stack(batch_next_obs['lane-phase'])        
        '''
        batch_actions = torch.stack(batch_actions)
        batch_rewards = torch.stack(batch_rewards)
        bacth_teminateds = torch.stack(bacth_teminateds)
        batch_mc_returns = torch.stack(batch_mc_returns)

        return [batch_obs, batch_next_obs, batch_actions, batch_rewards, bacth_teminateds, batch_mc_returns]

    def add_transition(self, old_obs, new_obs, action, rew, done):
        for id in old_obs.keys():
            self.dict_obs[id].append(old_obs[id])
            self.dict_next_obs[id].append(new_obs[id])
            self.dict_actions[id].append(action[id])
            # nomarlize reward!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # rew[id] = (rew[id]-self.mean_rew)/self.std_rew
            self.dict_rewards[id].append(rew[id])  # ((rew[id]-self.mean_rew)/self.std_rew) 
            self.dict_terminateds[id].append(done[id])
        self.reset()
        for id in old_obs.keys():
            self._obs+=self.dict_obs[id]
            self._next_obs+=self.dict_next_obs[id]
            self._actions+=self.dict_actions[id]
            self._rewards+=self.dict_rewards[id]
            self._terminateds+=self.dict_terminateds[id]
        '''
        if self._terminateds[-1]:
            self.cal_return_to_go()        
        '''

    def cal_return_to_go(self):
        self._mc_returns = []
        ep_len = 0
        cur_rewards = []
        terminals = []
        for i,(r,d) in enumerate(zip(self._rewards, self._terminateds)):
            cur_rewards.append(float(r))
            terminals.append(float(d))
            ep_len += 1
            if d:
                discounted_returns = [0] * ep_len
                prev_return = 0
                for e in reversed(range(ep_len)):
                    discounted_returns[e] = cur_rewards[e] + 0.99 * prev_return * (1 - terminals[e])
                    prev_return = discounted_returns[e]
                self._mc_returns += discounted_returns
                ep_len = 0
                cur_rewards = []
                terminals = []