import os
import sys
from .graph import SubtaskGraph
from sge.mazemap import Mazemap
import numpy as np
from .utils import get_id_from_ind_multihot, TimeProfiler
from sge.utils import WHITE, BLACK, DARK, LIGHT, GREEN, DARK_RED
import time


class MazeEnv(object):  # single batch
    def __init__(self, game_name, graph_param, game_len, gamma, render=False):
        if game_name == 'playground':
            from sge.playground import Playground
            game_config = Playground()
            graph_folder = os.path.join('.', 'data', 'subtask_graph_play')
            filename = 'play_{param}'.format(param=graph_param)

        elif game_name == 'mining':
            from sge.mining import Mining
            game_config = Mining()
            graph_folder = os.path.join('.', 'data', 'subtask_graph_mining')
            filename = 'mining_{param}'.format(param=graph_param)

        self.config = game_config
        self.max_task = self.config.nb_subtask_type
        self.subtask_list = self.config.subtask_list
        self._default_game_len = game_len

        # graph & map
        self.graph = SubtaskGraph(
            graph_folder, filename, self.max_task)  # just load all graph
        self.map = Mazemap(game_name, game_config, render=render)
        self.gamma = gamma
        self.count = 0

        # init
        self.step_reward = 0.0
        self.tpf = TimeProfiler('Environment/')

    def is_done(self):
        time_over = self.step_count >= self.game_length
        game_over = (self.eligibility * self.mask).sum().item() == 0
        return time_over or game_over

    def step(self, action):
        if self.graph.graph_index is None:
            raise RuntimeError('Environment has never been reset()')
        sub_id = -1
        if self.is_done():
            raise ValueError(
                'Environment has already been terminated. need to be reset!')
        oid = self.map.act(action)
        if (action, oid) in self.config.subtask_param_to_id:  # if (action, item) is one of the subtasks
            sid = self.config.subtask_param_to_id[(action, oid)]
            if sid in self.subtask_id_list:  # if sub_id is in the subtask graph
                sub_id = sid
            else:
                #print('Warning! Executed a non-existing subtask')
                pass
        #
        reward = self._act_subtask(sub_id)
        self.step_count += 1
        self.ret += reward * (self.gamma ** self.step_count)

        return self._get_state(), reward, self.is_done(), self._get_info(reward, 1)

    def reset(self, reset_task=True, task_index=None):  # after every episodeipdb
        ___tpf = self.tpf
        # 1. reset graph (2.5%)
        ___tpf.stamp(time.time())
        if reset_task:
            if task_index is None:
                task_index = np.random.permutation(self.graph.num_graph)[0]
            else:
                task_index = task_index % self.graph.num_graph
            self.graph.set_graph_index(task_index)
            self.nb_subtask = len(self.graph.subtask_id_list)
            self.rew_mag = self.graph.rew_mag
            self.subtask_id_list = self.graph.subtask_id_list
            self.game_length = int(np.random.uniform(0.8, 1.2) * self._default_game_len)
        ___tpf.stamp(time.time(), 'reset graph')

        # 2. reset subtask status (1.5%)
        self.executed_sub_ind = -1
        self.mask = np.ones(self.nb_subtask, dtype=np.uint8)
        self.mask_id = np.zeros(self.max_task, dtype=np.uint8)
        for ind, sub_id in self.graph.ind_to_id.items():
            self.mask_id[sub_id] = 1
        self.completion = np.zeros(self.nb_subtask, dtype=np.int8)
        self.comp_id = np.zeros(self.max_task, dtype=np.int8)
        self._compute_elig()
        self.step_count = 0
        ___tpf.stamp(time.time(), 'reset subtask status')

        # 3. reset map (96% of time)
        self.map.reset(subtask_id_list=self.subtask_id_list, reset_map=reset_task)
        ___tpf.stamp(time.time(), 'reset map')

        self.map.render(task=self.graph.graph_index, step=self.step_count, epi=self.count)
        self.count += 1
        
        # 4. initialize episodic statistics
        self.ret = 0.
        return self._get_state(), self._get_info()

    @property
    def state_spec(self):
        return [
            {'dtype': self.map.get_obs().dtype, 'name': 'observation',
             'shape': self.map.get_obs().shape},
            {'dtype': self.mask_id.dtype, 'name': 'mask',
                'shape': self.mask_id.shape},
            {'dtype': self.comp_id.dtype, 'name': 'completion',
                'shape': self.comp_id.shape},
            {'dtype': self.elig_id.dtype, 'name': 'eligibility',
                'shape': self.elig_id.shape},
            {'dtype': int, 'name': 'step', 'shape': ()}
        ]

    @property
    def action_spec(self):
        return self.config.legal_actions

    # internal
    def _get_state(self):
        step = self.game_length - self.step_count
        return {
            'observation': self.map.get_obs(),
            'mask_id': self.mask_id.astype(np.float),
            'comp_id': self.comp_id.astype(np.float),
            'elig_id': self.elig_id.astype(np.float),
            'mask_ind': self.mask.astype(np.float),
            'comp_ind': self.completion.astype(np.float),
            'elig_ind': self.eligibility.astype(np.float),
            'step': step
        }

    def _get_info(self, reward=None, steps=0):
        return {
            'graph': self.graph,
            # For SMDP
            'step_count': steps,
            'discount': self.gamma**steps,
            'raw_reward': reward,
            # episodic
            'return': self.ret,
        }

    def _act_subtask(self, sub_id):
        self.executed_sub_ind = -1
        reward = self.step_reward
        if sub_id < 0:
            return reward
        sub_ind = self.graph.id_to_ind[sub_id]
        if self.eligibility[sub_ind] == 1 and self.mask[sub_ind] == 1:
            self.completion[sub_ind] = 1
            self.comp_id[sub_id] = 1
            reward += self.rew_mag[sub_ind]
            self.executed_sub_ind = sub_ind
        self.mask[sub_ind] = 0
        self.mask_id[sub_id] = 0

        self._compute_elig()

        return reward

    def _compute_elig(self):
        self.eligibility = self.graph.get_elig(self.completion)
        self.elig_id = get_id_from_ind_multihot(
            self.eligibility, self.graph.ind_to_id, self.max_task)

class MazeOptionEnv(MazeEnv):  # single batch. option. single episode
    def __init__(self, game_name, graph_param, game_len, gamma, render=False):
        super().__init__(game_name, graph_param, game_len, gamma, render)

    def step(self, action):
        assert self.graph.graph_index is not None, 'Environment need to be reset'
        assert not self.is_done(), 'Environment has already been terminated.'
        assert action in self.subtask_id_list, 'Option %s does not exist!'%(action)
        sub_id = action

        # teleport to target obj pos
        interact, oid = self.config.subtask_param_list[sub_id]
        nav_step = self.map.move_to_closest_obj(oid=oid, max_step=self.game_length - self.step_count - 1)
        option_step = nav_step + 1
        
        if nav_step >= 0: # enough time.
            assert self.step_count + option_step <= self.game_length
            # interact with target obj in the map
            oid_ = self.map.act(interact)
            assert oid == oid_

            # process subtask
            reward = self._act_subtask(sub_id) # update comp, elig, mask.
            option_return = reward * (self.gamma ** option_step)
            self.step_count += option_step
            self.ret += reward * (self.gamma ** self.step_count)
        else: # not enough time. time over
            self.step_count = self.game_length
            reward = 0.
            option_return = 0.
        #self.map.render(self.step_count)
            
        return self._get_state(), option_return, self.is_done(), self._get_info(reward, option_step)

    @property
    def state_spec(self):
        return [
            {'dtype': self.map.get_obs().dtype, 'name': 'observation',
             'shape': self.map.get_obs().shape},
            {'dtype': self.mask_id.dtype, 'name': 'mask',
                'shape': self.mask_id.shape},
            {'dtype': self.comp_id.dtype, 'name': 'completion',
                'shape': self.comp_id.shape},
            {'dtype': self.elig_id.dtype, 'name': 'eligibility',
                'shape': self.elig_id.shape},
            {'dtype': float, 'name': 'remaining_step', 'shape': ()}
        ]

    @property
    def num_tasks(self):
        return self.graph.num_graph
        
    @property
    def option_spec(self):
        return list(range(self.config.nb_subtask_type))

    def get_option_param(self, tid):
        return self.config.subtask_list[tid]

class FewshotWrapper(MazeEnv):
    def __init__(self, env, max_epi=1):
        self.env = env
        self.max_epi = max_epi
        self.epi_count = 0

    def reset(self, reset_task=True, task_index=None):
        state, info = self.env.reset(reset_task=reset_task, task_index=task_index)
        if reset_task:
            self.epi_count = 0
        state['remaining_epi'] = np.log(self.max_epi - self.epi_count + 1)
        info['epi_count'] = self.epi_count
        self._keep(state, info)
        return state, info
    
    def _keep(self, state, info):
        self.prev_state = state
        self.prev_info = info

    @property
    def state_spec(self):
        return self.env.state_spec + [
            {'dtype': float, 'name': 'remaining_epi', 'shape': ()}
        ]

    def step(self, action):
        if self.trial_done:
          return self.prev_state, 0., self.trial_done, self.prev_info
        else:
          state, reward, done, info = self.env.step(action)
          if done:
              self.epi_count += 1
              if not self.trial_done: # trial is not over
                  self.last_state = state
                  self.last_info = info
                  state, info = self.env.reset(reset_task=False) # only reset episode
          state['remaining_epi'] = np.log(self.max_epi - self.epi_count + 1)
          info['epi_count'] = self.epi_count
        self._keep(state, info)
        return state, reward, self.trial_done, info
    
    def get_last_state_info(self):
        return self.last_state, self.last_info
    
    @property
    def option_spec(self):
        return self.env.option_spec
    
    def get_option_param(self, tid):
        return self.env.get_option_param(tid)
    
    @property
    def action_spec(self):
        return self.env.action_spec
    
    @property
    def trial_done(self):
        return self.epi_count == self.max_epi
    
class VecEnv:
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self, reset_task=True, task_index=None):
        states, infos = [], []
        for env in self.envs:
          state, info = env.reset(reset_task, task_index)
          states.append(state)
          infos.append(info)
        return states, infos

    def step(self, actions):
        states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        """
        # reset
        for bind in range(self.num_envs):
            if dones[bind]:
                state, info = self.envs[bind].reset()
                states[bind] = state
                infos[bind] = info"""
        return states, rewards, dones, infos
    
    def get_last_state_info(self):
        states, infos = [], []
        for env in self.envs:
          state, info = env.get_last_state_info()
          states.append(state)
          infos.append(info)
        return states, infos
    
    @property
    def state_spec(self):
        return self.envs[0].state_spec

    @property
    def option_spec(self):
        return self.envs[0].option_spec
    
    @property
    def num_tasks(self):
        return self.envs[0].num_tasks

    def get_option_param(self, tid):
        return self.envs[0].get_option_param(tid)

        