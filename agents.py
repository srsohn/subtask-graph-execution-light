import random
import numpy as np
from sge.utils import get_id_from_ind_multihot

class RandomAgent:
    def __init__(self, option_space, batch_size):
        self.batch_size = batch_size
        self.action_spec = action_spec

    def act(self, state):
        return random.sample(self.action_spec, 1)[0]

    def __repr__(self):
        return str(self)

class RandomOptionAgent:
    def __init__(self, option_space, batch_size):
        self.batch_size = batch_size
        self.option_space = option_space

    def act(self, state):
      raise NotImplementedError

    def sample_option(self, states):
      options = []
      for state in states:
        policy_mask = state['mask_id'] * state['elig_id']
        eligible_options = policy_mask.nonzero()[0].tolist()
        option = random.sample(eligible_options, 1)[0]
        options.append(option)
      return np.stack(options)

    def update_graph(self, graph):
      pass

    def __repr__(self):
        return str(self)

class GreedyOptionAgent(RandomOptionAgent):
    def __init__(self, option_space, batch_size):
        self.batch_size = batch_size
        self.option_space = option_space
        self.option_score = np.ones( (len(option_space)) )

    def sample_option(self, states):
      options = []
      for state in states:
          policy_mask = state['mask_id'] * state['elig_id']
          masked_score = policy_mask * self.option_score
          option = np.argmax(masked_score)
          options.append(option)
      return np.stack(options)

    def update_graph(self, graph):
      rew_mag_ind = np.array(graph.rew_mag)
      self.option_score = get_id_from_ind_multihot(rew_mag_ind, graph.ind_to_id, graph.max_task)
      self.option_score -= self.option_score.min()