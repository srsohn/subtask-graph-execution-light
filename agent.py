import random

class RandomAgent:
    def __init__(self, args, env):
        self.args = args
        self.action_set = env.get_actions()

    def act(self, state):
        return random.sample(list(self.action_set), 1)[0]

    def __repr__(self):
        return str(self)