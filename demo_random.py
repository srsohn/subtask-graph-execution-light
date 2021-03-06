import os
import sys
import random
import numpy as np

from agent import RandomAgent
from sge.mazeenv import MazeEnv
from sge.utils import KEY

###################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default='mining',
                        help='MazeEnv/config/%s.lua')
    parser.add_argument('--graph_param', default='train_1',
                        help='difficulty of subtask graph')
    parser.add_argument('--game_len', default=70,
                        type=int, help='episode length')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='discount factor')
    args = parser.parse_args()

    env = MazeEnv(args, args.game_name, args.graph_param,
                  args.game_len, args.gamma)
    agent = RandomAgent(args, env)

    state, info = env.reset()
    step, done = 0, False
    while not done:
      action = agent.act(state)
      state, rew, done, info = env.step(action)

      string = 'Step={:02d}, Action={}, Reward={:.2f}, Done={}'
      print(string.format(step, action, rew, done))
      step += 1
