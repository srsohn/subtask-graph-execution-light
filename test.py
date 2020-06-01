import os
import sys
import random
import numpy as np

from agent import *
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
    parser.add_argument('--iter', default=1000,
                        type=int, help='episode length')
    parser.add_argument('--agent', default='random', choices=['random'],
                        help='random seed')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='discount factor')
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = MazeEnv(args, args.game_name, args.graph_param,
                  args.game_len, args.gamma)

    # agent
    if args.agent == 'random':
        agent = RandomAgent(args, env)

    ep_rews, ep_rew = [], 0
    state, info = env.reset()
    for step in range(args.iter):
        action = agent.act(state)
        state, rew, done, info = env.step(action)
        ep_rew += rew

        if done:
            print('Step={:02d}, Ep Return={:.2f}'.format(step, ep_rew))
            env.reset()
            ep_rews.append(ep_rew)
            ep_rew = 0

        #string = 'Step={:02d}, Action={}, Reward={:.2f}, Done={}'
        #print(string.format(step, action, rew, done))

    print('Avg. Ep Return={:.2f}'.format(sum(ep_rews)/len(ep_rews)))
