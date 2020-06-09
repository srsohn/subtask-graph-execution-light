import os
import sys
import random
import numpy as np

from agent import RandomAgent
from sge.mazeenv import MazeEnv

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

    NUM_GRAPH = 100
    NUM_ITER = 32
    ep_rews = []
    for graph_id in range(NUM_GRAPH):
        for _ in range(NUM_ITER):
            ep_rew = 0
            state, info = env.reset(graph_index=graph_id)
            done = False
            while not done:
                action = agent.act(state)
                state, rew, done, info = env.step(action)
                ep_rew += rew
            ep_rews.append(ep_rew)

        string = 'Graph={:02d}/{:02d}, Return={:.4f}'
        print(string.format(graph_id, NUM_GRAPH, sum(ep_rews)/len(ep_rews)))

    print('Avg. Ep Return={:.4f}'.format(sum(ep_rews)/len(ep_rews)))
    print('This should be around 0.0455')
