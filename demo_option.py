import time
import numpy as np

import agents
from sge.mazeenv import MazeOptionEnv
from sge.utils import KEY
###################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='random',
                        help='Agent algorithm')
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
    parser.add_argument('--render', default=False,
                        action='store_true', help='rendering observation')
    args = parser.parse_args()

    # Initialize environment & agent
    env = MazeOptionEnv(args.game_name, args.graph_param,
                        args.game_len, args.gamma, render = args.render)
    agent_class = {
        'random': agents.RandomOptionAgent,
        'greedy': agents.GreedyOptionAgent,
    }[args.algo]
    agent = agent_class(env.option_spec)
    print('Running %s agent with pre-learned options!'%(args.algo))
    
    # Reset
    init = time.time()
    state, info = env.reset()
    
    count, ret, done = 0, 0., False
    while not done:
      option = agent.sample_option(state)
      state, rew, done, info = env.step(option)
      ret += rew

      string = 'Step={:02d}, Option={}, Reward={:.2f}, Done={}'
      print(string.format(info['step_count'], env.get_option_param(option)['name'], rew, done))
      count += 1
    print('Return=', ret)
    print('fps=', count / (time.time() - init)  )
