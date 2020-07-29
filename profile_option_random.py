import os
import sys
import random
import numpy as np

from agent import RandomOptionAgent
from sge.mazeenv import MazeOptionEnv
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
    parser.add_argument('--render', default=False,
                        action='store_true', help='rendering observation')
    args = parser.parse_args()

    #
    step_time, reset_time = 0., 0.
    total_count = 0
    env = MazeOptionEnv(args.game_name, args.graph_param,
                        args.game_len, args.gamma, render = args.render)
    agent = RandomOptionAgent(env.option_spec)
    import time
    t0 = time.time()
    for ind in range(1000):
      print('ind=',ind)
      init = time.time()
      state, info = env.reset()
      reset_t = time.time()
      done = False
      count = 0
      while not done:
        option = agent.sample_option(state)
        state, rew, done, info = env.step(option)

        #step_count = info['step_count']
        #string = 'Step={:02d}, Option={}, Reward={:.2f}, Done={}'
        #print(string.format(step_count, option, rew, done))
        count += 1
        total_count += 1
      step_time += (time.time() - reset_t)
      reset_time += reset_t - init
    
    env.map.tpf.period_over()
    env.map.tpf.print()

    print('reset_time=',reset_time)
    print('step_time=',step_time)
    print('total_count=',total_count)
    print('time=',time.time() - t0)
    print('fps=', total_count / (time.time() - t0)  )
