import time
import numpy as np

import agents
from sge.mazeenv import MazeOptionEnv, FewshotWrapper
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

    # Few-shot param
    EPI_BUDGET = 8
    NUM_TASKS = 500

    # Initialize environment & agent
    env = MazeOptionEnv(args.game_name, args.graph_param,
                        args.game_len, args.gamma, render = args.render)
    env = FewshotWrapper(env)
    agent_class = {
        'random': agents.RandomOptionAgent,
        'greedy': agents.GreedyOptionAgent,
    }[args.algo]
    agent = agent_class(env.option_spec)
    print('Running %s agent with pre-learned options in %d-shot RL setting!'%(args.algo, EPI_BUDGET))
    
    # Reset
    init = time.time()
    option_count, ret_sum = 0, 0.
    for task_ind in range(NUM_TASKS):
      print('===============   Task # %d/%d    ==============='%(task_ind, NUM_TASKS))
      state, info = env.reset(max_epi=EPI_BUDGET, reset_task=True, graph_index=task_ind)
      ret, trial_done = 0., False
      while not trial_done:
        option = agent.sample_option(state)
        state, rew, trial_done, info = env.step(option)
        ret += rew
        option_count += 1
      ret_sum += ret/EPI_BUDGET
      print('Mean Return=', ret_sum / (task_ind+1))
    print('fps=', option_count / (time.time() - init)  )
