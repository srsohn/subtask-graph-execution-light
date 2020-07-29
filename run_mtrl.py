import time
import numpy as np

import agents
import grprop
from sge.mazeenv import MazeOptionEnv, FewshotWrapper, VecEnv
from sge.utils import KEY

###################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='random',
                        help='Agent algorithm')
    parser.add_argument('--num_envs', default=4,
                        help='Number of environments')
    parser.add_argument('--game_name', default='mining',
                        help='MazeEnv/config/%s.lua')
    parser.add_argument('--graph_param', default='eval_1',
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

    NUM_EPISODES = 4

    # Initialize environment & agent
    env = VecEnv([MazeOptionEnv(args.game_name, args.graph_param,
                        args.game_len, args.gamma, render=args.render) for _ in range(args.num_envs)])
    num_graphs = env.num_tasks
    agent_class = {
        'random': agents.RandomOptionAgent,
        'greedy': agents.GreedyOptionAgent,
        'grprop': grprop.GRProp,
    }[args.algo]
    agent = agent_class(env.option_spec, batch_size=env.num_envs)
    print('Running %s agent with pre-learned options!'%(args.algo))
    
    # Reset
    init = time.time()
    ret_sum, score_sum = 0., 0.
    ep_count, step_count = 0, 0
    for task_ind in range(num_graphs):
      env.reset_task(task_index=task_ind)
      print('===============   Task # %d/%d    ==============='%(task_ind, num_graphs))
      for ep_ind in range(NUM_EPISODES):
        state, info = env.reset()
        score, done = 0., False
        while not np.all(done):
          agent.update_graph([i['graph'] for i in info])
          option = agent.sample_option(state)
          state, rew, done, info = env.step(option)
          score += np.stack([i['raw_reward'] for i in info])

          #string = 'Step={:02d}, Option={}, Reward={:.2f}, Done={}'
          #print(string.format(info['step_count'], env.get_option_param(option)['name'], info['raw_reward'], done))
          step_count += 1
          print('np.all(done)=',np.all(done))
        
        ret_sum += np.stack([i['return'] for i in info])
        score_sum += score
        ep_count += 1
      print('score_sum=',score_sum)
      print('Score=', score_sum.mean() / ep_count)
      print('Return=', ret_sum.mean() / ep_count)
      print('fps=', step_count / (time.time() - init)  )
      import ipdb; ipdb.set_trace()