import ipdb
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import time

from odium.utils.env_utils.make_env import make_env
from odium.utils.controller_utils.get_controller import get_controller

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str,
                    default='FetchPush-v1', help='Name of environment')
parser.add_argument('--seed', type=int, default=1, help='Seed of environment')
parser.add_argument('--no-render', action='store_true')
parser.add_argument('--num-expansions', type=int, default=3,
                    help='Number of expansions by astar')
parser.add_argument('--record', action='store_true')
parser.add_argument('--total-num-episodes', type=int, default=10)
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--env-id', type=int, default=None,
                    help='Env id for the FetchPushAmongObstacles env')
parser.add_argument('--deterministic', action='store_true')
args = parser.parse_args()

env = make_env(args.env_name, env_id=args.env_id, discrete=args.discrete)
env.seed(args.seed)

if args.deterministic:
    env.make_deterministic()

controller = get_controller(
    args.env_name, num_expansions=args.num_expansions, env_id=args.env_id, discrete=args.discrete)


obs = env.reset()
t = 0
num_successes = 0.
num_episodes = 0.

f_vals_best = []
f_vals_current_episode_best = []
f_vals_start = []
f_vals_current_episode_start = []

while num_episodes < args.total_num_episodes:
    ac, info = controller.act(obs)
    print(ac, info)
    if info['best_node_f'] > info['start_node_f']:
        print('Predicting stuck')
    f_vals_current_episode_best.append(info['best_node_f'])
    f_vals_current_episode_start.append(info['start_node_f'])
    obs, rew, _, _ = env.step(ac)
    t += 1
    if not args.no_render:
        env.render()

    if t == env._max_episode_steps:
        f_vals_best.append(f_vals_current_episode_best)
        f_vals_start.append(f_vals_current_episode_start)
        f_vals_current_episode_best = []
        f_vals_current_episode_start = []
        num_episodes += 1
        if rew == 0:
            num_successes += 1
            print('Reached goal')
        print('End')
        t = 0
        print('Success rate', num_successes / num_episodes)
        obs = env.reset()

if args.record:
    f_vals_best = np.array(f_vals_best)
    f_vals_start = np.array(f_vals_start)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, 10))
    for i in range(args.total_num_episodes):
        plt.plot(np.arange(env._max_episode_steps),
                 f_vals_best[i, :], color=colors[i], linestyle='-')
        plt.plot(np.arange(env._max_episode_steps),
                 f_vals_start[i, :], color=colors[i], linestyle=':')
    plt.ylabel('f-values')
    plt.xlabel('Timestep')
    custom_lines = [Line2D([0], [0], linestyle=':'),
                    Line2D([0], [0], linestyle='-')]
    plt.legend(custom_lines, ['Start node fval', 'Best node fval'])
    plt.title('Expansion of weighted A* on '+args.env_name)
    plt.show()
