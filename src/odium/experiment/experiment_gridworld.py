import ray
import numpy as np
import pickle

from odium.agents.gridworld_agents.arguments import get_args
from odium.experiment.train_gridworld import launch

import matplotlib.pyplot as plt

# Configure the experiment
np.random.seed(0)
incorrectness = [0, 0.2, 0.4, 0.6, 0.8]
seeds = np.random.randint(100000, size=50)
grid_size = 100
n_expansions = 5
epsilon = [0.1, 0.3, 0.5]

obstacle_results = {}

ray.init()

# Random obstacle experiments
args = get_args()
args.env = 'random_obstacle'
args.planning_env = 'empty'
args.grid_size = grid_size
args.max_timesteps = 100000

# RTS
args.agent = 'rts'
obstacle_results['rts'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    obstacle_results['rts'][incorrect] = result

# LRTA*
args.agent = 'lrtastar'
obstacle_results['lrtastar'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    obstacle_results['lrtastar'][incorrect] = result

# Q-learning
args.agent = 'qlearning'
obstacle_results['qlearning'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    best_result = []
    best_epsilon = None
    best_mean_result = np.inf
    for eps in epsilon:
        result = []
        args.epsilon = eps
        for seed in seeds:
            args.seed = seed
            n_steps = launch.remote(args)
            result.append(n_steps)
        result = ray.get(result)
        mean_result = np.mean(result)
        if best_mean_result > mean_result:
            best_mean_result = mean_result
            best_result = result
            best_epsilon = eps
    obstacle_results['qlearning'][incorrect] = best_result
    obstacle_results['qlearning']['epsilon'] = best_epsilon

# Eps-greedy
args.agent = 'eps'
obstacle_results['eps'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    best_result = []
    best_mean_result = np.inf
    best_epsilon = None
    for eps in epsilon:
        result = []
        args.epsilon = eps
        for seed in seeds:
            args.seed = seed
            n_steps = launch.remote(args)
            result.append(n_steps)
        result = ray.get(result)
        mean_result = np.mean(result)
        if best_mean_result > mean_result:
            best_mean_result = mean_result
            best_result = result
            best_epsilon = eps
    obstacle_results['eps'][incorrect] = best_result
    obstacle_results['eps']['epsilon'] = best_epsilon


slip_results = {}

# Random slip state experiments
args = get_args()
args.env = 'random_slip'
args.planning_env = 'empty'
args.grid_size = grid_size
args.max_timesteps = 100000

# RTS
args.agent = 'rts'
slip_results['rts'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    slip_results['rts'][incorrect] = result

# LRTA*
args.agent = 'lrtastar'
slip_results['lrtastar'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    slip_results['lrtastar'][incorrect] = result

# Q-learning
args.agent = 'qlearning'
slip_results['qlearning'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    best_result = []
    best_mean_result = np.inf
    best_epsilon = None
    for eps in epsilon:
        result = []
        args.epsilon = eps
        for seed in seeds:
            args.seed = seed
            n_steps = launch.remote(args)
            result.append(n_steps)
        result = ray.get(result)
        mean_result = np.mean(result)
        if best_mean_result > mean_result:
            best_mean_result = mean_result
            best_result = result
            best_epsilon = eps
    slip_results['qlearning'][incorrect] = best_result
    slip_results['qlearning']['epsilon'] = best_epsilon

# Eps-greedy
args.agent = 'eps'
slip_results['eps'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    best_result = []
    best_mean_result = np.inf
    best_epsilon = None
    for eps in epsilon:
        result = []
        args.epsilon = eps
        for seed in seeds:
            args.seed = seed
            n_steps = launch.remote(args)
            result.append(n_steps)
        result = ray.get(result)
        mean_result = np.mean(result)
        if best_mean_result > mean_result:
            best_mean_result = mean_result
            best_result = result
            best_epsilon = eps
    slip_results['eps'][incorrect] = best_result
    slip_results['eps']['epsilon'] = best_epsilon


pickle.dump([obstacle_results, slip_results], open(
    '/home/avemula/workspaces/odium_ws/src/odium/save/gridworld_experiments_results.pkl', 'wb'))

# obstacle_results, slip_results = pickle.load(open(
#     '/home/avemula/workspaces/odium_ws/src/odium/save/gridworld_experiments_results.pkl', 'rb'))


# def get_mean_std(results):
#     mean_results = []
#     std_results = []
#     incomplete = []
#     epsilon = []
#     for incorrect in incorrectness:
#         if incorrect == 'epsilon':
#             epsilon.append(results['epsilon'])
#             continue
#         data = np.array(results[incorrect])
#         solved_data = data[data != 100000]
#         unsolved_data = data[data == 100000]
#         mean_results.append(np.mean(solved_data))
#         std_results.append(np.std(solved_data) / np.sqrt(solved_data.shape[0]))
#         incomplete.append(unsolved_data.shape[0])

#     return np.array(mean_results), np.array(std_results), np.array(incomplete), np.array(epsilon)


# def plot_mean_std(xaxis, ymean, ystd, color, label):
#     plt.plot(xaxis, ymean, color=color, linestyle='-', label=label)
#     plt.fill_between(xaxis, ymean - ystd, ymean + ystd, color=color, alpha=0.2)


# # Obstacle experiments
# rts_mean_results, rts_std_results, rts_unsolved, _ = get_mean_std(
#     obstacle_results['rts'])
# lrtastar_mean_results, lrtastar_std_results, lrtastar_unsolved, _ = get_mean_std(
#     obstacle_results['lrtastar'])
# qlearning_mean_results, qlearning_std_results, qlearning_unsolved, qlearning_epsilon = get_mean_std(
#     obstacle_results['qlearning'])
# eps_mean_results, eps_std_results, eps_unsolved, eps_epsilon = get_mean_std(
#     obstacle_results['eps'])

# np.set_printoptions(precision=1)
# print('OBSTACLE')
# print('===============================================')
# print(rts_mean_results, rts_std_results, rts_unsolved)
# print(lrtastar_mean_results, lrtastar_std_results, lrtastar_unsolved)
# print(qlearning_mean_results, qlearning_std_results, qlearning_unsolved, qlearning_epsilon)
# print(eps_mean_results, eps_std_results, eps_unsolved, eps_epsilon)
# print('===============================================')

# # Slip experiments
# rts_mean_results, rts_std_results, rts_unsolved, _ = get_mean_std(
#     slip_results['rts'])
# lrtastar_mean_results, lrtastar_std_results, lrtastar_unsolved, _ = get_mean_std(
#     slip_results['lrtastar'])
# qlearning_mean_results, qlearning_std_results, qlearning_unsolved, qlearning_epsilon = get_mean_std(
#     slip_results['qlearning'])
# eps_mean_results, eps_std_results, eps_unsolved, eps_epsilon = get_mean_std(
#     slip_results['eps'])

# print('SLIP')
# print('===============================================')
# print(rts_mean_results, rts_std_results, rts_unsolved)
# print(lrtastar_mean_results, lrtastar_std_results, lrtastar_unsolved)
# print(qlearning_mean_results, qlearning_std_results, qlearning_unsolved, qlearning_epsilon)
# print(eps_mean_results, eps_std_results, eps_unsolved, eps_epsilon)
# print('===============================================')
