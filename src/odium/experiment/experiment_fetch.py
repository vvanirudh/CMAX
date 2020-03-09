import numpy as np
import pickle
import os.path as osp
import os

from odium.experiment.train_fetch import launch
from odium.agents.fetch_agents.arguments import get_args

# Configure experiment
np.random.seed(0)
seeds = np.random.randint(100000, size=20, dtype=int)
max_timesteps = 1000
path = osp.join(os.environ["HOME"], 'workspaces/odium_ws/src/odium/save')

# Configure arguments
args = get_args()


def load_args(args):
    if args.agent == 'dqn':
        args.batch_size = 64
        args.her = True
        args.replay_k = 4
        args.dqn_epsilon = 0.3
        args.polyak = 0.9
        args.n_online_planning_updates = 5
        args.planning_rollout_length = 3
        args.max_timesteps = max_timesteps
    elif args.agent == 'mbpo' or args.agent == 'mbpo_gp':
        args.batch_size = 64
        args.her = True
        args.replay_k = 4
        args.n_expansions = 5
        args.n_offline_expansions = 5
        args.n_rts_workers = 12 if args.n_rts_workers is None else args.n_rts_workers
        args.polyak = 0.0
        args.n_online_planning_updates = 5
        args.planning_rollout_length = 3
        args.max_timesteps = max_timesteps
    elif args.agent == 'rts' or args.agent == 'mbpo_knn':
        args.batch_size = 64
        args.her = True
        args.replay_k = 4
        args.n_expansions = 5
        args.n_offline_expansions = 5
        args.n_rts_workers = 12 if args.n_rts_workers is None else args.n_rts_workers
        args.neighbor_radius = 0.02 if args.agent == 'rts' else 0.04
        args.dynamic_residual_threshold = 1e-2
        args.polyak = 0.0
        args.n_online_planning_updates = 5
        args.planning_rollout_length = 3
        args.max_timesteps = max_timesteps
    else:
        raise Exception('Invalid agent argument')


if args.exp_agent is None or args.exp_model is None:
    raise Exception('Agent or model argument is not given')


# if args.exp_world is None or args.exp_world == 'empty':
if args.exp_model == 'accurate':

    # EMPTY WORLD EXPERIMENTS

    if args.exp_agent == 'dqn':

        # DQN EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 1
        args.agent = 'dqn'
        # args.render = True
        load_args(args)

        dqn_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            dqn_results.append(n_steps)

        pickle.dump(dqn_results, open(
            osp.join(path, 'fetch_experiments_accurate_model_dqn_results.pkl'), 'wb'))

    if args.exp_agent == 'mbpo':

        # MBPO EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 1
        args.agent = 'mbpo'
        # args.render = True
        load_args(args)

        mbpo_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            mbpo_results.append(n_steps)

        pickle.dump(mbpo_results, open(
            osp.join(path, 'fetch_experiments_accurate_model_mbpo_results.pkl'), 'wb'))

    if args.exp_agent == 'rts':

        # RTS EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 1
        args.agent = 'rts'
        # args.render = True
        load_args(args)

        rts_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            rts_results.append(n_steps)

        pickle.dump(rts_results, open(
            osp.join(path, 'fetch_experiments_accurate_model_rts_results.pkl'), 'wb'))

    if args.exp_agent == 'rts_correct':
        raise Exception('Useless experiment')

    if args.exp_agent == 'mbpo_knn':

        # MBPO EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 1
        args.agent = 'mbpo_knn'
        # args.render = True
        load_args(args)

        mbpo_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            mbpo_results.append(n_steps)

        pickle.dump(mbpo_results, open(
            osp.join(path, 'fetch_experiments_accurate_model_mbpo_knn_results.pkl'), 'wb'))

    if args.exp_agent == 'mbpo_gp':

        # MBPO EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 1
        args.agent = 'mbpo_gp'
        # args.render = True
        load_args(args)

        mbpo_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            mbpo_results.append(n_steps)

        pickle.dump(mbpo_results, open(
            osp.join(path, 'fetch_experiments_accurate_model_mbpo_gp_results.pkl'), 'wb'))

# if args.exp_world is None or args.exp_world == 'obstacle':
if args.exp_model == 'inaccurate':

    # OBSTACLE WORLD EXPERIMENTS

    if args.exp_agent == 'dqn':

        # DQN EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 4
        args.agent = 'dqn'
        # args.render = True
        load_args(args)

        dqn_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            dqn_results.append(n_steps)

        pickle.dump(dqn_results, open(
            osp.join(path, 'fetch_experiments_inaccurate_model_dqn_results.pkl'), 'wb'))

    if args.exp_agent == 'mbpo':

        # MBPO EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 4
        args.agent = 'mbpo'
        # args.render = True
        load_args(args)

        mbpo_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            mbpo_results.append(n_steps)

        pickle.dump(mbpo_results, open(
            osp.join(path, 'fetch_experiments_inaccurate_model_mbpo_results.pkl'), 'wb'))

    if args.exp_agent == 'rts':

        # RTS EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 4
        args.agent = 'rts'
        # args.render = True
        load_args(args)

        rts_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            rts_results.append(n_steps)

        pickle.dump(rts_results, open(
            osp.join(path, 'fetch_experiments_inaccurate_model_rts_results.pkl'), 'wb'))

    if args.exp_agent == 'rts_correct':

        # RTS Correct EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 1
        args.planning_env_id = 1
        args.agent = 'rts'
        # args.render = True
        load_args(args)

        rts_correct_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            rts_correct_results.append(n_steps)

        pickle.dump(rts_correct_results, open(
            osp.join(path, 'fetch_experiments_inaccurate_model_rts_correct_results.pkl'), 'wb'))

    if args.exp_agent == 'mbpo_knn':

        # MBPO EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 4
        args.agent = 'mbpo_knn'
        # args.render = True
        load_args(args)

        mbpo_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            mbpo_results.append(n_steps)

        pickle.dump(mbpo_results, open(
            osp.join(path, 'fetch_experiments_inaccurate_model_mbpo_knn_results.pkl'), 'wb'))

    if args.exp_agent == 'mbpo_gp':

        # MBPO EXPERIMENTS
        args = get_args()
        args.env_id = 1
        args.planning_env_id = 4
        args.agent = 'mbpo_gp'
        # args.render = True
        load_args(args)

        mbpo_results = []
        for seed in seeds:
            args.seed = int(seed)
            n_steps = launch(args)
            mbpo_results.append(n_steps)

        pickle.dump(mbpo_results, open(
            osp.join(path, 'fetch_experiments_inaccurate_model_mbpo_gp_results.pkl'), 'wb'))
