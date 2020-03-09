import pickle
import numpy as np
import os.path as osp
import os
from odium.experiment.train_fetch import launch
from odium.agents.fetch_agents.arguments import get_args

# Configure experiment
np.random.seed(0)
seeds = np.random.randint(1000, size=10, dtype=int)
max_timesteps = 400
path = osp.join(os.environ["HOME"], 'workspaces/odium_ws/src/odium/save')

# Configure args
args = get_args()
args.agent = 'rts'
args.env_id = 1
args.planning_env_id = 4
args.batch_size = 64
args.her = True
args.replay_k = 4
args.n_expansions = 5
args.n_offline_expansions = 5
args.n_rts_workers = 12 if args.n_rts_workers is None else args.n_rts_workers
args.dynamic_residual_threshold = 1e-2
args.polyak = 0.0
args.n_online_planning_updates = 5
args.planning_rollout_length = 3
args.max_timesteps = max_timesteps


radii = [0.01, 0.02, 0.04, 0.06, 0.08]

trial_data = []
for seed in seeds:
    args.seed = int(seed)
    radii_data = []
    for radius in radii:
        args.neighbor_radius = radius
        n_steps = launch(args)
        radii_data.append(n_steps)

    trial_data.append(radii_data)

pickle.dump(trial_data, open(
    osp.join(path, 'fetch_experiments_radius.pkl'), 'wb'))
