import os
import random
from mpi4py import MPI
import numpy as np

from odium.agents.hardcoded_agent.arguments import get_args
from odium.agents.hardcoded_agent.hardcoded_agent import hardcoded_agent
import odium.utils.logger as logger
from odium.utils.env_utils.make_env import make_env


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    assert args.env_name.startswith('Residual'), 'Only residual envs allowed'
    env = make_env(args.env_name)
    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # Configure logger
    if MPI.COMM_WORLD.Get_rank() == 0:
        if args.log_dir or logger.get_dir() is None:
            logger.configure(dir=os.path.join(
                'logs', 'hardcoded', args.log_dir), format_strs=['tensorboard', 'log', 'csv', 'json', 'stdout'])
        else:
            logger.configure(dir=os.path.join('logs', 'hardcoded', args.env_name), format_strs=[
                'tensorboard', 'log', 'csv', 'json', 'stdout'])
    args.log_dir = logger.get_dir()
    assert args.log_dir is not None
    os.makedirs(args.log_dir, exist_ok=True)

    env_params = get_env_params(env)
    hardcoded_controller = hardcoded_agent(args, env, env_params)
    hardcoded_controller.eval_agent()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
