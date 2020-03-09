import ray

from odium.utils.simple_utils import set_global_seed

from odium.agents.fetch_agents.arguments import get_args
from odium.agents.fetch_agents.fetch_4d_rts_agent import fetch_4d_rts_agent
from odium.controllers.fetch_4d_controller import fetch_4d_controller
from odium.envs.fetch_4d_env import fetch_4d_env

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


def get_env_params(args, env):
    # Construct envirnment params
    obs = env.reset()

    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'num_actions': 4,
              'qpos': obs['sim_state'].qpos.shape[0],
              'qvel': obs['sim_state'].qvel.shape[0],
              }

    return params


def launch(args):
    ray.init()

    env = fetch_4d_env(args.env_id, args.seed)
    planning_env = fetch_4d_env(args.planning_env_id, args.seed)

    set_global_seed(args.seed)

    controller = fetch_4d_controller(fetch_4d_env(args.planning_env_id, args.seed),
                                     args.num_expansions)
    env_params = get_env_params(args, env)

    fetch_trainer = fetch_4d_rts_agent(
        args, env_params, env, planning_env, controller)

    n_steps = fetch_trainer.learn_online_in_real_world(args.max_timesteps)
    print('REACHED GOAL in', n_steps, 'by agent', args.agent)
    ray.shutdown()
    return n_steps


if __name__ == '__main__':
    args = get_args()
    launch(args)
