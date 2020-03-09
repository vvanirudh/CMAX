import os
import os.path as osp
import ray
import time
import logging
# import rospy

from odium.utils.env_utils.make_env import make_env
from odium.utils.controller_utils.get_controller import get_controller
from odium.utils.simple_utils import set_global_seed
import odium.utils.logger as logger

from odium.agents.fetch_agents.fetch_rts_agent import fetch_rts_agent
from odium.agents.fetch_agents.fetch_dqn_agent import fetch_dqn_agent
from odium.agents.fetch_agents.arguments import get_args

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


def get_env_params(args, env):
    # Construct environment parameters
    obs = env.reset()

    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'num_actions': env.num_discrete_actions,
              'qpos': obs['sim_state'].qpos.shape[0],
              'qvel': obs['sim_state'].qvel.shape[0],
              'num_features': env.num_features,
              'max_timesteps': args.planning_rollout_length,
              'offline_max_timesteps': 50,
              }
    return params


def launch(args):
    # rospy.init_node('rts_trainer', anonymous=True)
    # Start ray
    ray.init(logging_level=logging.ERROR)
    # Create environments
    env = make_env(env_name=args.env_name,
                   env_id=args.env_id,
                   discrete=True,
                   reward_type=args.reward_type)
    planning_env = make_env(env_name=args.env_name,
                            env_id=args.planning_env_id,
                            discrete=True,
                            reward_type=args.reward_type)
    # Set random seeds
    env.seed(args.seed)
    planning_env.seed(args.seed)
    # Set global seeds
    set_global_seed(args.seed)
    # Make deterministic, if you have to
    if args.deterministic:
        env.make_deterministic()
        planning_env.make_deterministic()
    # Set logger level to debug, if you have to
    if args.debug:
        logger.set_level(logger.DEBUG)
    # Create controller
    controller = get_controller(env_name=args.env_name,
                                env_id=args.planning_env_id,
                                discrete=True,
                                num_expansions=args.n_expansions,
                                reward_type=args.reward_type,
                                seed=args.seed)
    # Configure logger
    if args.log_dir:
        logger.configure(dir=osp.join('logs', 'rts', args.log_dir),
                         format_strs=['tensorboard', 'log', 'csv', 'json', 'stdout'])
    os.makedirs(logger.get_dir(), exist_ok=True)

    # Configure save dir
    # if args.save_dir:
    #     args.save_dir = osp.join('saved', 'rts', args.save_dir)
    #     os.makedirs(args.save_dir, exist_ok=True)

    # if args.load_dir:
    #     args.load_dir = osp.join('saved', 'rts', args.load_dir)
    #     # TODO: CHeck if dir exists

    # Get env params
    env_params = get_env_params(args, env)
    # Get agent
    if args.agent == 'rts' or args.agent == 'mbpo' or args.agent == 'mbpo_knn' or args.agent == 'mbpo_gp':
        fetch_trainer = fetch_rts_agent(args,
                                        env_params,
                                        env,
                                        planning_env,
                                        controller)
    elif args.agent == 'dqn':
        fetch_trainer = fetch_dqn_agent(args,
                                        env_params,
                                        env,
                                        controller)
    # elif args.agent == 'mbpo':
    #     fetch_trainer = fetch_model_agent(args,
    #                                       env_params,
    #                                       env,
    #                                       planning_env,
    #                                       controller)
    # Start
    if args.offline:
        # Train in simulation
        raise Exception('Only online mode is required')
        fetch_trainer.learn_offline_in_model()
    else:
        n_steps = fetch_trainer.learn_online_in_real_world(args.max_timesteps)
        print('REACHED GOAL in', n_steps, 'by agent', args.agent)
        ray.shutdown()
        time.sleep(5)
        return n_steps


if __name__ == '__main__':
    args = get_args()
    launch(args)
