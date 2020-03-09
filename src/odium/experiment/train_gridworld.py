import numpy as np
import random
import ray
# import rospy

from odium.envs.gridworld.gridworld_env import make_gridworld_env
from odium.controllers.gridworld_controller import get_gridworld_controller
from odium.controllers.gridworld_qlearning_controller import get_gridworld_qlearning_controller
from odium.agents.gridworld_agents.gridworld_rts_agent import gridworld_rts_agent
from odium.agents.gridworld_agents.gridworld_epsgreedy_agent import gridworld_epsgreedy_agent
from odium.agents.gridworld_agents.gridworld_lrtastar_agent import gridworld_lrtastar_agent
from odium.agents.gridworld_agents.gridworld_qlearning_agent import gridworld_qlearning_agent
from odium.agents.gridworld_agents.arguments import get_args

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


@ray.remote
def launch(args):
    # rospy.init_node('rts_trainer', anonymous=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = make_gridworld_env(args.env, args.grid_size,
                             args.render, args.incorrectness)
    planning_env = make_gridworld_env(
        args.planning_env, args.grid_size, False, args.incorrectness)
    controller = get_gridworld_controller(
        args.planning_env, args.grid_size, args.n_expansions)

    if args.agent == 'rts':
        gridworld_trainer = gridworld_rts_agent(
            args, env, planning_env, controller)
    elif args.agent == 'eps':
        gridworld_trainer = gridworld_epsgreedy_agent(
            args, env, controller)
    elif args.agent == 'lrtastar':
        gridworld_trainer = gridworld_lrtastar_agent(
            args, env, controller)
    elif args.agent == 'qlearning':
        controller = get_gridworld_qlearning_controller(args.grid_size)
        gridworld_trainer = gridworld_qlearning_agent(
            args, env, controller)
    # try:
    n_steps = gridworld_trainer.learn_online_in_real_world(args.max_timesteps)
    print('REACHED GOAL in', n_steps, 'by agent', args.agent)
    return n_steps
    # except rospy.ROSInterruptException or KeyboardInterrupt:
    #     pass


if __name__ == '__main__':
    args = get_args()
    ray.init()
    ray.get(launch.remote(args))
