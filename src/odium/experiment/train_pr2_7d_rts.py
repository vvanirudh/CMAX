import numpy as np
import random
from odium.envs.pr2.pr2_7d_ros_env import pr2_7d_ros_env
from odium.envs.pr2.pr2_7d_env import pr2_7d_env
from odium.agents.pr2_agents.pr2_7d_rts_agent import pr2_7d_rts_agent
from odium.agents.pr2_agents.pr2_7d_rtaastar_agent import pr2_7d_rtaastar_agent
from odium.controllers.pr2_controller import pr2_controller
from odium.agents.pr2_agents.arguments import get_args

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


def launch(args, env=None):
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.kernel = True
    if env is None:
        env = pr2_7d_ros_env(args)
    planning_env = pr2_7d_env(args,
                              env.start_cell,
                              env.goal_pose,
                              env.get_pose,
                              env.check_goal)
    controller = pr2_controller(planning_env, args.n_expansions)
    if args.agent == 'rts':
        pr2_rts_trainer = pr2_7d_rts_agent(args, env, planning_env, controller)
    elif args.agent == 'rtaastar':
        pr2_rts_trainer = pr2_7d_rtaastar_agent(
            args, env, planning_env, controller)

    n_steps, broken_joint_moved, n_broken_joint_moved = pr2_rts_trainer.learn_online_in_real_world()
    print('Reached goal in', n_steps, 'steps')
    return n_steps, broken_joint_moved, n_broken_joint_moved


if __name__ == '__main__':
    args = get_args()
    launch(args)
