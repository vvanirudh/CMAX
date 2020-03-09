# import rospy
import numpy as np

from odium.envs.pr2.pr2_3d_ros_env import pr2_3d_ros_env
from odium.envs.pr2.pr2_3d_env import pr2_3d_env
from odium.agents.pr2_agents.pr2_3d_rts_agent import pr2_3d_rts_agent
from odium.controllers.pr2_controller import pr2_controller
from odium.agents.pr2_agents.arguments import get_args

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


def launch(args):
    # rospy.init_node('pr2_rts_trainer', anonymous=True)
    env = pr2_3d_ros_env(args)
    # env = pr2_3d_env(args, np.array([8, 0]), np.array([1, 0]))
    planning_env = pr2_3d_env(args, env.start_cell,
                              env.goal_cell, env.get_obstacle_cells())
    controller = pr2_controller(pr2_3d_env(args, env.start_cell, env.goal_cell, env.get_obstacle_cells()),
                                num_expansions=args.n_expansions)
    pr2_rts_trainer = pr2_3d_rts_agent(args, env, planning_env, controller)

    # try:
    n_steps = pr2_rts_trainer.learn_online_in_real_world()
    print('Reached goal in', n_steps, 'steps')
    # except rospy.ROSInterruptException:
    #    pass


if __name__ == '__main__':
    args = get_args()
    launch(args)
