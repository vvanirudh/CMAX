import pickle
from odium.experiment.train_pr2_7d_rts import launch
from odium.envs.pr2.pr2_7d_ros_env import pr2_7d_ros_env
from odium.agents.pr2_agents.arguments import get_args

args = get_args()
args.goal_tolerance = 0.1
args.n_expansions = 10
args.max_timesteps = 300
args.grid_size = 20
args.weight = 1
args.gamma = 0.1
args.neighbor_radius = 10
args.goal_distance_heuristic = True
env = pr2_7d_ros_env(args, table=False)

joints_data = []
for broken_joint in [1]:
    trials_data = []
    trial_num = 0
    while trial_num < 10:
        env._setup_env()
        steps_data = []
        # Sample start and goal
        env.broken_joint = False
        start_config, start_cell, goal_config, goal_cell, goal_pose = env.sample_valid_start_and_goal(
            broken_joint)

        args.broken_joint = True
        args.broken_joint_index = broken_joint
        env.set_start_and_goal(start_config, goal_config)
        args.agent = 'rts'
        n_steps, broken_joint_moved, n_times = launch(args, env)
        if not broken_joint_moved or n_steps == 100 or n_steps <= 10 or n_times <= 4:
            continue
        print('BROKEN JOINT MOVED!')
        # radius_steps_data.append(n_steps)

        env.broken_joint = True
        env.broken_joint_index = broken_joint

        # First, our approach
        env.set_start_and_goal(start_config, goal_config)
        args.agent = 'rts'
        n_steps, _, _ = launch(args, env)
        steps_data.append(n_steps)
        print('RTS reached in', n_steps)
        # Now, RTAA*
        env.set_start_and_goal(start_config, goal_config)
        args.agent = 'rtaastar'
        n_steps, _, _ = launch(args, env)
        steps_data.append(n_steps)
        print('RTAA* reached in', n_steps)
        trials_data.append(steps_data)
        trial_num += 1

        pickle.dump(trials_data, open(
            '/home/avemula/workspaces/odium_ws/src/odium/save/pr2_experiments_rtaastar.pkl', 'wb'))

    joints_data.append(trials_data)

print(joints_data)
pickle.dump(joints_data, open(
    '/home/avemula/workspaces/odium_ws/src/odium/save/pr2_experiments_rtaastar.pkl', 'wb'))
