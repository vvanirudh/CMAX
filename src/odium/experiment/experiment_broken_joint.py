import pickle
from odium.experiment.train_pr2_7d_rts import launch
from odium.envs.pr2.pr2_7d_ros_env import pr2_7d_ros_env
from odium.agents.pr2_agents.arguments import get_args

args = get_args()
args.goal_tolerance = 0.2
args.n_expansions = 5
args.max_timesteps = 100
args.grid_size = 15
args.weight = 10
args.goal_distance_heuristic = True
env = pr2_7d_ros_env(args, table=False)

joints_data = []
for broken_joint in [1]:
    trials_data = []
    trial_num = 0
    while trial_num < 10:
        gamma_steps_data = []
        # Sample start and goal
        env.broken_joint = False
        start_config, start_cell, goal_config, goal_cell, goal_pose = env.sample_valid_start_and_goal(
            broken_joint)

        args.broken_joint = True
        args.broken_joint_index = broken_joint
        env.set_start_and_goal(start_config, goal_config)
        args.gamma = 0.1
        n_steps, broken_joint_moved = launch(args, env)
        if not broken_joint_moved or n_steps == 100 or n_steps <= 10:
            continue
        print('BROKEN JOINT MOVED!')
        # radius_steps_data.append(n_steps)

        env.broken_joint = True
        env.broken_joint_index = broken_joint

        # for radius in [1, 3, 5, 7, 9]:
        # Larger gamma means more variance in the approximation
        # Smaller gamma means more smoothness in the approximation
        for gamma in [0.01, 0.1, 1.0, 10.0, 100.0]:
            env.set_start_and_goal(start_config, goal_config)
            # Set gamma
            args.gamma = gamma
            n_steps, _ = launch(args, env)
            gamma_steps_data.append(n_steps)

        trials_data.append(gamma_steps_data)
        trial_num += 1

    joints_data.append(trials_data)

print(joints_data)
pickle.dump(joints_data, open(
    '/home/avemula/workspaces/odium_ws/src/odium/save/pr2_experiments_broken_joint.pkl', 'wb'))
