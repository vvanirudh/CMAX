import copy
import numpy as np

from gym.envs.robotics.utils import reset_mocap2body_xpos


def set_object_position_in_sim(env, new_object_pos):
    '''
    Setting new XY position of the object
    '''
    object_qpos = env.env.sim.data.get_joint_qpos(
        'object0:joint').copy()
    object_qpos[:2] = new_object_pos
    env.env.sim.data.set_joint_qpos(
        'object0:joint', object_qpos)
    env.env.sim.forward()
    # for _ in range(5):
    #     env.env.sim.step()
    return True


def set_gripper_position_in_sim(env, new_gripper_pos, old_gripper_pos):
    '''
    Setting new XY position of the gripper
    '''
    residual_pos = new_gripper_pos - old_gripper_pos
    reset_mocap2body_xpos(env.env.sim)
    mocap_pos = env.env.sim.data.get_mocap_pos(
        'robot0:mocap').copy()
    mocap_pos[0:2] = mocap_pos[0:2] + residual_pos
    env.env.sim.data.set_mocap_pos(
        'robot0:mocap', mocap_pos)
    env.env.sim.forward()
    for _ in range(2):
        env.env.sim.step()
    # env.env.sim.step()
    return True


def set_sim_state_and_goal(env, qpos, qvel, goal, reset=True):
    if reset:
        env.reset()
    set_goal(env, goal.copy())
    return set_sim_state(env, qpos, qvel)


def set_goal(env, goal):
    return env.env.set_goal(goal)


def set_sim_state(env, qpos, qvel):
    mj_sim_state = copy.deepcopy(env.env.sim.get_state())
    mj_sim_state.qpos[:] = qpos.copy()
    mj_sim_state.qvel[:] = qvel.copy()
    env.env.sim.set_state(mj_sim_state)
    env.env.sim.forward()
    observation = env.get_obs()
    return observation


def get_sim_state(env):
    mj_sim_state = copy.deepcopy(env.env.sim.get_state())
    qpos = mj_sim_state.qpos.copy()
    qvel = mj_sim_state.qvel.copy()

    return qpos, qvel


def get_sim_state_and_goal(env):
    goal = env.env.goal.copy()
    qpos, qvel = get_sim_state(env)

    return qpos, qvel, goal


def set_gridworld_state_and_goal(env,
                                 current_state,
                                 goal_state):
    env.current_state = current_state.copy()
    env.goal_state = goal_state.copy()
    return True


def apply_dynamics_residual(env,
                            residual_dynamics_fn,
                            observation,
                            info,
                            action,
                            next_observation):
    # Apply the learned dynamics residual
    residual_pos = residual_dynamics_fn(observation, np.array(action))
    next_obj_pos = next_observation['observation'][3:5].copy()
    next_gripper_pos = next_observation['observation'][0:2].copy()
    corrected_obj_pos = next_obj_pos + residual_pos[2:4]
    corrected_gripper_pos = next_gripper_pos + residual_pos[0:2]
    next_observation['observation'][0:2] = corrected_gripper_pos.copy()
    next_observation['observation'][3:5] = corrected_obj_pos.copy()
    next_observation['observation'][6:8] = corrected_obj_pos - \
        corrected_gripper_pos
    next_observation['achieved_goal'][0:2] = corrected_obj_pos.copy()
    # Set gripper position
    set_gripper_position_in_sim(
        env, corrected_gripper_pos, next_gripper_pos)
    # Set object position
    set_object_position_in_sim(env, corrected_obj_pos)
    # Compute reward
    rew = env.compute_reward(
        next_observation['achieved_goal'], next_observation['desired_goal'], None, info['penetration'])

    return next_observation, rew


def apply_4d_dynamics_residual(env,
                               residual_dynamics_fn,
                               observation,
                               ac,
                               next_observation):
    # Apply the learned dynamics residual
    residual_pos = residual_dynamics_fn(observation, ac)
    next_obj_pos = next_observation['continuous_observation'][2:].copy()
    next_gripper_pos = next_observation['continuous_observation'][0:2].copy()
    corrected_obj_pos = next_obj_pos + residual_pos[2:4]
    corrected_gripper_pos = next_gripper_pos + residual_pos[0:2]
    next_observation['continuous_observation'][0:2] = corrected_gripper_pos.copy()
    next_observation['continuous_observation'][2:4] = corrected_obj_pos.copy()
    grid_cell = env._continuous_to_grid(
        corrected_gripper_pos, corrected_obj_pos)
    next_observation['observation'] = grid_cell.copy()
    next_observation['achieved_goal'] = grid_cell[2:].copy()
    # Set gripper position
    set_gripper_position_in_sim(
        env, corrected_gripper_pos, next_gripper_pos)
    # Set object position
    set_object_position_in_sim(env, corrected_obj_pos)
    # Compute cost
    cost = env.compute_cost(
        next_observation['achieved_goal'], next_observation['desired_goal'])

    return next_observation, cost
