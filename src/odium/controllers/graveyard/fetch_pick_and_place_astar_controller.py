import numpy as np
from odium.utils.graph_search_utils.astar import Node, Astar
from odium.controllers.controller import Controller


class FetchPickAndPlaceAstarController(Controller):
    def __init__(self, model, action_scale, num_expansions=10, discrete=False):
        super(FetchPickAndPlaceAstarController, self).__init__()
        # TODO: Get rid of action_scale
        self.model = model
        self.model.reset()
        self.action_scale = action_scale
        self.num_expansions = num_expansions
        self.discrete = discrete
        self.n_bins = 3
        self.discrete_actions = self.model.discrete_actions

        self.astar = Astar(self.heuristic, self.get_successors, self.check_goal,
                           num_expansions, self.actions())

        self.relative_grasp_position = (0., 0., -0.02)
        self.atol = 1e-3

    def actions(self):
        if not self.discrete:
            scale = 1.
            return [
                (-scale, 0., 0., -1),
                (scale, 0., 0., -1),
                (0., -scale, 0., -1.),
                (0., scale, 0., -1.),
                (0., 0., -scale, -1.),
                (-0., 0., scale, -1.),
                (-scale, 0., 0., 1.),
                (scale, 0., 0., 1.),
                (0., -scale, 0., 1.),
                (0., scale, 0., 1.),
                (0., 0., -scale, 1.),
                (-0., 0., scale, 1.),
            ]
        else:
            return self.discrete_actions

    def heuristic(self, node):
        obs = node.obs
        gripper_position = obs['observation'][:3]
        block_position = obs['observation'][3:6]
        goal = obs['desired_goal']
        gripper_state = obs['observation'][9:11]

        # Object to goal distance
        object_to_goal_distance = int(np.linalg.norm(
            np.subtract(block_position, goal)) / 0.01)

        # Gripper to object distance
        gripper_to_object_distance = int(np.linalg.norm(
            np.subtract(gripper_position, block_position)) / 0.01)

        # If block is close and the gripper is open
        relative_position = np.subtract(gripper_position, block_position)
        if np.sum(np.subtract(relative_position, self.relative_grasp_position)**2) < self.atol:
            # Block is inside grippers
            if abs(gripper_state[0] - 0.05) < self.atol:
                # Grippers are open
                gripper_cost = 1
            else:
                # Grippers are closed
                gripper_cost = 0
        else:
            # Block is not inside grippers
            if abs(gripper_state[0] - 0.05) < self.atol:
                # Grippers are open
                gripper_cost = 0
            else:
                # Grippers are closed
                gripper_cost = 1

        hval = 10 * object_to_goal_distance + \
            gripper_to_object_distance + 1000*gripper_cost

        return hval

    def get_successors(self, node, action):
        obs = node.obs
        self.model.env.goal = obs['desired_goal']
        mj_sim_state = self.model.env.sim.get_state()
        mj_sim_state_qpos_size = mj_sim_state.qpos.size
        mj_sim_state_qvel_size = mj_sim_state.qvel.size
        mj_sim_state.qpos[:] = obs['sim_state'].qpos[:mj_sim_state_qpos_size]
        mj_sim_state.qvel[:] = obs['sim_state'].qvel[:mj_sim_state_qvel_size]
        self.model.env.sim.set_state(mj_sim_state)
        self.model.env.sim.forward()

        next_obs, rew, _, _ = self.model.step(
            self.action_scale * np.array(action))
        mj_sim_state = self.model.env.sim.get_state()
        next_obs['sim_state'] = mj_sim_state

        next_node = Node(next_obs)

        return next_node, -rew

    def check_goal(self, node):
        obs = node.obs
        obj_position = obs['achieved_goal']
        goal_position = obs['desired_goal']
        rew = self.model.compute_reward(obj_position, goal_position, {})
        if rew == 0:
            return True
        return False

    def act(self, obs):
        start_node = Node(obs)
        best_action, info = self.astar.act(start_node)
        return np.array(best_action), info
