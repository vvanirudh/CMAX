import copy
import numpy as np
from odium.utils.graph_search_utils.astar import Node, Astar
from odium.utils.simulation_utils import set_sim_state_and_goal, apply_dynamics_residual
from odium.controllers.controller import Controller
from odium.agents.fetch_agents.discrepancy_utils import apply_discrepancy_penalty


class fetch_push_astar_controller(Controller):
    def __init__(self,
                 model,
                 num_expansions=10,
                 discrete=False,
                 reward_type='sparse'):
        '''
        model - gym env (should be wrapper env)
        num_expansions - Number of expansions to be done by A*
        discrete - Is the env discrete action space or not?
        reward_type - env cost function sparse or dense?
        '''
        super(fetch_push_astar_controller, self).__init__()
        self.model = model
        self.model.reset()
        self.num_expansions = num_expansions
        self.discrete = discrete
        self.reward_type = reward_type
        self.n_bins = 3  # TODO: Make an argument. Also needs to be the same as that of env
        if discrete:
            self.discrete_actions = self.model.discrete_actions

        self.astar = Astar(self.heuristic,
                           self.get_successors,
                           self.check_goal,
                           num_expansions,
                           self.actions())

        self.residual_heuristic_fn = None
        self.discrepancy_fn = None
        self.residual_dynamics_fn = None

    def actions(self):
        if not self.discrete:
            scale = 1.
            return [
                (-scale, 0., 0., 0.),
                (scale, 0., 0., 0.),
                (0., -scale, 0., 0.),
                (0., scale, 0., 0.),
            ]
        else:
            # Return all discrete actions except the null action
            acs = self.discrete_actions
            if (0, 0, 0, 0) in acs:
                del acs[(0, 0, 0, 0)]
            return acs

    def heuristic(self, node, augment=True):
        obs = node.obs
        gripper_position = obs['observation'][:3]
        block_position = obs['observation'][3:6]
        goal = obs['desired_goal']

        block_width = 0.1
        block_to_goal_angle = np.arctan2(
            goal[0] - block_position[0], goal[1] - block_position[1])
        target_gripper_position = block_position.copy()
        target_gripper_position[0] += -1. * \
            np.sin(block_to_goal_angle) * block_width / 2.0
        target_gripper_position[1] += -1. * \
            np.cos(block_to_goal_angle) * block_width / 2.0
        target_gripper_position[2] += 0.005

        object_to_goal = np.abs(block_position[:2] - goal[:2])
        gripper_to_target = np.abs(
            gripper_position - target_gripper_position)
        object_to_goal_distance = np.linalg.norm(block_position[:2] - goal[:2])

        # If object is already near goal
        distance_threshold = 0.05
        if object_to_goal_distance < distance_threshold:
            return 0

        if self.reward_type == 'dense':
            # If its dense reward, then heuristic should capture
            # a weird sum of distances along the optimal path
            scale = 1.0
            steps_from_gripper_to_target = np.linalg.norm(
                gripper_position - target_gripper_position)
            steps_from_object_to_goal = np.linalg.norm(
                block_position[:2] - goal[:2])
        else:
            # If its sparse reward, then heuristic should capture number of steps
            if self.discrete:
                # Discrete actions take 0.04 step
                scale = 0.04
            else:
                # Continuous control action take 0.05 step
                scale = 0.05
            steps_from_object_to_goal = np.sum(
                np.round(object_to_goal / scale, decimals=2))
            steps_from_gripper_to_target = np.sum(np.round(
                gripper_to_target / scale, decimals=2))

        if (self.residual_heuristic_fn is None) or (not augment):
            return steps_from_object_to_goal + steps_from_gripper_to_target
        else:
            return steps_from_object_to_goal + steps_from_gripper_to_target + self.residual_heuristic_fn(obs)

    def heuristic_obs_g(self, observation, g):
        obs = {}
        obs['observation'] = observation
        obs['desired_goal'] = g
        node = Node(obs)
        return self.heuristic(node, augment=False)

    def get_qvalue(self, observation, ac):
        # Set the model state
        set_sim_state_and_goal(self.model,
                               observation['sim_state'].qpos.copy(),
                               observation['sim_state'].qvel.copy(),
                               observation['desired_goal'].copy())
        # Get next observation
        next_observation, rew, _, _ = self.model.step(np.array(ac))
        # Get heuristic of the next observation
        next_node = Node(next_observation)
        next_heuristic = self.heuristic(next_node, augment=False)

        return (-rew) + next_heuristic

    def get_qvalue_obs_ac(self, obs, g, qpos, qvel, ac):
        set_sim_state_and_goal(self.model,
                               qpos.copy(),
                               qvel.copy(),
                               g.copy())
        next_observation, rew, _, _ = self.model.step(np.array(ac))
        next_node = Node(next_observation)
        next_heuristic = self.heuristic(next_node, augment=False)

        return (-rew) + next_heuristic

    def get_all_qvalues(self, observation):
        qvalues = []
        for ac in self.model.discrete_actions_list:
            qvalues.append(self.get_qvalue(observation, ac))
        return np.array(qvalues)

    def get_successors(self, node, action):
        obs = node.obs
        set_sim_state_and_goal(
            self.model,
            obs['sim_state'].qpos.copy(),
            obs['sim_state'].qvel.copy(),
            obs['desired_goal'].copy()
        )
        next_obs, rew, _, info = self.model.step(
            np.array(action))

        if self.discrepancy_fn is not None:
            rew = apply_discrepancy_penalty(
                obs, action, rew, self.discrepancy_fn)

        if self.residual_dynamics_fn is not None:
            next_obs, rew = apply_dynamics_residual(self.model,
                                                    self.residual_dynamics_fn,
                                                    obs,
                                                    info,
                                                    action,
                                                    next_obs)

        mj_sim_state = copy.deepcopy(self.model.env.sim.get_state())
        next_obs['sim_state'] = mj_sim_state
        next_node = Node(next_obs)

        return next_node, -rew

    def check_goal(self, node):
        obs = node.obs
        obj_position = obs['achieved_goal']
        goal_position = obs['desired_goal']
        return self.model.env._is_success(obj_position, goal_position)

    def act(self, obs):
        start_node = Node(obs)
        best_action, info = self.astar.act(
            start_node)
        info['start_node_h_estimate'] = self.heuristic(
            start_node, augment=False)
        return np.array(best_action), info

    def reconfigure_heuristic(self, residual_heuristic_fn):
        self.residual_heuristic_fn = residual_heuristic_fn
        return True

    def reconfigure_num_expansions(self, n_expansions):
        self.num_expansions = n_expansions
        self.astar = Astar(self.heuristic,
                           self.get_successors,
                           self.check_goal,
                           self.num_expansions,
                           self.actions())
        return True

    def reconfigure_discrepancy(self, discrepancy_fn):
        self.discrepancy_fn = discrepancy_fn
        return True

    def reconfigure_residual_dynamics(self, residual_dynamics_fn):
        self.residual_dynamics_fn = residual_dynamics_fn
        return True
