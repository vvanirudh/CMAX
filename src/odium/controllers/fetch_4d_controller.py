import copy
import numpy as np
from odium.utils.graph_search_utils.astar import Node, Astar
from odium.utils.simulation_utils import apply_4d_dynamics_residual
from odium.controllers.controller import Controller
from odium.agents.fetch_agents.discrepancy_utils import apply_4d_discrepancy_penalty


class fetch_4d_controller(Controller):
    def __init__(self, model, num_expansions):
        super(fetch_4d_controller, self).__init__()
        self.model = model
        self.model.reset()
        self.num_expansions = num_expansions

        self.astar = Astar(self.heuristic,
                           self.get_successors,
                           self.check_goal,
                           num_expansions,
                           self.actions())

        self.residual_heuristic_fn = None
        self.discrepancy_fn = None
        self.residual_dynamics_fn = None

    def actions(self):
        return np.arange(4)

    def heuristic(self, node, augment=True):
        obs = node.obs
        gripper_cell = obs['observation'][:2]
        obj_cell = obs['observation'][2:]
        goal_cell = obs['desired_goal']

        obj_to_goal_angle = np.arctan2(goal_cell[0] - obj_cell[0],
                                       goal_cell[1] - obj_cell[1])
        target_cell = obj_cell.copy()
        if np.sin(obj_to_goal_angle) >= 0:
            target_cell[0] += -1
        else:
            target_cell[0] += 1
        if np.cos(obj_to_goal_angle) >= 0:
            target_cell[1] += -1
        else:
            target_cell[1] += 1

        if self.model._is_success(obj_cell, goal_cell):
            # Already at goal
            return 0

        steps_from_obj_to_goal = np.abs(obj_cell - goal_cell).sum()
        steps_from_gripper_to_target = np.abs(gripper_cell - target_cell).sum()

        if self.residual_heuristic_fn is None or not augment:
            return steps_from_gripper_to_target + steps_from_obj_to_goal
        else:
            return steps_from_gripper_to_target + steps_from_obj_to_goal + self.residual_heuristic_fn(obs)

    def heuristic_obs_g(self, obs, g):
        observation = {'observation': obs, 'desired_goal': g}
        node = Node(observation)
        return self.heuristic(node, augment=False)

    def set_sim_state_and_goal(self, goal, qpos, qvel):
        # Set goal
        self.model.goal_cell = goal.copy()
        # Set sim state
        sim_state = self.model.sim.get_state()
        sim_state.qpos[:] = qpos.copy()
        sim_state.qvel[:] = qvel.copy()
        self.model.sim.set_state(sim_state)
        return self.model._get_obs()

    def get_qvalue(self, observation, ac):
        goal = observation['desired_goal'].copy()
        qpos = observation['sim_state'].qpos.copy()
        qvel = observation['sim_state'].qvel.copy()
        self.set_sim_state_and_goal(goal, qpos, qvel)
        # Get next observation
        next_observation, cost, _, _ = self.model.step(ac)
        # Get heuristic of next observation
        next_node = Node(next_observation)
        next_heuristic = self.heuristic(next_node, augment=False)

        return cost + next_heuristic

    def get_qvalue_obs_ac(self, obs, goal, qpos, qvel, ac):
        self.set_sim_state_and_goal(goal, qpos, qvel)
        # Get next observation
        next_observation, cost, _, _ = self.model.step(ac)
        # Get heuristic of next observation
        next_node = Node(next_observation)
        next_heuristic = self.heuristic(next_node, augment=False)

        return cost + next_heuristic

    def get_all_qvalues(self, observation):
        qvalues = []
        for ac in self.actions():
            qvalues.append(self.get_qvalue(observation, ac))
        return np.array(qvalues)

    def get_successors(self, node, ac):
        observation = node.obs
        self.set_sim_state_and_goal(observation['desired_goal'].copy(),
                                    observation['sim_state'].qpos.copy(),
                                    observation['sim_state'].qvel.copy())
        # Get next observation
        next_observation, cost, _, _ = self.model.step(ac)

        if self.discrepancy_fn is not None:
            cost = apply_4d_discrepancy_penalty(observation,
                                                ac,
                                                cost,
                                                self.discrepancy_fn)
        if self.residual_dynamics_fn is not None:
            next_observation, cost = apply_4d_dynamics_residual(self.model,
                                                                self.residual_dynamics_fn,
                                                                observation,
                                                                ac,
                                                                next_observation)

        mj_sim_state = copy.deepcopy(self.model.sim.get_state())
        next_observation['sim_state'] = mj_sim_state
        next_node = Node(next_observation)

        return next_node, cost

    def check_goal(self, node):
        observation = node.obs
        return self.model._is_success(observation['achieved_goal'], observation['desired_goal'])

    def act(self, observation):
        start_node = Node(observation)
        best_ac, info = self.astar.act(start_node)
        info['start_node_h_estimate'] = self.heuristic(
            start_node, augment=False)
        return np.array(best_ac), info

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
