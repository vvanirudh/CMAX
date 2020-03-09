import numpy as np

from odium.controllers.controller import Controller
from odium.utils.graph_search_utils.astar import Node, Astar
from odium.utils.simulation_utils import set_gridworld_state_and_goal

from odium.envs.gridworld.gridworld_env import make_gridworld_env


class GridWorldController(Controller):
    def __init__(self,
                 model,
                 num_expansions=3):
        super(GridWorldController, self).__init__()
        self.model = model
        self.model.reset()
        self.num_expansions = num_expansions
        self.actions = self.model.get_actions()

        self.astar = Astar(self.heuristic,
                           self.get_successors,
                           self.check_goal,
                           num_expansions,
                           self.actions)

        self.heuristic_fn = None
        self.discrepancy_fn = None

    def heuristic(self, node, augment=True):
        if isinstance(node, Node):
            obs = node.obs
        else:
            obs = node

        if self.heuristic_fn is not None and augment:
            steps_to_goal = self.heuristic_fn(obs)

        return steps_to_goal

    def manhattan_dist(self, obs):
        current_state = obs['observation'].copy()
        goal_state = obs['desired_goal'].copy()

        # Simple manhattan distance would do
        manhattan_dist = np.abs(goal_state - current_state).sum()

        return manhattan_dist

    def get_successors(self, node, action):
        obs = node.obs
        set_gridworld_state_and_goal(
            self.model,
            obs['observation'].copy(),
            obs['desired_goal'].copy(),
        )

        next_obs, cost, _, _ = self.model.step(action)

        if self.discrepancy_fn is not None:
            cost = self.discrepancy_fn(obs, action, cost, next_obs)

        next_node = Node(next_obs)

        return next_node, cost

    def check_goal(self, node):
        obs = node.obs
        current_state = obs['observation'].copy()
        goal_state = obs['desired_goal'].copy()
        return self.model.check_goal(current_state, goal_state)

    def act(self, obs):
        start_node = Node(obs)
        best_action, info = self.astar.act(start_node)
        return best_action, info

    def reconfigure_heuristic(self, heuristic_fn):
        self.heuristic_fn = heuristic_fn
        return True

    def reconfigure_discrepancy(self, discrepancy_fn):
        self.discrepancy_fn = discrepancy_fn
        return True


def get_gridworld_controller(env, grid_size, n_expansions):
    return GridWorldController(make_gridworld_env(env, grid_size), n_expansions)
