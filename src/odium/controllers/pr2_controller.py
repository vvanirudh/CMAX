import numpy as np
import copy

from odium.utils.graph_search_utils.astar import Node, Astar


class pr2_controller:
    def __init__(self, model, num_expansions=3):
        self.model, self.num_expansions = model, num_expansions
        self.model.reset()
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
        manhattan_dist = np.abs(self.model.goal_cell - obs).sum()
        return manhattan_dist

    def zero_heuristic(self, obs):
        return 0

    def get_successors(self, node, ac):
        obs = node.obs
        self.model.set_cell(obs)

        next_obs = self.model.successor(ac)
        cost = self.model.get_cost(obs, ac, next_obs)

        if self.discrepancy_fn is not None:
            cost = self.discrepancy_fn(obs, ac, cost)

        next_node = Node(next_obs)

        return next_node, cost

    def check_goal(self, node):
        obs = node.obs
        return self.model.check_goal(obs)

    def act(self, obs, greedy_path=None, n_steps=None):
        start_node = Node(obs)
        if greedy_path:
            node = start_node
            path = []
            path.append(node.obs)
            for _ in range(n_steps):
                # Get successors
                best_node = None
                best_heuristic = np.inf
                for ac in self.actions:
                    next_node, cost = self.get_successors(node, ac)
                    h = self.heuristic(next_node)
                    if cost + h < best_heuristic:
                        best_node = copy.deepcopy(next_node)
                        best_heuristic = cost + h
                node = best_node
                path.append(node.obs)

            return path

        best_action, info = self.astar.act(start_node)
        return best_action, info

    def reconfigure_heuristic(self, heuristic_fn):
        self.heuristic_fn = heuristic_fn
        return True

    def reconfigure_discrepancy(self, discrepancy_fn):
        self.discrepancy_fn = discrepancy_fn
        return True
