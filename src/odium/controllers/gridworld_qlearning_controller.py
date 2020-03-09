import numpy as np

from odium.controllers.controller import Controller


class GridWorldQlearningController(Controller):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.qvalue_fn = None

    def get_initial_state_values(self, goal_state):
        self.initial_state_values = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.initial_state_values[x, y] = self.manhattan_dist(
                    np.array([x, y]), goal_state)

        return self.initial_state_values

    def manhattan_dist(self, state, goal_state):
        return np.abs(goal_state - state).sum()

    def act(self, obs):
        if self.qvalue_fn is None:
            return None
        else:
            qvalues = self.qvalue_fn(obs)
            action = np.argmin(qvalues)
            return action

    def reconfigure_qvalue_fn(self, qvalue_fn):
        self.qvalue_fn = qvalue_fn


def get_gridworld_qlearning_controller(grid_size):
    return GridWorldQlearningController(grid_size)
