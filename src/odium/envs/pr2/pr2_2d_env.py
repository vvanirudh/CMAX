import numpy as np


class pr2_2d_env:
    def __init__(self, args, start_cell, goal_cell):
        self.args = args

        self.start_cell = start_cell.copy()
        self.goal_cell = goal_cell.copy()
        self.initialize_cost_map()

        self.current_cell = self.start_cell

    def initialize_cost_map(self):
        self.cost_map = np.ones(
            (self.args.grid_size, self.args.grid_size)) * 10
        # LEFT to RIGHT cost should decrease
        # for x in range(self.args.grid_size):
        #     self.cost_map[x, :] = np.flip(np.arange(self.args.grid_size))

        # # DOWN to UP cost should decrease in the left half
        # for y in range(self.args.grid_size // 2):
        #     self.cost_map[:, y] += np.flip(np.arange(self.args.grid_size))
        # # DOWN to UP cost should increase in the right half
        # for y in range(self.args.grid_size // 2, self.args.grid_size):
        #     self.cost_map[:, y] += np.arange(self.args.grid_size)

        goal = self.goal_cell
        start = self.start_cell
        # First column should be all low
        self.cost_map[start[0], :] = 1
        # Top row should all be low
        self.cost_map[:, self.args.grid_size-1] = 1
        # Last column should all be low
        self.cost_map[goal[0], ] = 1

    def set_cell(self, cell):
        self.current_cell = cell.copy()
        return self.current_cell

    def successor(self, ac):
        # Action space is simply an element of {0, 1, 2, 3}
        # 0 - move left, 1 - move up, 2 - move down, 3 - move right
        # left-right is X direction, up-down is Y direction
        if ac == 0:
            return np.array([max(self.current_cell[0] - 1, 0), self.current_cell[1]])
        elif ac == 1:
            return np.array([self.current_cell[0], min(self.current_cell[1] + 1, self.args.grid_size-1)])
        elif ac == 2:
            return np.array([self.current_cell[0], max(self.current_cell[1] - 1, 0)])
        elif ac == 3:
            return np.array([min(self.current_cell[0] + 1, self.args.grid_size-1), self.current_cell[1]])
        else:
            raise Exception('Invalid action')

    def get_cost(self, cell):
        return self.cost_map[cell[0], cell[1]]

    def get_current_grid_cell(self):
        return self.current_cell.copy()

    def move_to_cell(self, newcell):
        self.current_cell = newcell.copy()
        return self.current_cell.copy()

    def get_actions(self):
        return np.arange(4)

    def reset(self):
        self.current_cell = self.start_cell
        return self.current_cell

    def check_goal(self, current_cell, goal_cell=None):
        if goal_cell is None:
            goal_cell = self.goal_cell.copy()
        if np.array_equal(current_cell, goal_cell):
            return True
        return False

    def set_start_and_goal(self, start_cell, goal_cell):
        self.start_cell = start_cell.copy()
        self.goal_cell = goal_cell.copy()
        self.initialize_cost_map()
        return True
