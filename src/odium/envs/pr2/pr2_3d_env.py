import numpy as np


class pr2_3d_env:
    def __init__(self, args, start_cell, goal_cell, obstacle_cells=None):
        self.args = args

        self.start_cell = start_cell.copy()
        self.goal_cell = goal_cell.copy()
        self.obstacle_cells = obstacle_cells
        self.initialize_cost_map()

        self.current_cell = self.start_cell.copy()

    def initialize_cost_map(self):
        self.cost_map = np.ones(
            (self.args.grid_size, self.args.grid_size, self.args.grid_size)) * 2
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
        self.cost_map[start[0], start[1], :] = 1
        # Top row should all be low
        self.cost_map[start[0], :, self.args.grid_size-1] = 1
        # Last column should all be low
        self.cost_map[goal[0], goal[1], :] = 1

        # Increase all cells corresponding to obstacle cells to large values
        if self.obstacle_cells:
            for cell in self.obstacle_cells:
                if self.out_of_bounds(cell):
                    continue
                self.cost_map[cell] = 1000

    def out_of_bounds(self, cell):
        np_cell = np.array(cell)
        if np.any(np_cell < 0) or np.any(np_cell >= self.args.grid_size):
            return True
        return False

    def set_cell(self, cell):
        self.current_cell = cell.copy()
        return self.current_cell

    def successor(self, ac):
        # Action space is simply an element of {0, 1, 2, 3, 4, 5}
        # 0 - move in -X, 1 - move in +X, 2 - move in -Y, 3 - move in +Y, 4 - move in -Z, 5 - move in +Z
        if ac == 0:
            next_cell = np.array(
                [max(self.current_cell[0] - 1, 0), self.current_cell[1], self.current_cell[2]])
        elif ac == 1:
            next_cell = np.array(
                [min(self.current_cell[0] + 1, self.args.grid_size-1),
                 self.current_cell[1], self.current_cell[2]])
        elif ac == 2:
            next_cell = np.array(
                [self.current_cell[0], max(self.current_cell[1]-1, 0), self.current_cell[2]])
        elif ac == 3:
            next_cell = np.array(
                [self.current_cell[0], min(self.current_cell[1]+1, self.args.grid_size-1),
                 self.current_cell[2]])
        elif ac == 4:
            next_cell = np.array(
                [self.current_cell[0], self.current_cell[1],
                    max(self.current_cell[2]-1, 0)]
            )
        elif ac == 5:
            next_cell = np.array(
                [self.current_cell[0], self.current_cell[1],
                    min(self.current_cell[2]+1, self.args.grid_size-1)]
            )
        else:
            raise Exception('Invalid action')

        # if tuple(next_cell) in self.obstacle_cells:
        #     # Obstacle, so remain in the current cell
        #     return self.current_cell

        return next_cell

    def get_cost(self, cell, ac, next_cell):
        return self.cost_map[next_cell[0], next_cell[1], next_cell[2]]

    def get_current_grid_cell(self):
        return self.current_cell.copy()

    def move_to_cell(self, newcell):
        self.current_cell = newcell.copy()
        return self.current_cell.copy()

    def get_actions(self):
        return np.arange(6)

    def reset(self):
        self.current_cell = self.start_cell
        return self.current_cell

    def check_goal(self, current_cell, goal_cell=None):
        if goal_cell is None:
            goal_cell = self.goal_cell.copy()
        if np.array_equal(current_cell, goal_cell):
            return True
        return False
