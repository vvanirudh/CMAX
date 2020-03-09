import time
import numpy as np


class pr2_7d_rtaastar_agent:
    def __init__(self, args, env, planning_env, controller):
        self.args, self.env = args, env
        self.planning_env, self.controller = planning_env, controller

        # Get the FK function
        self.get_pose = self.env.get_pose

        # Get the goal pose
        self.goal_pose = self.env.goal_pose

        # Weight for heuristic updates
        self.weight = args.weight

        # Residual state values
        self.residual_cell_values = np.zeros(tuple([args.grid_size]*7))

        self.max_timesteps = args.max_timesteps if args.max_timesteps is not None else np.inf

    def get_cell_residual_value(self, cell):
        return self.residual_cell_values[tuple(cell)]

    def get_heuristic(self, cell, augment=True):
        pose = self.get_pose(cell)
        # heuristic = np.linalg.norm(pose - self.goal_pose)
        # heuristic = 0
        heuristic = np.floor(np.linalg.norm(
            pose - self.goal_pose) * 100) / 100.0
        if heuristic < self.args.goal_tolerance:
            return 0.0

        if augment:
            heuristic += self.get_cell_residual_value(cell)

        return heuristic

    def learn_online_in_real_world(self):
        cell = self.env.get_current_grid_cell()
        self.planning_env.set_cell(cell)

        self.controller.reconfigure_heuristic(self.get_heuristic)

        total_n_steps = 0
        start = time.time()
        while total_n_steps < self.max_timesteps:
            # print('-------------')
            # print('Time step', total_n_steps)
            # print('Current cell', cell)
            # print('Current cell heuristic',
            #       self.get_heuristic(cell))
            ac, info = self.controller.act(cell)

            # print('Action', ac)
            self.planning_env.set_cell(cell)
            cellsimnext = self.planning_env.successor(ac)
            cellnext = self.env.move_to_cell(cellsimnext.copy())
            total_n_steps += 1
            # print('Distance to goal', np.linalg.norm(
            #     self.get_pose(cellnext) - self.goal_pose))

            if self.env.check_goal(cellnext):
                break

            if not np.array_equal(cellnext, cellsimnext):
                # print('Discrepancy!')
                # Discrepancy
                self.planning_env.update(cell, ac, cellnext)

            _, info = self.controller.act(cell)
            for k in info['closed']:
                cell_closed = k.obs
                self.residual_cell_values[tuple(
                    cell_closed)] = info['best_node']._h + self.weight * (info['best_node']._g - k._g) - self.get_heuristic(cell_closed, augment=False)

            cell = cellnext.copy()

        end = time.time()
        print('Finished in time', end - start, 'secs')
        return total_n_steps, None, None
