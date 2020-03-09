import numpy as np
import time
from odium.utils.graph_search_utils.astar import Node
from odium.utils.graph_search_utils.dijkstra import Dijkstra


class pr2_2d_rts_agent:
    def __init__(self, args, env, planning_env, controller):
        self.args, self.env = args, env
        self.planning_env, self.controller = planning_env, controller

        self.start_cell = self.env.start_cell
        self.goal_cell = self.env.goal_cell

        self.cell_values = np.zeros((args.grid_size, args.grid_size))
        self._fill_cell_values(planning_env)
        self.discrepancy_matrix = np.zeros((4, args.grid_size, args.grid_size))

    def get_cell_value(self, cell):
        return self.cell_values[cell[0], cell[1]]

    def get_discrepancy(self, cell, ac, cost):
        # return cost + self.discrepancy_matrix[ac, cell[0], cell[1]]
        if self.discrepancy_matrix[ac, cell[0], cell[1]] > 0:
            cost = 1000000

        return cost

    def _fill_cell_values(self, env):
        '''
        Do Dijkstra and get good heuristic
        '''
        goal_cell = env.goal_cell
        goal_node = Node(goal_cell)
        dijkstra_search = Dijkstra(
            self.controller.get_successors, self.controller.actions)

        closed_set = dijkstra_search.get_dijkstra_heuristic(goal_node)

        for node in closed_set:
            cell = node.obs
            self.cell_values[cell[0], cell[1]] = node._g

        return True

    def learn_online_in_real_world(self):
        # Reset environment
        cell = self.env.get_current_grid_cell()
        self.planning_env.set_cell(cell)
        # Configure heuristic for controller
        self.controller.reconfigure_heuristic(
            self.get_cell_value
        )
        # Configure discrepancy
        self.controller.reconfigure_discrepancy(
            self.get_discrepancy
        )

        total_n_steps = 0
        while True:
            print('-------------')
            print('Current cell', cell)
            print('Current cell heuristic',
                  self.get_cell_value(cell))
            time.sleep(0.1)
            ac, _ = self.controller.act(cell)
            print('Action', ac)
            self.planning_env.set_cell(cell)
            cellsimnext = self.planning_env.successor(ac)
            cellnext = self.env.move_to_cell(cellsimnext)
            if self.env.check_goal(cellnext):
                print('REACHED GOAL!')
                # TODO: Open the grippers and move away
                break
            total_n_steps += 1
            # Is there a discrepancy?
            if cellnext[0] != cellsimnext[0] or cellnext[1] != cellsimnext[1]:
                print('Discrepancy!')
                self.discrepancy_matrix[ac, cell[0], cell[1]] += 1

            _, info = self.controller.act(cell)
            # Update all cells on closed list
            for k in info['closed']:
                cell = k.obs
                gval = k._g
                self.cell_values[tuple(
                    cell)] = info['best_node_f'] - gval
            # update only the current cell
            # self.cell_value_residual[tuple(
            #     cell)] = info['best_node_f'] - self.controller.zero_heuristic(cell)
            # Move to next iteration
            cell = cellnext.copy()

        return total_n_steps
