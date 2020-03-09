'''
This file contains the simulator for the PR2 7D env
'''
import rospy
import numpy as np


class pr2_7d_env:
    def __init__(self, args, start_cell, goal_pose, get_pose_fn, check_goal_fn):
        self.args = args

        self.start_cell = start_cell.copy()
        self.goal_pose = goal_pose
        self.get_pose = get_pose_fn
        self.check_goal = check_goal_fn

        self.current_cell = self.start_cell.copy()

        self.transition_dict = {}

    def out_of_bounds(self, cell):
        np_cell = np.array(cell)
        if np.any(np_cell < 0) or np.any(np_cell >= self.args.grid_size):
            return True
        return False

    def set_cell(self, cell):
        self.current_cell = cell.copy()
        return self.current_cell

    def successor(self, ac):
        '''
        ac should be one of 14 values (2 actions for each joint)
        '''
        # TODO: Do we not need a null action?
        if ac >= 14 or ac < 0:
            raise Exception('Invalid action')

        if self.current_cell.tobytes() in self.transition_dict:
            if ac in self.transition_dict[self.current_cell.tobytes()]:
                return self.transition_dict[self.current_cell.tobytes()][ac]

        joint_index = int(ac / 2)
        displacement = ac % 2

        # Displace the correct joint by the right displacement
        next_cell = self.current_cell.copy()
        next_cell[joint_index] = min(max(
            next_cell[joint_index] + (2 * displacement - 1), 0), self.args.grid_size - 1)

        return next_cell

    def get_cost(self, cell, ac, next_cell):
        if self.check_goal(next_cell):
            return 0
        else:
            # The cost should be the reduction in distance
            prev_goal_distance = np.linalg.norm(
                self.get_pose(cell) - self.goal_pose)
            next_goal_distance = np.linalg.norm(
                self.get_pose(next_cell) - self.goal_pose)

            reduction_in_distance = next_goal_distance - prev_goal_distance
            if np.abs(reduction_in_distance) < 1e-3 and (not self.args.goal_6d):
                # Stupid wrist rotations
                return 1
            if self.args.goal_distance_heuristic:
                return np.floor(next_goal_distance * 100) / 100.0
            else:
                return reduction_in_distance

    def get_current_grid_cell(self):
        return self.current_cell.copy()

    def move_to_cell(self, newcell):
        self.current_cell = newcell.copy()
        return self.current_cell.copy()

    def get_actions(self):
        return np.arange(14, dtype=int)

    def reset(self):
        self.current_cell = self.start_cell
        return self.current_cell

    def check_goal(self, current_cell):
        current_pose = self.get_pose(current_cell)
        # If its close enough then, return True
        if np.linalg.norm(current_pose - self.goal_pose) < 1e-2:
            return True
        return False

    def update(self, cell, action, next_cell):
        bytes_cell = cell.tobytes()
        if bytes_cell not in self.transition_dict:
            self.transition_dict[bytes_cell] = {}
            self.transition_dict[bytes_cell][action] = next_cell

        else:
            self.transition_dict[bytes_cell][action] = next_cell
