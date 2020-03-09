'''
This file will contain the implementation of the 7D PR2 agent
'''
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KDTree
import time
import rospy
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.hidden = 32
#         self.fc1 = nn.Linear(7, self.hidden)
#         self.fc2 = nn.Linear(self.hidden, self.hidden)
#         self.fc3 = nn.Linear(self.hidden, 1)

#         self.fc3.weight.data = torch.zeros(1, self.hidden)
#         self.fc3.bias.data = torch.zeros(1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class pr2_7d_rts_agent:
    def __init__(self, args, env, planning_env, controller):
        self.args, self.env = args, env
        self.planning_env, self.controller = planning_env, controller

        # Get the FK function
        self.get_pose = self.env.get_pose

        # Residual heuristic
        if self.args.kernel:
            self.residual_heuristic = KernelRidge(alpha=args.alpha,
                                                  kernel='rbf',
                                                  gamma=args.gamma)
        else:
            # self.residual_heuristic = Network()
            # self.residual_heuristic_optim = optim.SGD(
            #     self.residual_heuristic.parameters(), lr=args.lr)
            raise NotImplementedError('Only kernel supported')
        self.is_fit = False

        # Discrepancy model
        self.kdtrees = [None for _ in range(
            self.planning_env.get_actions().shape[0])]
        self.neighbor_radius = self.args.neighbor_radius

        # Get the goal pose
        self.goal_pose = self.env.goal_pose

        # Memories
        self.discrepancy_memory = [[] for _ in range(
            self.planning_env.get_actions().shape[0])]
        self.states_memory = []

        # Weight for heuristic updates
        self.weight = args.weight

        # Broken joint, if present
        self.broken_joint = args.broken_joint
        self.broken_joint_index = args.broken_joint_index
        # self.broken_joint_lower_limit = args.broken_lower_limit
        # self.broken_joint_upper_limit = args.broken_upper_limit
        self.max_timesteps = args.max_timesteps if args.max_timesteps else np.inf

    def get_heuristic(self, cell, augment=True):
        # Get the pose of the cell
        pose = self.get_pose(cell)
        # Compute a simple euclidean heuristic
        heuristic = np.floor(np.linalg.norm(
            pose - self.goal_pose) * 100) / 100.0
        # heuristic = 0
        if heuristic < self.args.goal_tolerance:
            # In the goal region
            return 0.0
        # Add the residual
        if self.is_fit and augment:
            if self.args.kernel:
                heuristic += self.residual_heuristic.predict(
                    cell.reshape(1, -1)).squeeze()
            else:
                # heuristic += self.residual_heuristic(
                #     torch.as_tensor(cell, dtype=torch.float32).view(1, -1)).item()
                raise NotImplementedError('Only kernel supported')

        return heuristic

    def get_discrepancy(self, cell, ac, cost):
        if self.kdtrees[ac] is None:
            return cost

        num_neighbors = self.kdtrees[ac].query_radius(cell.reshape(1, -1),
                                                      self.neighbor_radius,
                                                      count_only=True).squeeze()

        if num_neighbors > 0:
            cost = 1000000

        return cost

    def _add_to_discrepancy_memory(self, cell, ac):
        self.discrepancy_memory[ac].append(cell)

    def _add_to_states_memory(self, cell):
        self.states_memory.append(cell)

    def _update_discrepancy_model(self):
        for ac in self.planning_env.get_actions():
            # Get cells
            cells = self.discrepancy_memory[ac]
            # Add to the KDTree
            if len(cells) == 0:
                # No data points for this action
                continue

            self.kdtrees[ac] = KDTree(np.array(cells), metric='manhattan')

        # Configure the discrepancy model for the controller
        self.controller.reconfigure_discrepancy(self.get_discrepancy)

        return

    def _update_state_value_residual(self):
        if self.args.kernel:
            states = self.states_memory
        else:
            # states = np.array(self.states_memory)[indices, :]
            raise NotImplementedError('Only kernel supported')
        cells_closed = []
        values_closed = []
        heuristic_closed = []
        for cell in states:
            _, info = self.controller.act(cell)
            for k in info['closed']:
                cells_closed.append(k.obs)
                # Implement the weighted update
                values_closed.append(
                    info['best_node']._h + self.weight * (info['best_node']._g - k._g))
                # values_closed.append(info['best_node_f'] - k._g)
                heuristic_closed.append(
                    self.get_heuristic(k.obs, augment=False))

        cells_closed = np.array(cells_closed)
        values_closed = np.array(values_closed)
        heuristic_closed = np.array(heuristic_closed)
        # Compute target residuals
        targets_closed = values_closed - heuristic_closed
        if self.args.kernel:
            targets_closed = np.maximum(targets_closed, -heuristic_closed)
            self.residual_heuristic.fit(cells_closed, targets_closed)
            predicted_closed = self.residual_heuristic.predict(cells_closed)
            loss = np.mean(
                (predicted_closed - targets_closed)**2)
        else:
            # targets_closed = torch.as_tensor(
            #     targets_closed, dtype=torch.float32)
            # # Clip so that heuristic is never less than zero
            # targets_closed = torch.max(
            #     targets_closed, -torch.as_tensor(heuristic_closed, dtype=torch.float32))

            # predicted_closed = self.residual_heuristic(
            #     torch.as_tensor(cells_closed, dtype=torch.float32))

            # loss = (predicted_closed - targets_closed).pow(2).mean()
            # self.residual_heuristic_optim.zero_grad()
            # loss.backward()
            # self.residual_heuristic_optim.step()
            raise NotImplementedError('Only kernel supported')

        self.is_fit = True
        # Configure the heuristic for the controller
        self.controller.reconfigure_heuristic(self.get_heuristic)

        return loss

    def learn_online_in_real_world(self):
        # Get current cell
        cell = self.env.get_current_grid_cell()
        self.planning_env.set_cell(cell)

        broken_joint_moved = False
        n_broken_joint_moved = 0

        if self.broken_joint:
            broken_joint_state = cell[self.broken_joint_index]

        # Configure the heuristic for the controller
        self.controller.reconfigure_heuristic(self.get_heuristic)
        # Configure the discrepancy model for the controller
        self.controller.reconfigure_discrepancy(self.get_discrepancy)

        total_n_steps = 0
        start = time.time()
        while total_n_steps < self.max_timesteps:
            # print('-------------')
            # print('Time step', total_n_steps)
            # print('Current cell', cell)
            # print('Current cell heuristic',
            #       self.get_heuristic(cell))
            ac, info = self.controller.act(cell)
            path = self.controller.act(
                cell, greedy_path=True, n_steps=self.args.visualization_timesteps)
            self.env.visualize_path(path)
            # print('Action', ac)
            self.planning_env.set_cell(cell)
            cellsimnext = self.planning_env.successor(ac)
            cellnext = self.env.move_to_cell(cellsimnext.copy())
            total_n_steps += 1
            # print('Distance to goal', np.linalg.norm(
            #     self.get_pose(cellnext) - self.goal_pose))

            if self.broken_joint:
                # if cellnext[self.broken_joint_index] < self.broken_joint_lower_limit or cellnext[self.broken_joint_index] > self.broken_joint_upper_limit:
                if cellnext[self.broken_joint_index] != broken_joint_state:
                    # Broken joint moved
                    broken_joint_moved = True
                    n_broken_joint_moved += 1

            if self.env.check_goal(cellnext):
                # self.env.rotate_gripper()
                # rospy.sleep(1)
                # self.env.open_right_gripper()
                # rospy.sleep(1)
                # self.env.goto_postmanip_pose()
                # print('REACHED GOAL')
                break
            # Add to memory
            self._add_to_states_memory(cell)
            # Is there a discrepancy?
            # Extract only the non-revolute joints

            if np.abs(cellnext - cellsimnext).sum() >= self.args.discrepancy_threshold:
                # print('DISCREPANCY!')
                self.env._publish_discrepancy_marker('Discrepancy observed')
                # Update the discrepancy model
                self._add_to_discrepancy_memory(cell, ac)
                self._update_discrepancy_model()

            # Update the residual heuristic
            residual_loss = self._update_state_value_residual()
            # print('Residual loss', residual_loss)

            cell = cellnext.copy()

        end = time.time()
        print('Finished in time', end-start, 'secs')
        return total_n_steps, broken_joint_moved, n_broken_joint_moved
