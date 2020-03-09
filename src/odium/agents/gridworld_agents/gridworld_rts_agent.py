from odium.utils.simulation_utils import set_gridworld_state_and_goal
import numpy as np
import copy


class gridworld_rts_agent:
    def __init__(self, args, env, planning_env, controller):
        # Store all given arguments
        self.args, self.env = args, env
        self.planning_env, self.controller = planning_env, controller

        self.state_values = np.zeros((args.grid_size, args.grid_size))
        self._fill_state_values(env)
        self.discrepancy_matrix = np.zeros((4, args.grid_size, args.grid_size))

    def get_state_value(self, obs):
        current_state = obs['observation']
        return self.state_values[tuple(current_state)]

    def get_discrepancy(self, obs, ac, cost, next_obs):
        current_state = obs['observation']
        # next_state = next_obs['observation']
        if self.discrepancy_matrix[ac, current_state[0], current_state[1]] > 0:
            # If discrepancy
            # cost = cost + \
            #     self.discrepancy_matrix[ac, current_state[0], current_state[1]]
            cost = self.args.grid_size**2

        return cost

    def _fill_state_values(self, env):
        goal_state = env.goal_state
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                # Fill with manhattan distances
                self.state_values[i, j] = np.abs(
                    goal_state - np.array([i, j])).sum()

        return True

    def learn_online_in_real_world(self, max_timesteps=None):
        # Reset environment
        obs = self.env.reset()
        # Configure heuristic for controller
        self.controller.reconfigure_heuristic(
            self.get_state_value
        )
        self.controller.reconfigure_discrepancy(
            self.get_discrepancy
        )

        total_n_steps = 0
        while True:
            current_state = obs['observation'].copy()
            ac, info_online = self.controller.act(obs)
            next_obs, cost, _, _ = self.env.step(ac)
            if self.args.verbose:
                print('t', total_n_steps)
                print('STATE', current_state)
                print('ACTION', ac)
                print('VALUE PREDICTED', info_online['start_node_h'])
            if self.env.check_goal(next_obs['observation'],
                                   next_obs['desired_goal']):
                print('REACHED GOAL!')
                break
            total_n_steps += 1
            # # Get the next obs in planning env
            set_gridworld_state_and_goal(self.planning_env,
                                         obs['observation'],
                                         obs['desired_goal'])
            next_obs_sim, _, _, _ = self.planning_env.step(ac)
            if not np.array_equal(next_obs['observation'],
                                  next_obs_sim['observation']):
                # Report discrepancy
                self.discrepancy_matrix[ac,
                                        current_state[0], current_state[1]] += 1
            # Plan in model
            # Do a simple RTDP update
            _, info = self.controller.act(obs)
            # # Update heuristic for all states on closed list
            for node in info['closed']:
                state = node.obs['observation']
                gval = node._g
                self.state_values[tuple(
                    state)] = info['best_node_f'] - gval
            # Update only the current state
            self.state_values[tuple(current_state)] = info['best_node_f']
            # Move to next iteration
            obs = copy.deepcopy(next_obs)

            if max_timesteps and total_n_steps >= max_timesteps:
                break

        return total_n_steps
