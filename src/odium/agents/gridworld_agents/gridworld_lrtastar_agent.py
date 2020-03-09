import numpy as np
import copy


class gridworld_lrtastar_agent:
    def __init__(self, args, env, controller):
        # Store all given arguments
        self.args, self.env = args, env
        self.controller = controller

        self.state_values = np.zeros((args.grid_size, args.grid_size))
        self._fill_state_values(env)

    def _fill_state_values(self, env):
        goal_state = env.goal_state
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                # Fill with manhattan distances
                self.state_values[i, j] = np.abs(
                    goal_state - np.array([i, j])).sum()

        return True

    def get_state_value(self, obs):
        current_state = obs['observation']
        return self.state_values[tuple(current_state)]

    def learn_online_in_real_world(self, max_timesteps=None):
        # Reset environment
        obs = self.env.reset()
        # Configure heuristic for controller
        self.controller.reconfigure_heuristic(
            self.get_state_value
        )

        total_n_steps = 0

        while True:
            current_state = obs['observation'].copy()
            ac, info = self.controller.act(obs)
            next_obs, cost, _, _ = self.env.step(ac)
            next_state = next_obs['observation'].copy()
            # Update model
            self.controller.model.update(current_state, ac, next_state)
            if self.args.verbose:
                print('t', total_n_steps)
                print('STATE', current_state)
                print('ACTION', ac)
                print('VALUE PREDICTED', info['start_node_h'])

            if self.env.check_goal(next_obs['observation'],
                                   next_obs['desired_goal']):
                print('REACHED GOAL!')
                break
            total_n_steps += 1

            # Do a simple RTDP update
            _, info = self.controller.act(obs)
            # update all states on the closed list
            for node in info['closed']:
                state = node.obs['observation']
                gval = node._g
                self.state_values[tuple(state)] = info['best_node_f'] - gval

            # Only update the current state
            # self.state_values[tuple(
            #    current_state)] = info['best_node_f']
            # Move to next iteration
            obs = copy.deepcopy(next_obs)

            if max_timesteps and total_n_steps >= max_timesteps:
                break

        return total_n_steps
