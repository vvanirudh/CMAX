import numpy as np
import copy


class gridworld_qlearning_agent:
    def __init__(self, args, env, controller):
        self.args, self.env = args, env
        self.controller = controller

        self.num_actions = self.env.get_actions().shape[0]

        self.initial_state_values = controller.get_initial_state_values(
            self.env.goal_state)
        self.qvalues = np.tile(
            self.initial_state_values, (self.num_actions, 1, 1))
        self.epsilon = args.epsilon

    def get_qvalues(self, obs):
        current_state = obs['observation']
        return self.qvalues[:, current_state[0], current_state[1]]

    def learn_online_in_real_world(self, max_timesteps=None):
        # Reset environmetn
        obs = self.env.reset()
        # Configure qvalue fn for controller
        self.controller.reconfigure_qvalue_fn(self.get_qvalues)

        total_n_steps = 0
        while True:
            current_state = obs['observation'].copy()
            ac = self.controller.act(obs)
            toss = np.random.rand()
            if toss < self.epsilon:
                # Choose a random action
                ac = self.env.action_space.sample()

            next_obs, cost, _, _ = self.env.step(ac)

            if self.args.verbose:
                print('t', total_n_steps)
                print('STATE', current_state)
                print('ACTION', ac)

            if self.env.check_goal(next_obs['observation'],
                                   next_obs['desired_goal']):
                print('REACHED GOAL!')
                break
            total_n_steps += 1

            # Do a Q learning update
            self.qvalues[ac, current_state[0], current_state[1]
                         ] = cost + np.min(self.get_qvalues(next_obs))
            # Move to next iteration
            obs = copy.deepcopy(next_obs)

            if max_timesteps and total_n_steps >= max_timesteps:
                break

        return total_n_steps
