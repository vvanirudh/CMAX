import copy
import torch
import numpy as np

import odium.utils.logger as logger
from odium.utils.agent_utils.save_load_agent import save_agent, load_agent

from odium.agents.fetch_agents.sampler import rts_sampler
from odium.agents.fetch_agents.memory import rts_memory
from odium.agents.fetch_agents.approximators import StateActionValueResidual
from odium.agents.fetch_agents.normalizer import FeatureNormalizer


class fetch_dqn_agent:
    def __init__(self, args, env_params, env, controller):
        '''
        Initialization function

        Parameters
        ----------
        args: arguments
        env_params: environment parameters
        env: the actual environment
        controller: A controller to get the hardcoded heuristic from
          Only used for the heuristic value, we do not use the controller
        '''
        self.args, self.env_params, self.env = args, env_params, env
        self.controller = controller

        # Get the controller heuristic function
        self.controller_heuristic_fn = controller.heuristic_obs_g

        # Sampler
        # Initialize sampler
        self.sampler = rts_sampler(args,
                                   env.compute_reward,
                                   self.controller_heuristic_fn,
                                   env.extract_features)

        # Memory
        # Initialize buffers to store transitions
        self.memory = rts_memory(args,
                                 env_params,
                                 self.sampler)

        # Approximators
        # Initialize all relevant approximators
        self.state_action_value_residual = StateActionValueResidual(env_params)
        self.state_action_value_target_residual = StateActionValueResidual(
            env_params)
        self.state_action_value_target_residual.load_state_dict(
            self.state_action_value_target_residual.state_dict())

        # Optimizers
        # Initialize all optimizers
        self.state_action_value_residual_optim = torch.optim.Adam(
            self.state_action_value_residual.parameters(),
            lr=args.lr_value_residual,
            weight_decay=args.l2_reg_value_residual
        )

        # Normalizers
        # Initialize all normalizers
        self.features_normalizer = FeatureNormalizer(env_params)

        return

    '''
    ONLINE
    '''

    def online_rollout(self, initial_observation):
        n_steps = 0
        r_obs, r_ag, r_g, r_actions, r_heuristic = [], [], [], [], []
        r_reward, r_qpos, r_qvel, r_features = [], [], [], []
        r_qvalues, r_penetration = [], []
        observation = copy.deepcopy(initial_observation)
        obs = observation['observation']
        g = observation['desired_goal']
        ag = observation['achieved_goal']
        qpos = observation['sim_state'].qpos
        qvel = observation['sim_state'].qvel
        heuristic = self.controller_heuristic_fn(obs, g)
        features = self.env.extract_features(obs, g)
        qvalues = self.controller.get_all_qvalues(observation)
        for _ in range(self.env_params['max_timesteps']):
            features_norm = self.features_normalizer.normalize(features)
            with torch.no_grad():
                inputs = torch.as_tensor(
                    features_norm, dtype=torch.float32).view(1, -1)
                q_vector = self.state_action_value_residual(inputs).squeeze()
            # Get best action as predicted by state-action values (in this case,
            # advantages)
            # Since we are dealing with costs
            qvalues_tensor = torch.as_tensor(qvalues, dtype=torch.float32)
            ac_ind = torch.argmin(q_vector + qvalues_tensor).item()
            # Eps-greedy
            toss = np.random.random()
            if toss < self.args.dqn_epsilon:
                # Choose action randomly
                # TODO: This can choose the null action as well
                ac_ind = np.random.randint(self.env_params['num_actions'])
            # Get the corresponding action
            ac = self.env.discrete_actions_list[ac_ind]
            # Take a step
            observation_new, rew, _, info = self.env.step(np.array(ac))
            penetration = info['penetration']
            if self.args.render:
                self.env.render()
            # Check if we reached the goal
            if self.env.env._is_success(observation_new['achieved_goal'], g):
                return n_steps, observation_new, True
            # Increment counter
            n_steps += 1
            # Else, add it to data
            r_obs.append(obs.copy())
            r_ag.append(ag.copy())
            r_g.append(g.copy())
            r_actions.append(ac_ind)
            r_heuristic.append(heuristic)
            r_reward.append(rew)
            r_qpos.append(qpos.copy())
            r_qvel.append(qvel.copy())
            r_features.append(features.copy())
            r_qvalues.append(qvalues.copy())
            r_penetration.append(penetration)

            obs = observation_new['observation']
            ag = observation_new['achieved_goal']
            qpos = observation['sim_state'].qpos
            qvel = observation['sim_state'].qvel
            heuristic = self.controller_heuristic_fn(obs, g)
            features = self.env.extract_features(obs, g)
            qvalues = self.controller.get_all_qvalues(observation_new)

        r_obs.append(obs.copy())
        r_ag.append(ag.copy())
        r_heuristic.append(heuristic)
        r_features.append(features.copy())
        r_qvalues.append(qvalues.copy())

        r_obs = np.expand_dims(np.array(r_obs), 0)
        r_ag = np.expand_dims(np.array(r_ag), 0)
        r_g = np.expand_dims(np.array(r_g), 0)
        r_actions = np.expand_dims(np.array(r_actions), 0)
        r_heuristic = np.expand_dims(np.array(r_heuristic), 0)
        r_reward = np.expand_dims(np.array(r_reward), 0)
        r_qpos = np.expand_dims(np.array(r_qpos), 0)
        r_qvel = np.expand_dims(np.array(r_qvel), 0)
        r_features = np.expand_dims(np.array(r_features), 0)
        r_qvalues = np.expand_dims(np.array(r_qvalues), 0)
        r_penetration = np.expand_dims(np.array(r_penetration), 0)

        self.memory.store_internal_model_rollout(
            [r_obs, r_ag, r_g, r_actions, r_heuristic,
             r_reward, r_qpos, r_qvel, r_features, r_qvalues, r_penetration], qvalue=True)
        self.features_normalizer.update_normalizer(
            [r_obs, r_ag, r_g, r_actions, r_heuristic,
             r_reward, r_qpos, r_qvel, r_features, r_penetration], self.sampler)

        return n_steps, observation_new, False

    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _update_state_action_value_residual(self):
        transitions = self.memory.sample_internal_world_memory(
            self.args.batch_size, qvalue=True)

        qvalues, qvalues_next = transitions['qvalues'], transitions['qvalues_next']
        f, f_next = transitions['features'], transitions['features_next']
        f_norm, f_next_norm = self.features_normalizer.normalize(
            f), self.features_normalizer.normalize(f_next)
        inputs_norm_tensor = torch.as_tensor(f_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.as_tensor(
            f_next_norm, dtype=torch.float32)

        h = transitions['heuristic']
        r = transitions['r']
        c_tensor = -torch.as_tensor(r, dtype=torch.float32).view(-1, 1)
        h_tensor = torch.as_tensor(h, dtype=torch.float32).view(-1, 1)
        ac_tensor = torch.as_tensor(
            transitions['actions'], dtype=torch.long).view(-1, 1)

        qvalues_tensor = torch.as_tensor(qvalues, dtype=torch.float32)
        qvalues_next_tensor = torch.as_tensor(
            qvalues_next, dtype=torch.float32)

        with torch.no_grad():
            next_target_residual_tensor = self.state_action_value_target_residual(
                inputs_next_norm_tensor)
            '''
            Double Q-learning update
            '''
            next_residual_tensor = self.state_action_value_residual(
                inputs_next_norm_tensor)
            target_ac = torch.argmin(
                next_residual_tensor + qvalues_next_tensor, dim=1, keepdim=True)
            next_target_residual_tensor = next_target_residual_tensor.gather(
                1, target_ac)

            next_target_residual_tensor = next_target_residual_tensor.detach()
            target_state_action_value = c_tensor + \
                next_target_residual_tensor + \
                qvalues_next_tensor.gather(1, target_ac)
            # Clip target state action value so that it is not less than zero
            target_residual_tensor = torch.max(
                target_state_action_value - qvalues_tensor.gather(1, ac_tensor), -qvalues_tensor.gather(1, ac_tensor))
            # Clip target state action vlaue so that it is not greater than horizon
            if self.args.offline:
                # Only offline, we clip the Q value from above
                target_residual_tensor = torch.min(
                    target_residual_tensor, self.env_params['offline_max_timesteps'] - h_tensor)

        # Compute predictions
        residual_tensor = self.state_action_value_residual(inputs_norm_tensor)
        residual_tensor = residual_tensor.gather(1, ac_tensor.view(-1, 1))
        # Compute loss
        residual_loss = (residual_tensor -
                         target_residual_tensor).pow(2).mean()
        # Take a GD step
        self.state_action_value_residual_optim.zero_grad()
        residual_loss.backward()
        self.state_action_value_residual_optim.step()

        return residual_loss.item()

    def save(self, epoch, success_rate):
        return save_agent(path=self.args.save_dir+'/fetch_dqn_agent.pth',
                          network_state_dict=self.state_action_value_residual.state_dict(),
                          optimizer_state_dict=self.state_action_value_residual_optim.state_dict(),
                          normalizer_state_dict=self.features_normalizer.state_dict(),
                          epoch=epoch,
                          success_rate=success_rate)

    def load(self):
        load_dict, load_dict_keys = load_agent(
            self.args.load_dir+'/fetch_dqn_agent.pth')
        self.state_action_value_residual.load_state_dict(
            load_dict['network_state_dict'])
        self.state_action_value_target_residual.load_state_dict(
            load_dict['network_state_dict'])
        if 'optimizer_state_dict' in load_dict_keys:
            self.state_action_value_residual_optim.load_state_dict(
                load_dict['optimizer_state_dict'])
        if 'normalizer_state_dict' in load_dict_keys:
            self.features_normalizer.load_state_dict(
                load_dict['normalizer_state_dict'])
        return

    def learn_online_in_real_world(self, max_timesteps=None):
        # If any pre-existing model is given, load it
        if self.args.load_dir:
            self.load()
        # Reset the environment
        observation = self.env.reset()

        # Count of total number of steps
        total_n_steps = 0
        while True:
            # Rollout for a few steps to collect some transitions
            n_steps, final_observation, done = self.online_rollout(observation)
            # Update the counter
            total_n_steps += n_steps
            # Check if we have reached the goal
            if done:
                break
            # Batch updates
            losses = []
            # for _ in range(self.args.planning_rollout_length):
            for _ in range(self.args.n_online_planning_updates):
                # Update the state-action value residual
                loss = self._update_state_action_value_residual()
                losses.append(loss)
                # Update the target network
                self._update_target_network(self.state_action_value_target_residual,
                                            self.state_action_value_residual)
            # Log
            logger.record_tabular('n_steps', total_n_steps)
            logger.record_tabular('residual_loss', np.mean(losses))
            # logger.dump_tabular()
            # Move to next iteration
            observation = copy.deepcopy(final_observation)

            if max_timesteps and total_n_steps >= max_timesteps:
                break

        return total_n_steps
