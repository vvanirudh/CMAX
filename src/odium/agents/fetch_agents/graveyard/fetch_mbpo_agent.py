import warnings
import torch
import numpy as np
import copy

from odium.agents.fetch_agents.normalizer import FeatureNormalizer
from odium.agents.fetch_agents.approximators import StateValueResidual, get_state_value_residual, Dynamics, get_next_observation
from odium.agents.fetch_agents.memory import rts_memory
from odium.agents.fetch_agents.sampler import rts_sampler

import odium.utils.logger as logger
from odium.utils.simple_utils import multi_append, convert_to_list_of_np_arrays
from odium.utils.agent_utils.save_load_agent import save_agent, load_agent


class fetch_mbpo_agent:
    def __init__(self, args, env_params, env, controller):
        # Save arguments
        self.args, self.env_params = args, env_params
        self.env, self.controller = env, controller

        # Sampler
        self.sampler = rts_sampler(args,
                                   env.compute_reward,
                                   controller.heuristic_obs_g,
                                   env.extract_features)

        # Memory
        self.memory = rts_memory(args,
                                 env_params,
                                 self.sampler)

        # Approximators
        self.state_value_residual = StateValueResidual(env_params)
        self.state_value_target_residual = StateValueResidual(env_params)
        self.state_value_target_residual.load_state_dict(
            self.state_value_residual.state_dict())

        self.learned_model_dynamics = Dynamics(env_params)
        # Configure controller dynamics residual
        self.controller.reconfigure_learned_model_dynamics(lambda obs, ac: get_next_observation(
            obs,
            ac,
            self.preproc_dynamics_inputs,
            self.learned_model_dynamics))

        # Optimizers
        # Initialize all optimizers
        # STATE VALUE RESIDUAL
        self.state_value_residual_optim = torch.optim.Adam(
            self.state_value_residual.parameters(),
            lr=args.lr_value_residual,
            weight_decay=args.l2_reg_value_residual)
        # DYNAMICS
        self.learned_model_dynamics_optim = torch.optim.Adam(
            self.learned_model_dynamics.parameters(),
            lr=args.lr_dynamics,
            weight_decay=args.l2_reg_dynamics)

        # Normalizers
        # Initialize all normalizers
        # FEATURES
        self.features_normalizer = FeatureNormalizer(env_params)

    def collect_internal_model_trajectories(self,
                                            num_rollouts,
                                            rollout_length,
                                            initial_observations=None):
        n_steps = 0
        mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic = [], [], [], [], []
        mb_reward, mb_features = [], []
        # Start rollouts
        for n in range(num_rollouts):
            # Set initial state
            if initial_observations is not None:
                observation = copy.deepcopy(initial_observations[n])
            else:
                observation = self.env.get_obs()
            # Data structures
            r_obs, r_ag, r_g, r_actions, r_heuristic = [], [], [], [], []
            r_reward, r_features = [], []
            # Start
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            features = self.env.extract_features(obs, g)
            heuristic = self.controller.heuristic_obs_g(obs, g)
            for _ in range(rollout_length):
                ac, _ = self.controller.act(observation)
                ac_ind = self.env.discrete_actions[tuple(ac)]
                # Get the next observation and reward using the learned model
                observation_new = get_next_observation(
                    observation,
                    ac,
                    self.preproc_dynamics_inputs,
                    self.learned_model_dynamics)
                rew = self.env.compute_reward(observation_new['achieved_goal'],
                                              observation_new['desired_goal'], {})
                n_steps += 1
                # Add to data structures
                multi_append([r_obs, r_ag, r_g, r_actions, r_heuristic, r_reward, r_features],
                             [obs.copy(), ag.copy(), g.copy(), ac_ind, heuristic, rew,
                              features.copy()])
                # Move to next step
                observation = copy.deepcopy(observation_new)
                obs = observation['observation']
                ag = observation['achieved_goal']
                g = observation['desired_goal']
                features = self.env.extract_features(obs, g)
                heuristic = self.controller.heuristic_obs_g(obs, g)
            multi_append([r_obs, r_ag, r_heuristic, r_features],
                         [obs.copy(), ag.copy(), heuristic, features.copy()])
            multi_append([mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
                          mb_reward, mb_features],
                         [r_obs, r_ag, r_g, r_actions, r_heuristic,
                          r_reward, r_features])

        mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic, mb_reward, mb_features = convert_to_list_of_np_arrays(
            [mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
             mb_reward, mb_features]
        )
        # Store in memory
        self.memory.store_internal_model_rollout([mb_obs, mb_ag, mb_g,
                                                  mb_actions, mb_heuristic, mb_reward,
                                                  mb_features], sim=False)
        # Update normalizer
        self._update_normalizer([mb_obs, mb_ag, mb_g,
                                 mb_actions, mb_heuristic, mb_reward,
                                 mb_features])

        return n_steps

    def _update_normalizer(self, batch):
        obs, ag, g, actions, heuristic, r, features = batch
        obs_next = obs[:, 1:, :]
        ag_next = ag[:, 1:, :]
        heuristic_next = heuristic[:, 1:]
        num_transitions = actions.shape[1]
        buffer_temp = {'obs': obs, 'ag': ag, 'g': g, 'actions': actions, 'heuristic': heuristic, 'r': r,
                       'features': features, 'obs_next': obs_next, 'ag_next': ag_next, 'heuristic_next': heuristic_next}
        transitions = self.sampler.sample(buffer_temp, num_transitions)
        self.features_normalizer.update(transitions['features'])
        self.features_normalizer.recompute_stats()
        return True

    def learn_offline_in_model(self):
        if not self.args.offline:
            warnings.warn('SHOULD NOT BE USED ONLINE')

        best_success_rate = 0.0
        n_steps = 0
        for epoch in range(self.args.n_epochs):
            # Reset the environment
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['offline_max_timesteps']):
                # Get action
                ac, info = self.controller.act(observation)
                ac_ind = self.env.discrete_actions[tuple(ac)]
                # Get the next observation and reward from the environment
                observation_new, rew, _, _ = self.env.step(ac)
                n_steps += 1
                obs_new = observation_new['observation']
                # Store the transition in memory
                self.memory.store_real_world_transition(
                    [obs, g, ac_ind, obs_new], sim=False)
                observation = copy.deepcopy(observation_new)
                obs = obs_new.copy()

            # Update state value residual from model rollouts
            transitions = self.memory.sample_real_world_memory(
                batch_size=self.args.n_cycles)
            losses = []
            model_losses = []
            for i in range(self.args.n_cycles):
                observation = {}
                observation['observation'] = transitions['obs'][i].copy()
                observation['achieved_goal'] = transitions['obs'][i][:3].copy()
                observation['desired_goal'] = transitions['g'][i].copy()
                # Collect model rollouts

                self.collect_internal_model_trajectories(num_rollouts=1,
                                                         rollout_length=self.env_params[
                                                             'offline_max_timesteps'],
                                                         initial_observations=[observation])
                # Update state value residuals
                for _ in range(self.args.n_batches):
                    state_value_residual_loss = self._update_state_value_residual().item()
                    losses.append(state_value_residual_loss)
                self._update_target_network(self.state_value_target_residual,
                                            self.state_value_residual)

                # Update dynamics model
                for _ in range(self.args.n_batches):
                    loss = self._update_learned_dynamics_model().item()
                    model_losses.append(loss)

            # Evaluate agent in the model
            mean_success_rate, mean_return = self.eval_agent_in_model()
            # Check if this is a better residual
            if mean_success_rate > best_success_rate:
                best_success_rate = mean_success_rate
                print('Best success rate so far', best_success_rate)
                if self.args.save_dir is not None:
                    print('Saving residual')
                    self.save(epoch, best_success_rate)

            # log
            logger.record_tabular('epoch', epoch)
            logger.record_tabular('n_steps', n_steps)
            logger.record_tabular('success_rate', mean_success_rate)
            logger.record_tabular('return', mean_return)
            logger.record_tabular(
                'state_value_residual_loss', np.mean(losses))
            logger.record_tabular('dynamics_loss', np.mean(model_losses))
            logger.dump_tabular()

    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _update_state_value_residual(self):
        transitions = self.memory.sample_internal_world_memory(
            self.args.batch_size)

        obs, g, ag = transitions['obs'], transitions['g'], transitions['ag']
        features, heuristic = transitions['features'], transitions['heuristic']
        targets = []

        for i in range(self.args.batch_size):
            observation = {}
            observation['observation'] = obs[i].copy()
            observation['desired_goal'] = g[i].copy()
            observation['achieved_goal'] = ag[i].copy()

            _, info = self.controller.act(observation)
            targets.append(info['best_node_f'])
        targets = np.array(targets).reshape(-1, 1)
        features_norm = self.features_normalizer.normalize(features)

        inputs_norm = torch.as_tensor(features_norm, dtype=torch.float32)
        targets = torch.as_tensor(targets, dtype=torch.float32)

        h_tensor = torch.as_tensor(
            heuristic, dtype=torch.float32).unsqueeze(-1)
        # Compute target residuals
        target_residual_tensor = targets - h_tensor
        # Clip target residual tenssor to avoid value function less than zero
        target_residual_tensor = torch.max(
            target_residual_tensor, -h_tensor)
        # Clip target residual tensor to avoid value function greater than horizon
        if self.args.offline:
            target_residual_tensor = torch.min(target_residual_tensor,
                                               self.env_params['offline_max_timesteps'] - h_tensor)

        # COmpute predictions
        residual_tensor = self.state_value_residual(inputs_norm)
        # COmpute loss
        state_value_residual_loss = (
            residual_tensor - target_residual_tensor).pow(2).mean()

        # Backprop and step
        self.state_value_residual_optim.zero_grad()
        state_value_residual_loss.backward()
        self.state_value_residual_optim.step()

        # Configure heuristic for controller
        self.controller.reconfigure_heuristic(
            lambda obs: get_state_value_residual(obs,
                                                 self.preproc_inputs,
                                                 self.state_value_residual))

        return state_value_residual_loss

    def _update_learned_dynamics_model(self):
        transitions = self.memory.sample_real_world_memory(
            self.args.batch_size)
        obs, ac_ind, obs_next = transitions['obs'], transitions['actions'], transitions['obs_next']
        gripper_pos = obs[:, :2]
        obj_pos = obs[:, 3:5]
        s = np.concatenate([gripper_pos, obj_pos], axis=1)
        s_tensor = torch.as_tensor(s, dtype=torch.float32)
        a_tensor = torch.as_tensor(ac_ind, dtype=torch.long)

        # Get predicted next state
        predicted_s_next_tensor = self.learned_model_dynamics(
            s_tensor, a_tensor)

        # Get true next state
        gripper_pos_next = obs_next[:, :2]
        obj_pos_next = obs_next[:, 3:5]
        s_next = np.concatenate([gripper_pos_next, obj_pos_next], axis=1)
        s_next_tensor = torch.as_tensor(s_next, dtype=torch.float32)

        # Compute MSE loss
        loss = (predicted_s_next_tensor - s_next_tensor).pow(2).mean()
        # Backprop and step
        self.learned_model_dynamics_optim.zero_grad()
        loss.backward()
        self.learned_model_dynamics_optim.step()

        # Configure new dynamics model for controller
        self.controller.reconfigure_learned_model_dynamics(
            lambda observation, ac: get_next_observation(
                observation,
                ac,
                self.preproc_dynamics_inputs,
                self.learned_model_dynamics)
        )

        return loss

    def preproc_inputs(self, obs, g):
        '''
        Function to preprocess inputs
        '''
        features = self.env.extract_features(obs, g)
        features_norm = self.features_normalizer.normalize(features)
        inputs = torch.as_tensor(
            features_norm, dtype=torch.float32).unsqueeze(0)
        return inputs

    def preproc_dynamics_inputs(self, obs, ac):
        gripper_pos = obs[:2]
        obj_pos = obs[3:5]
        s = np.concatenate([gripper_pos, obj_pos])
        ac_ind = self.env.discrete_actions[tuple(ac)]

        s_tensor = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        a_tensor = torch.as_tensor(ac_ind, dtype=torch.long).unsqueeze(0)

        return s_tensor, a_tensor

    def save(self, epoch, success_rate):
        return save_agent(path=self.args.save_dir+'/fetch_mbpo_agent.pth',
                          network_state_dict=self.state_value_residual.state_dict(),
                          optimizer_state_dict=self.state_value_residual_optim.state_dict(),
                          normalizer_state_dict=self.features_normalizer.state_dict(),
                          dynamics_state_dict=self.learned_model_dynamics.state_dict(),
                          dynamics_optimizer_state_dict=self.learned_model_dynamics_optim.state_dict(),
                          epoch=epoch,
                          success_rate=success_rate)

    def load(self):
        load_dict, load_dict_keys = load_agent(
            self.args.load_dir+'/fetch_mbpo_agent.pth')
        self.state_value_residual.load_state_dict(
            load_dict['network_state_dict'])
        self.state_value_target_residual.load_state_dict(
            load_dict['network_state_dict'])
        if 'optimizer_state_dict' in load_dict_keys:
            self.state_value_residual_optim.load_state_dict(
                load_dict['optimizer_state_dict'])
        if 'normalizer_state_dict' in load_dict_keys:
            self.features_normalizer.load_state_dict(
                load_dict['normalizer_state_dict'])
        if 'dynamics_state_dict' in load_dict_keys:
            self.learned_model_dynamics.load_state_dict(
                load_dict['dynamics_state_dict'])
        if 'dynamics_optimizer_state_dict' in load_dict_keys:
            self.learned_model_dynamics_optim.load_state_dict(
                load_dict['dynamics_optimizer_state_dict'])
        return

    def eval_agent_in_model(self):
        total_success_rate, total_return = [], []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate, per_return = [], 0
            observation = self.env.reset()
            for _ in range(self.env_params['offline_max_timesteps']):
                ac, _ = self.controller.act(observation)
                observation, rew, _, info = self.env.step(np.array(ac))
                per_success_rate.append(info['is_success'])
                per_return += rew

            total_success_rate.append(per_success_rate)
            total_return.append(per_return)

        total_success_rate = np.array(total_success_rate)
        mean_success_rate = np.mean(total_success_rate[:, -1])
        mean_return = np.mean(total_return)
        return mean_success_rate, mean_return
