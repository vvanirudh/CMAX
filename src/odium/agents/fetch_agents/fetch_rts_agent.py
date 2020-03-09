import copy
import pickle
import numpy as np
import ray
import torch

from sklearn.neighbors import KDTree

from odium.utils.simple_utils import convert_to_list_of_np_arrays, multi_append
from odium.utils.agent_utils.save_load_agent import save_agent, load_agent
from odium.utils.simulation_utils import set_sim_state_and_goal, apply_dynamics_residual

import odium.utils.logger as logger

from odium.agents.fetch_agents.sampler import rts_sampler
from odium.agents.fetch_agents.memory import rts_memory
from odium.agents.fetch_agents.approximators import StateValueResidual, get_state_value_residual, DynamicsResidual, get_next_observation, KNNDynamicsResidual, GPDynamicsResidual
from odium.agents.fetch_agents.normalizer import FeatureNormalizer
from odium.agents.fetch_agents.worker import InternalRolloutWorker, set_workers_num_expansions
from odium.agents.fetch_agents.discrepancy_utils import get_discrepancy_neighbors, apply_discrepancy_penalty


class fetch_rts_agent:
    def __init__(self, args, env_params, env, planning_env, controller):
        '''
        args - arguments
        env_params - environment parameters
        env - real world
        planning_env - internal model
        controller - planner
        '''
        # Store all given arguments
        self.args, self.env_params, self.env = args, env_params, env
        self.planning_env, self.controller = planning_env, controller

        # Sampler
        # Initialize sampler to sample from buffers
        self.sampler = rts_sampler(args,
                                   planning_env.compute_reward,
                                   controller.heuristic_obs_g,
                                   planning_env.extract_features)

        # Memory
        # Initialize memory/buffers to store transition
        self.memory = rts_memory(args,
                                 env_params,
                                 self.sampler)

        # Approximators
        # Initialize all relevant approximators
        # STATE VALUE RESIDUAL
        self.state_value_residual = StateValueResidual(env_params)
        self.state_value_target_residual = StateValueResidual(env_params)
        self.state_value_target_residual.load_state_dict(
            self.state_value_residual.state_dict())

        # Dynamics model
        if self.args.agent == 'mbpo':
            self.residual_dynamics = DynamicsResidual(env_params)
        elif self.args.agent == 'mbpo_knn':
            self.residual_dynamics = [KNNDynamicsResidual(
                args, env_params) for _ in range(self.env_params['num_actions'])]
        else:
            self.residual_dynamics = [GPDynamicsResidual(
                args, env_params) for _ in range(self.env_params['num_actions'])]

        # KD Tree models
        self.kdtrees = [None for _ in range(self.env_params['num_actions'])]

        # Optimizers
        # Initialize all optimizers
        # STATE VALUE RESIDUAL
        self.state_value_residual_optim = torch.optim.Adam(
            self.state_value_residual.parameters(),
            lr=args.lr_value_residual,
            weight_decay=args.l2_reg_value_residual)
        # LEARNED DYNAMICS
        if self.args.agent == 'mbpo':
            self.residual_dynamics_optim = torch.optim.Adam(
                self.residual_dynamics.parameters(),
                lr=args.lr_dynamics,
                weight_decay=args.l2_reg_dynamics)

        # Normalizers
        # Initialize all normalizers
        # FEATURES
        self.features_normalizer = FeatureNormalizer(env_params)

        # Workers
        # Initialize all workers
        self.internal_rollout_workers = [InternalRolloutWorker.remote(args,
                                                                      env_params,
                                                                      worker_id=i)
                                         for i in range(args.n_rts_workers)]

        # Trackers
        self.n_planning_steps = 0
        return

    '''
    ONLINE
    '''

    def save(self, epoch, success_rate):
        return save_agent(path=self.args.save_dir+'/fetch_rts_agent.pth',
                          network_state_dict=self.state_value_residual.state_dict(),
                          optimizer_state_dict=self.state_value_residual_optim.state_dict(),
                          normalizer_state_dict=self.features_normalizer.state_dict(),
                          epoch=epoch,
                          success_rate=success_rate)

    def load(self):
        load_dict, load_dict_keys = load_agent(
            self.args.load_dir+'/fetch_rts_agent.pth')
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
        if 'knn_dynamics_residuals_serialized' in load_dict_keys:
            self.knn_dynamics_residuals = pickle.loads(
                load_dict['knn_dynamics_residuals_serialized'])
        return

    def _update_target_network(self, target, source):
        # Simply copy parameter values
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _update_batch_residual_dynamics(self):
        # Sample transitions
        if self.args.offline:
            # If offline
            transitions = self.memory.sample_internal_world_memory(
                self.args.batch_size)
            obs, ac_ind, obs_next = transitions['obs'], transitions['actions'], transitions['obs_next']
            raise Exception('Online mode is required')
        else:
            # If online
            transitions = self.memory.sample_real_world_memory(
                self.args.batch_size)
            obs, ac_ind, obs_next = transitions['obs'], transitions['actions'], transitions['obs_next']
            obs_sim_next = transitions['obs_sim_next']

        gripper_pos = obs[:, :2]
        obj_pos = obs[:, 3:5]
        s = np.concatenate([gripper_pos, obj_pos], axis=1)
        s_tensor = torch.as_tensor(s, dtype=torch.float32)
        a_tensor = torch.as_tensor(ac_ind, dtype=torch.long)

        # Get predicted next state
        predicted_s_next_tensor = self.residual_dynamics(
            s_tensor, a_tensor)

        # Get true next state
        gripper_pos_next = obs_next[:, :2]
        obj_pos_next = obs_next[:, 3:5]
        s_next = np.concatenate([gripper_pos_next, obj_pos_next], axis=1)
        s_next_tensor = torch.as_tensor(s_next, dtype=torch.float32)

        # Get sim next state
        gripper_pos_sim_next = obs_sim_next[:, :2]
        obj_pos_sim_next = obs_sim_next[:, 3:5]
        s_sim_next = np.concatenate(
            [gripper_pos_sim_next, obj_pos_sim_next], axis=1)
        s_sim_next_tensor = torch.as_tensor(s_sim_next, dtype=torch.float32)

        # Compute target
        target_tensor = s_next_tensor - s_sim_next_tensor

        # Compute MSE loss
        loss = (predicted_s_next_tensor - target_tensor).pow(2).mean()
        # Backprop and step
        self.residual_dynamics_optim.zero_grad()
        loss.backward()
        self.residual_dynamics_optim.step()

        assert self.args.agent == 'mbpo'
        self.controller.reconfigure_residual_dynamics(
            self.get_residual_dynamics)

        return loss

    def _update_residual_dynamics(self):
        # Sample all real world transitions
        transitions = self.memory.sample_real_world_memory()
        obs, ac_ind, obs_next = transitions['obs'], transitions['actions'], transitions['obs_next']
        obs_sim_next = transitions['obs_sim_next']

        gripper_pos = obs[:, :2]
        obj_pos = obs[:, 3:5]
        s = np.concatenate([gripper_pos, obj_pos], axis=1)

        # Get true next state
        gripper_pos_next = obs_next[:, :2]
        obj_pos_next = obs_next[:, 3:5]
        s_next = np.concatenate([gripper_pos_next, obj_pos_next], axis=1)

        # Get sim next state
        gripper_pos_sim_next = obs_sim_next[:, :2]
        obj_pos_sim_next = obs_sim_next[:, 3:5]
        s_sim_next = np.concatenate(
            [gripper_pos_sim_next, obj_pos_sim_next], axis=1)

        # Compute target
        target = s_next - s_sim_next

        loss = 0
        for i in range(self.env_params['num_actions']):
            ac_mask = ac_ind == i
            s_mask = s[ac_mask]
            target_mask = target[ac_mask]

            if s_mask.shape[0] == 0:
                # No data points for this action
                continue
            if self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp':
                # Fit the KNN/GP model
                loss += self.residual_dynamics[i].fit(s_mask, target_mask)
            else:
                raise Exception(
                    'This agent does not support residual dynamics')

        assert self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp'
        self.controller.reconfigure_residual_dynamics(
            self.get_residual_dynamics)

        return loss

    def _update_state_value_residual(self):
        # Sample transitions
        transitions = self.memory.sample_internal_world_memory(
            self.args.batch_size)
        qpos, qvel = transitions['qpos'], transitions['qvel']
        obs, g, ag = transitions['obs'], transitions['g'], transitions['ag']
        # features, heuristic = transitions['features'], transitions['heuristic']

        # Compute target by restarting search from the sampled states
        num_workers = self.args.n_rts_workers
        if self.args.batch_size < self.args.n_rts_workers:
            num_workers = self.args.batch_size
        num_per_worker = self.args.batch_size // num_workers
        # Put residual in object store
        value_target_residual_state_dict_id = ray.put(
            self.state_value_target_residual.state_dict())
        # Put normalizer in object store
        feature_norm_dict_id = ray.put(self.features_normalizer.state_dict())
        # Put knn dynamics residuals in object store
        if self.args.agent == 'rts':
            kdtrees_serialized_id = ray.put(pickle.dumps(
                self.kdtrees))
        else:
            kdtrees_serialized_id = None
        # Put residual dynamics in object store
        if self.args.agent == 'mbpo':
            residual_dynamics_state_dict_id = ray.put(
                self.residual_dynamics.state_dict())
        elif self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp':
            residual_dynamics_state_dict_id = ray.put(
                pickle.dumps(self.residual_dynamics))
        else:
            residual_dynamics_state_dict_id = None
        results, count = [], 0
        # Set all workers num expansions
        set_workers_num_expansions(self.internal_rollout_workers,
                                   self.args.n_offline_expansions)
        for worker_id in range(num_workers):
            if worker_id == num_workers - 1:
                # last worker takes the remaining load
                num_per_worker = self.args.batch_size - count
            # Set parameters
            ray.get(self.internal_rollout_workers[worker_id].set_worker_params.remote(
                value_residual_state_dict=value_target_residual_state_dict_id,
                feature_norm_dict=feature_norm_dict_id,
                kdtrees_serialized=kdtrees_serialized_id,
                residual_dynamics_state_dict=residual_dynamics_state_dict_id))
            # Send Job
            results.append(
                self.internal_rollout_workers[worker_id].lookahead_batch.remote(
                    obs[count:count+num_per_worker],
                    g[count:count+num_per_worker],
                    ag[count:count+num_per_worker],
                    qpos[count:count+num_per_worker],
                    qvel[count:count+num_per_worker]))
            count += num_per_worker
        # Check if all transitions have targets
        assert count == self.args.batch_size
        # Get all targets
        results = ray.get(results)
        target_infos = [item for sublist in results for item in sublist]

        # Extract the states, their features and their corresponding targets
        obs_closed = [k.obs['observation'].copy()
                      for info in target_infos for k in info['closed']]
        goals_closed = [k.obs['desired_goal'].copy()
                        for info in target_infos for k in info['closed']]
        heuristic_closed = [self.controller.heuristic_obs_g(
            obs_closed[i], goals_closed[i]) for i in range(len(obs_closed))]
        features_closed = [self.env.extract_features(
            obs_closed[i], goals_closed[i]) for i in range(len(obs_closed))]
        targets_closed = [info['best_node_f'] -
                          k._g for info in target_infos for k in info['closed']]

        targets_closed = np.array(targets_closed).reshape(-1, 1)
        targets_tensor = torch.as_tensor(targets_closed, dtype=torch.float32)
        # Set all workers num expansions
        set_workers_num_expansions(self.internal_rollout_workers,
                                   self.args.n_expansions)
        # Normalize features
        inputs_norm = torch.as_tensor(self.features_normalizer.normalize(features_closed),
                                      dtype=torch.float32)
        heuristic_tensor = torch.as_tensor(
            heuristic_closed, dtype=torch.float32).view(-1, 1)

        # Compute target residuals
        target_residual_tensor = targets_tensor - heuristic_tensor
        # Clip target residual tenssor to avoid value function less than zero
        target_residual_tensor = torch.max(
            target_residual_tensor, -heuristic_tensor)
        # Clip target residual tensor to avoid value function greater than horizon
        if self.args.offline:
            target_residual_tensor = torch.min(target_residual_tensor,
                                               self.env_params['offline_max_timesteps'] - heuristic_tensor)

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

    def _update_discrepancy_model(self):
        # For now updating the KDTrees in batches, which is not really efficient
        # Future TODO: is to make it incremental and efficient
        # Get all transitions with discrepancy in dynamics
        transitions = self.memory.sample_real_world_memory()
        # Extract relevant quantities
        obs, ac_ind = transitions['obs'], transitions['actions']

        # Construct 4D points
        # obs[0:2] is gripper 2D position
        # obs[3:5] is object 2D position
        real_pos = np.concatenate([obs[:, 0:2], obs[:, 3:5]], axis=1)

        # Add it to the respective KDTrees
        for i in range(self.env_params['num_actions']):
            # Get points corresponding to this action
            ac_mask = ac_ind == i
            points = real_pos[ac_mask]

            if points.shape[0] == 0:
                # No data points for this action
                continue

            # Fit the KDTree
            self.kdtrees[i] = KDTree(points)

        # Configure discrepancy model for controller
        assert self.args.agent == 'rts'
        self.controller.reconfigure_discrepancy(
            lambda obs, ac: get_discrepancy_neighbors(obs,
                                                      ac,
                                                      self.construct_4d_point,
                                                      self.kdtrees,
                                                      self.args.neighbor_radius)
        )

        return

    def _check_dynamics_transition(self, transition):
        obs, _, _, _, _, obs_next, obs_sim_next = transition
        real_next_pos = np.concatenate([obs_next[0:2], obs_next[3:5]])
        sim_next_pos = np.concatenate([obs_sim_next[0:2], obs_sim_next[3:5]])
        residual = real_next_pos - sim_next_pos
        if np.linalg.norm(residual) < self.args.dynamic_residual_threshold:
            return False
        return True

    def preproc_inputs(self, obs, g):
        '''
        Function to preprocess inputs
        '''
        features = self.env.extract_features(obs, g)
        features_norm = self.features_normalizer.normalize(features)
        inputs = torch.as_tensor(
            features_norm, dtype=torch.float32).view(1, -1)
        return inputs

    def construct_4d_point(self, obs, ac):
        # Concatenate 2D gripper pos and 2D object pos
        pos = np.concatenate([obs[0:2], obs[3:5]]).reshape(1, -1)
        ac_ind = self.env.discrete_actions[tuple(ac)]
        return pos, ac_ind

    def online_rollout(self, initial_observation):
        n_steps = 0
        mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic = [], [], [], [], []
        mb_reward, mb_qpos, mb_qvel, mb_features = [], [], [], []
        mb_penetration = []
        # Set initial state
        observation = copy.deepcopy(initial_observation)
        # Data structures
        r_obs, r_ag, r_g, r_actions, r_heuristic = [], [], [], [], []
        r_reward, r_qpos, r_qvel, r_features = [], [], [], []
        r_penetration = []
        # Start
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        qpos = observation['sim_state'].qpos
        qvel = observation['sim_state'].qvel
        set_sim_state_and_goal(
            self.planning_env,
            qpos.copy(),
            qvel.copy(),
            g.copy())
        features = self.env.extract_features(obs, g)
        heuristic = self.controller.heuristic_obs_g(obs, g)
        for _ in range(self.env_params['max_timesteps']):
            ac, _ = self.controller.act(observation)
            ac_ind = self.env.discrete_actions[tuple(ac)]
            next_observation, rew, _, info = self.planning_env.step(ac)
            penetration = info['penetration']
            if self.args.agent == 'rts':
                rew = apply_discrepancy_penalty(
                    observation, ac, rew, self.controller.discrepancy_fn)
            if self.args.agent == 'mbpo' or self.args.agent == 'mbpo_knn':
                next_observation, rew = apply_dynamics_residual(self.planning_env,
                                                                self.get_residual_dynamics,
                                                                observation,
                                                                info,
                                                                ac,
                                                                next_observation)
            n_steps += 1
            # Add to data structures
            multi_append([r_obs, r_ag, r_g, r_actions, r_heuristic, r_reward, r_qpos, r_qvel, r_features, r_penetration],
                         [obs.copy(), ag.copy(), g.copy(), ac_ind, heuristic,
                          rew, qpos.copy(), qvel.copy(), features.copy(), penetration])
            # Move to next step
            observation = copy.deepcopy(next_observation)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            qpos = observation['sim_state'].qpos
            qvel = observation['sim_state'].qvel
            features = self.env.extract_features(obs, g)
            heuristic = self.controller.heuristic_obs_g(obs, g)
        multi_append([r_obs, r_ag, r_heuristic, r_features],
                     [obs.copy(), ag.copy(), heuristic, features.copy()])
        multi_append([mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
                      mb_reward, mb_qpos, mb_qvel, mb_features, mb_penetration],
                     [r_obs, r_ag, r_g, r_actions, r_heuristic,
                      r_reward, r_qpos, r_qvel, r_features, r_penetration])

        mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic, mb_reward, mb_qpos, mb_qvel, mb_features, mb_penetration = convert_to_list_of_np_arrays(
            [mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
             mb_reward, mb_qpos, mb_qvel, mb_features, mb_penetration]
        )

        # Store in memory
        self.memory.store_internal_model_rollout([mb_obs, mb_ag, mb_g,
                                                  mb_actions, mb_heuristic, mb_reward,
                                                  mb_qpos, mb_qvel, mb_features, mb_penetration])
        # Update normalizer
        self.features_normalizer.update_normalizer([mb_obs, mb_ag, mb_g,
                                                    mb_actions, mb_heuristic, mb_reward,
                                                    mb_qpos, mb_qvel, mb_features, mb_penetration], self.sampler)

        return n_steps

    def plan_online_in_model(self, n_planning_updates, initial_observation):
        # Clear memory
        # self.memory.clear(internal=True, real=False)
        n_updates = 0
        losses = []
        while n_updates < n_planning_updates:
            n_updates += 1
            self.online_rollout(initial_observation)
            state_value_residual_loss = self._update_state_value_residual()
            losses.append(state_value_residual_loss.item())
            self._update_target_network(self.state_value_target_residual,
                                        self.state_value_residual)

        return np.mean(losses)

    def get_residual_dynamics(self, obs, ac):
        if self.args.agent == 'mbpo':
            return get_next_observation(obs, ac,
                                        self.preproc_dynamics_inputs, self.residual_dynamics)
        elif self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp':
            gripper_pos = obs['observation'][:2]
            obj_pos = obs['observation'][3:5]
            s = np.concatenate([gripper_pos, obj_pos])
            ac_ind = self.env.discrete_actions[tuple(ac)]
            residual = self.residual_dynamics[ac_ind].predict(
                s.reshape(1, -1)).squeeze()
            return residual
        else:
            raise NotImplementedError

    def preproc_dynamics_inputs(self, obs, ac):
        gripper_pos = obs[:2]
        obj_pos = obs[3:5]
        s = np.concatenate([gripper_pos, obj_pos])
        ac_ind = self.env.discrete_actions[tuple(ac)]

        s_tensor = torch.as_tensor(s, dtype=torch.float32).view(1, -1)
        a_tensor = torch.as_tensor(ac_ind, dtype=torch.long).view(1, -1)

        return s_tensor, a_tensor

    def learn_online_in_real_world(self, max_timesteps=None):
        # If any pre-existing model is given, load it
        if self.args.load_dir:
            self.load()

        # Reset the environment
        observation = self.env.reset()
        # Configure heuristic for controller
        self.controller.reconfigure_heuristic(
            lambda obs: get_state_value_residual(obs,
                                                 self.preproc_inputs,
                                                 self.state_value_residual))
        # Configure dynamics for controller
        if self.args.agent == 'rts':
            self.controller.reconfigure_discrepancy(
                lambda obs, ac: get_discrepancy_neighbors(obs,
                                                          ac,
                                                          self.construct_4d_point,
                                                          self.kdtrees,
                                                          self.args.neighbor_radius))

        # Configure dynamics for controller
        if self.args.agent == 'mbpo' or self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp':
            self.controller.reconfigure_residual_dynamics(
                self.get_residual_dynamics)
        # Count of total number of steps
        total_n_steps = 0
        while True:
            obs = observation['observation']
            g = observation['desired_goal']
            qpos = observation['sim_state'].qpos
            qvel = observation['sim_state'].qvel
            # Get action from the controller
            ac, info = self.controller.act(observation)
            if self.args.agent == 'rts':
                assert self.controller.residual_dynamics_fn is None
            if self.args.agent == 'mbpo' or self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp':
                assert self.controller.discrepancy_fn is None
            # Get discrete action index
            ac_ind = self.env.discrete_actions[tuple(ac)]
            # Get the next observation
            next_observation, rew, _, _ = self.env.step(ac)
            # if np.array_equal(obs, next_observation['observation']):
            #     import ipdb
            #     ipdb.set_trace()
            # print('ACTION', ac)
            # print('VALUE PREDICTED', info['start_node_h'])
            # print('COST', -rew)
            if self.args.render:
                self.env.render()
            total_n_steps += 1
            # Check if we reached the goal
            if self.env.env._is_success(next_observation['achieved_goal'], g):
                print('REACHED GOAL!')
                break
            # Get the next obs
            obs_next = next_observation['observation']
            # Get the sim next obs
            set_sim_state_and_goal(self.planning_env,
                                   qpos.copy(),
                                   qvel.copy(),
                                   g.copy())
            next_observation_sim, _, _, _ = self.planning_env.step(ac)
            obs_sim_next = next_observation_sim['observation']
            # Store transition
            transition = [obs.copy(),
                          g.copy(),
                          ac_ind,
                          qpos.copy(),
                          qvel.copy(),
                          obs_next.copy(),
                          obs_sim_next.copy()]
            dynamics_losses = []
            # RTS
            if self.args.agent == 'rts' and self._check_dynamics_transition(transition):
                # print('DISCREPANCY IN DYNAMICS')
                self.memory.store_real_world_transition(transition)
                # # Fit model
                self._update_discrepancy_model()
            # MBPO
            elif self.args.agent == 'mbpo' or self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp':
                self.memory.store_real_world_transition(transition)
                # Update the dynamics
                if self.args.agent == 'mbpo':
                    for _ in range(self.args.n_online_planning_updates):
                        # Update dynamics
                        loss = self._update_batch_residual_dynamics()
                        dynamics_losses.append(loss.item())
                else:
                    loss = self._update_residual_dynamics()
                    dynamics_losses.append(loss)
            # # Plan in the model
            value_loss = self.plan_online_in_model(n_planning_updates=self.args.n_online_planning_updates,
                                                   initial_observation=copy.deepcopy(observation))

            # Log
            logger.record_tabular('n_steps', total_n_steps)
            if self.args.agent == 'mbpo' or self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp':
                logger.record_tabular(
                    'dynamics loss', np.mean(dynamics_losses))
            logger.record_tabular('residual_loss', value_loss)
            # logger.dump_tabular()
            # Move to next iteration
            observation = copy.deepcopy(next_observation)

            if max_timesteps and total_n_steps >= max_timesteps:
                break

        return total_n_steps
