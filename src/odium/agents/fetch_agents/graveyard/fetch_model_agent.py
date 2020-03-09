from datetime import datetime
import copy
import pickle
import numpy as np
import ray
import torch
import warnings

from odium.utils.simple_utils import multi_merge, convert_to_list_of_np_arrays, multi_append
from odium.utils.agent_utils.save_load_agent import save_agent, load_agent
from odium.utils.simulation_utils import set_sim_state_and_goal, apply_dynamics_residual

import odium.utils.logger as logger

from odium.agents.fetch_agents.sampler import rts_sampler
from odium.agents.fetch_agents.memory import rts_memory
from odium.agents.fetch_agents.approximators import StateValueResidual, get_state_value_residual, Dynamics, get_next_observation
from odium.agents.fetch_agents.normalizer import FeatureNormalizer
from odium.agents.fetch_agents.worker import InternalRolloutWorker, set_workers_num_expansions


class fetch_model_agent:
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
        self.residual_dynamics = Dynamics(env_params)

        # Fake KDTrees not used
        self.kdtrees = [None for _ in range(self.env_params['num_actions'])]

        # Optimizers
        # Initialize all optimizers
        # STATE VALUE RESIDUAL
        self.state_value_residual_optim = torch.optim.Adam(
            self.state_value_residual.parameters(),
            lr=args.lr_value_residual,
            weight_decay=args.l2_reg_value_residual)
        # LEARNED DYNAMICS
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
    OFFLINE
    '''

    # def learn_offline_in_model(self):
    #     '''
    #     This function is actually not used online; this is to compute a near-optimal
    #     policy in simulation offline
    #     '''
    #     if not self.args.offline:
    #         warnings.warn('SHOULD NOT BE USED ONLINE')
    #     logger.info("Training")
    #     # self.memory.clear(internal=True, real=False)
    #     # For each epoch
    #     best_success_rate = 0.0  # Maintain best success rate so far
    #     n_updates = 0
    #     for epoch in range(self.args.n_epochs):
    #         state_value_residual_losses = []
    #         dynamics_losses = []
    #         # For each cycle
    #         for _ in range(self.args.n_cycles):
    #             # Collect trajectories
    #             self.n_planning_steps += self.collect_internal_model_trajectories(
    #                 self.args.n_rollouts_per_cycle)
    #             # Update state value residuals
    #             for _ in range(self.args.n_batches):
    #                 state_value_residual_loss = self._update_state_value_residual()
    #                 dynamics_loss = self._update_residual_dynamics()
    #                 n_updates += 1
    #                 state_value_residual_losses.append(
    #                     state_value_residual_loss.item())
    #                 dynamics_losses.append(dynamics_loss.item())

    #             # Update target network
    #             self._update_target_network(self.state_value_target_residual,
    #                                         self.state_value_residual)
    #         # Evaluate agent in the model
    #         mean_success_rate, mean_return = self.eval_agent_in_model()
    #         # Check if this is a better residual
    #         if mean_success_rate > best_success_rate:
    #             best_success_rate = mean_success_rate
    #             print('Best success rate so far', best_success_rate)
    #             if self.args.save_dir is not None:
    #                 print('Saving residual and learned dynamics')
    #                 self.save(epoch, best_success_rate)
    #         # Print to stdout
    #         print('[{}] epoch is: {}, Num steps: {}, eval success rate is: {:.3f}'.format(
    #             datetime.now(), epoch, self.n_planning_steps, mean_success_rate))
    #         # Log all relevant values
    #         logger.record_tabular('epoch', epoch)
    #         logger.record_tabular('n_steps', self.n_planning_steps)
    #         logger.record_tabular('success_rate', mean_success_rate)
    #         logger.record_tabular('return', mean_return)
    #         logger.record_tabular(
    #             'state_value_residual_loss', np.mean(state_value_residual_losses))
    #         logger.record_tabular('dynamics_loss', np.mean(dynamics_losses))
    #         logger.dump_tabular()

    # def eval_agent_in_model(self):
    #     '''
    #     This function is not actually used; It can be used to evaluate intermediate policies
    #     while training offline in simulation
    #     '''
    #     if not self.args.offline:
    #         warnings.warn('SHOULD NOT BE USED ONLINE')
    #     num_test_rollouts = self.args.n_test_rollouts
    #     # Compute number of rollouts per worker
    #     num_workers = self.args.n_rts_workers
    #     if num_test_rollouts < self.args.n_rts_workers:
    #         num_workers = num_test_rollouts
    #     num_per_worker = num_test_rollouts // num_workers
    #     # assign jobs to workers
    #     results, count = [], 0
    #     # Put residual in object store
    #     value_residual_state_dict_id = ray.put(
    #         self.state_value_residual.state_dict())
    #     # Put normalizer in object store
    #     feature_norm_dict_id = ray.put(self.features_normalizer.state_dict())
    #     # Put knn dynamics residuals in object store
    #     kdtrees_serialized_id = ray.put(pickle.dumps(
    #         self.kdtrees))
    #     for worker_id in range(num_workers):
    #         if worker_id == num_workers - 1:
    #             # last worker takes the remaining load
    #             num_per_worker = num_test_rollouts - count
    #         # assign worker params
    #         ray.get(self.internal_rollout_workers[worker_id].set_worker_params.remote(
    #             value_residual_state_dict_id,
    #             feature_norm_dict_id,
    #             kdtrees_serialized_id))
    #         # Send job
    #         results.append(
    #             self.internal_rollout_workers[worker_id].evaluate.remote(
    #                 num_per_worker))
    #         count += num_per_worker

    #     # Check if all jobs have been assigned
    #     assert count == num_test_rollouts
    #     # Get all results
    #     results = ray.get(results)
    #     total_success_rate, total_return = [], []
    #     for result in results:
    #         per_success_rate, per_return = result
    #         total_success_rate += per_success_rate
    #         total_return += per_return

    #     # Compute stats
    #     total_success_rate = np.array(total_success_rate)
    #     mean_success_rate = np.mean(total_success_rate[:, -1])
    #     mean_return = np.mean(total_return)

    #     return mean_success_rate, mean_return

    # def _update_state_value_residual(self):
    #     # Sample tranistions
    #     transitions = self.memory.sample_internal_world_memory(
    #         self.args.batch_size)
    #     qpos, qvel = transitions['qpos'], transitions['qvel']
    #     obs, g, ag = transitions['obs'], transitions['g'], transitions['ag']
    #     features, heuristic = transitions['features'], transitions['heuristic']
    #     targets = []

    #     # Compute target by restarting search from the sampled states
    #     num_workers = self.args.n_rts_workers
    #     if self.args.batch_size < self.args.n_rts_workers:
    #         num_workers = self.args.batch_size
    #     num_per_worker = self.args.batch_size // num_workers
    #     # Put residual in object store
    #     value_target_residual_state_dict_id = ray.put(
    #         self.state_value_target_residual.state_dict())
    #     # Put normalizer in object store
    #     feature_norm_dict_id = ray.put(self.features_normalizer.state_dict())
    #     # Put knn dynamics residuals in object store
    #     kdtrees_serialized_id = ray.put(pickle.dumps(
    #         self.kdtrees))
    #     results, count = [], 0
    #     # Set all workers num expansions
    #     set_workers_num_expansions(self.internal_rollout_workers,
    #                                self.args.n_offline_expansions)
    #     for worker_id in range(num_workers):
    #         if worker_id == num_workers - 1:
    #             # last worker takes the remaining load
    #             num_per_worker = self.args.batch_size - count
    #         # Set parameters
    #         ray.get(self.internal_rollout_workers[worker_id].set_worker_params.remote(
    #             value_target_residual_state_dict_id,
    #             feature_norm_dict_id,
    #             kdtrees_serialized_id))
    #         # Send Job
    #         results.append(
    #             self.internal_rollout_workers[worker_id].lookahead_batch.remote(
    #                 obs[count:count+num_per_worker],
    #                 g[count:count+num_per_worker],
    #                 ag[count:count+num_per_worker],
    #                 qpos[count:count+num_per_worker],
    #                 qvel[count:count+num_per_worker]))
    #         count += num_per_worker
    #     # Check if all transitions have targets
    #     assert count == self.args.batch_size
    #     # Get all targets
    #     results = ray.get(results)
    #     targets = [item for sublist in results for item in sublist]
    #     targets = np.array(targets).reshape(-1, 1)
    #     targets = torch.as_tensor(targets, dtype=torch.float32)
    #     # Set all workers num expansions
    #     set_workers_num_expansions(self.internal_rollout_workers,
    #                                self.args.n_expansions)
    #     # Normalize features
    #     inputs_norm = torch.as_tensor(self.features_normalizer.normalize(features),
    #                                   dtype=torch.float32)
    #     heuristic_tensor = torch.as_tensor(
    #         heuristic, dtype=torch.float32).view(-1, 1)

    #     # Compute target residuals
    #     target_residual_tensor = targets - heuristic_tensor
    #     # Clip target residual tenssor to avoid value function less than zero
    #     target_residual_tensor = torch.max(
    #         target_residual_tensor, -heuristic_tensor)
    #     # Clip target residual tensor to avoid value function greater than horizon
    #     if self.args.offline:
    #         target_residual_tensor = torch.min(target_residual_tensor,
    #                                            self.env_params['offline_max_timesteps'] - heuristic_tensor)

    #     # COmpute predictions
    #     residual_tensor = self.state_value_residual(inputs_norm)
    #     # COmpute loss
    #     state_value_residual_loss = (
    #         residual_tensor - target_residual_tensor).pow(2).mean()

    #     # Backprop and step
    #     self.state_value_residual_optim.zero_grad()
    #     state_value_residual_loss.backward()
    #     self.state_value_residual_optim.step()

    #     # Configure heuristic for controller
    #     self.controller.reconfigure_heuristic(
    #         lambda obs: get_state_value_residual(obs,
    #                                              self.preproc_inputs,
    #                                              self.state_value_residual))

    #     return state_value_residual_loss

    # def collect_internal_model_trajectories(self, num_rollouts, initial_state=None):
    #     '''
    #     Collects trajectories based on controller and learned residual
    #     '''
    #     # Caluculate number of jobs per worker
    #     num_workers = self.args.n_rts_workers
    #     if num_rollouts < self.args.n_rts_workers:
    #         num_workers = num_rollouts
    #     num_per_worker = num_rollouts // num_workers
    #     # Data structures
    #     mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic = [], [], [], [], []
    #     mb_reward, mb_qpos, mb_qvel, mb_features = [], [], [], []
    #     mb_n_steps = 0
    #     # assign jobs to workers
    #     results, count = [], 0
    #     # Put residual in object store
    #     value_residual_state_dict_id = ray.put(
    #         self.state_value_residual.state_dict())
    #     # Put normalizer in object store
    #     feature_norm_dict_id = ray.put(self.features_normalizer.state_dict())
    #     # Put knn dynamics residuals in object store
    #     kdtrees_serialized_id = ray.put(pickle.dumps(
    #         self.kdtrees))
    #     for worker_id in range(num_workers):
    #         if worker_id == num_workers - 1:
    #             # Last worker takes the remaining load
    #             num_per_worker = num_rollouts - count
    #         # Set worker params
    #         current_worker = self.internal_rollout_workers[worker_id]
    #         ray.get(current_worker.set_worker_params.remote(value_residual_state_dict_id,
    #                                                         feature_norm_dict_id,
    #                                                         kdtrees_serialized_id))
    #         # Do rollouts
    #         results.append(current_worker.do_rollouts.remote(num_rollouts=num_per_worker,
    #                                                          initial_state=initial_state))
    #         count += num_per_worker
    #     # Check that all rollouts are done
    #     assert count == num_rollouts
    #     # GET ALL RESULTS
    #     results = ray.get(results)
    #     # Compile all the rollout data
    #     for result in results:
    #         c_obs, c_ag, c_g, c_actions, c_heuristic, c_reward, c_qpos, c_qvel, c_features, c_steps = result
    #         mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic, mb_reward, mb_qpos, mb_qvel, mb_features = multi_merge(
    #             [mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
    #              mb_reward, mb_qpos, mb_qvel, mb_features],
    #             [c_obs, c_ag, c_g, c_actions, c_heuristic,
    #              c_reward, c_qpos, c_qvel, c_features]
    #         )
    #         mb_n_steps += c_steps

    #     # Convert to np arrays
    #     mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic, mb_reward, mb_qpos, mb_qvel, mb_features = convert_to_list_of_np_arrays(
    #         [mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
    #          mb_reward, mb_qpos, mb_qvel, mb_features]
    #     )
    #     # Store in memory
    #     self.memory.store_internal_model_rollout([mb_obs, mb_ag, mb_g,
    #                                               mb_actions, mb_heuristic, mb_reward,
    #                                               mb_qpos, mb_qvel, mb_features])
    #     # Update normalizer
    #     self.features_normalizer.update_normalizer([mb_obs, mb_ag, mb_g,
    #                                                 mb_actions, mb_heuristic, mb_reward,
    #                                                 mb_qpos, mb_qvel, mb_features],
    #                                                self.sampler)
    #     return mb_n_steps

    '''
    ONLINE
    '''

    def save(self, epoch, success_rate):
        return save_agent(path=self.args.save_dir+'/fetch_model_agent.pth',
                          network_state_dict=self.state_value_residual.state_dict(),
                          optimizer_state_dict=self.state_value_residual_optim.state_dict(),
                          normalizer_state_dict=self.features_normalizer.state_dict(),
                          dynamics_state_dict=self.residual_dynamics.state_dict(),
                          dynamics_optimizer_state_dict=self.residual_dynamics_optim.state_dict(),
                          epoch=epoch,
                          success_rate=success_rate)

    def load(self):
        load_dict, load_dict_keys = load_agent(
            self.args.load_dir+'/fetch_model_agent.pth')
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
            self.residual_dynamics.load_state_dict(
                load_dict['dynamics_state_dict'])
        if 'dynamics_optimizer_state_dict' in load_dict_keys:
            self.residual_dynamics_optim.load_state_dict(
                load_dict['dynamics_optimizer_state_dict'])
        return

    def _update_target_network(self, target, source):
        # Simply copy parameter values
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _update_residual_dynamics(self):
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

        return loss

    def preproc_inputs(self, obs, g):
        '''
        Function to preprocess inputs
        '''
        features = self.env.extract_features(obs, g)
        features_norm = self.features_normalizer.normalize(features)
        inputs = torch.as_tensor(
            features_norm, dtype=torch.float32).view(1, -1)
        return inputs

    def preproc_dynamics_inputs(self, obs, ac):
        gripper_pos = obs[:2]
        obj_pos = obs[3:5]
        s = np.concatenate([gripper_pos, obj_pos])
        ac_ind = self.env.discrete_actions[tuple(ac)]

        s_tensor = torch.as_tensor(s, dtype=torch.float32).view(1, -1)
        a_tensor = torch.as_tensor(ac_ind, dtype=torch.long).view(1, -1)

        return s_tensor, a_tensor

    def online_rollout(self, initial_observation):
        n_steps = 0
        mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic = [], [], [], [], []
        mb_reward, mb_qpos, mb_qvel, mb_features = [], [], [], []
        # Set initial state
        observation = copy.deepcopy(initial_observation)
        # Data structures
        r_obs, r_ag, r_g, r_actions, r_heuristic = [], [], [], [], []
        r_reward, r_qpos, r_qvel, r_features = [], [], [], []
        # Start
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        qpos = observation['sim_state'].qpos
        qvel = observation['sim_state'].qvel
        set_sim_state_and_goal(
            self.planning_env, qpos.copy(), qvel.copy(), g.copy())
        features = self.env.extract_features(obs, g)
        heuristic = self.controller.heuristic_obs_g(obs, g)
        for _ in range(self.env_params['max_timesteps']):
            ac, _ = self.controller.act(observation)
            ac_ind = self.env.discrete_actions[tuple(ac)]
            next_observation, rew, _, info = self.planning_env.step(ac)
            # Get the next observation and reward using the learned model
            next_observation, rew = apply_dynamics_residual(self.planning_env,
                                                            self.get_residual_dynamics,
                                                            observation,
                                                            info,
                                                            ac,
                                                            next_observation)
            n_steps += 1
            # Add to data structures
            multi_append([r_obs, r_ag, r_g, r_actions, r_heuristic, r_reward, r_qpos, r_qvel, r_features],
                         [obs.copy(), ag.copy(), g.copy(), ac_ind, heuristic, rew, qpos.copy(), qvel.copy(), features.copy()])
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
                      mb_reward, mb_qpos, mb_qvel, mb_features],
                     [r_obs, r_ag, r_g, r_actions, r_heuristic,
                      r_reward, r_qpos, r_qvel, r_features])

        mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic, mb_reward, mb_qpos, mb_qvel, mb_features = convert_to_list_of_np_arrays(
            [mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
             mb_reward, mb_qpos, mb_qvel, mb_features]
        )

        # Store in memory
        self.memory.store_internal_model_rollout([mb_obs, mb_ag, mb_g,
                                                  mb_actions, mb_heuristic, mb_reward, mb_qpos, mb_qvel, mb_features])
        # Update normalizer
        self.features_normalizer.update_normalizer([mb_obs, mb_ag, mb_g,
                                                    mb_actions, mb_heuristic, mb_reward, mb_qpos, mb_qvel, mb_features], self.sampler)

        return n_steps

    def _update_state_value_residual_online(self):
        transitions = self.memory.sample_internal_world_memory(
            self.args.batch_size)

        obs, g, ag = transitions['obs'], transitions['g'], transitions['ag']
        qpos, qvel = transitions['qpos'], transitions['qvel']
        features_closed, heuristic_closed = [], []
        targets_closed = []

        num_workers = self.args.n_rts_workers
        if self.args.batch_size < self.args.n_rts_workers:
            num_workers = self.args.batch_size
        num_per_worker = self.args.batch_size // num_workers
        # Put residual in object store
        value_target_residual_state_dict_id = ray.put(
            self.state_value_target_residual.state_dict())
        # Put normalizer in object store
        feature_norm_dict_id = ray.put(self.features_normalizer.state_dict())
        # Put residual dynamics in object store
        residual_dynamics_state_dict_id = ray.put(
            self.residual_dynamics.state_dict())

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
        features_norm = self.features_normalizer.normalize(features_closed)
        inputs_norm = torch.as_tensor(features_norm, dtype=torch.float32)

        h_tensor = torch.as_tensor(
            heuristic_closed, dtype=torch.float32).view(-1, 1)
        # Compute target residuals
        target_residual_tensor = targets_tensor - h_tensor
        # Clip target residual tenssor to avoid value function less than zero
        target_residual_tensor = torch.max(
            target_residual_tensor, -h_tensor)

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

        import ipdb
        ipdb.set_trace()

        return state_value_residual_loss

    def plan_online_in_model(self, n_planning_updates, initial_observation):
        # Clear memory
        self.memory.clear(internal=True, real=False)
        n_updates = 0
        residual_losses = []
        while n_updates < n_planning_updates:
            n_updates += 1
            self.online_rollout(initial_observation)
            state_value_residual_loss = self._update_state_value_residual_online()
            residual_losses.append(state_value_residual_loss.item())
            self._update_target_network(self.state_value_target_residual,
                                        self.state_value_residual)

        return np.mean(residual_losses)

    def get_residual_dynamics(self, obs, ac):
        return get_next_observation(obs, ac,
                                    self.preproc_dynamics_inputs, self.residual_dynamics)

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
        self.controller.reconfigure_residual_dynamics(
            self.get_residual_dynamics)

        # Count total number of steps
        total_n_steps = 0
        while True:
            obs = observation['observation']
            g = observation['desired_goal']
            qpos = observation['sim_state'].qpos
            qvel = observation['sim_state'].qvel

            # Get action from controller
            ac, info = self.controller.act(observation)
            # Get discrete action index
            ac_ind = self.env.discrete_actions[tuple(ac)]
            # Get next observation
            next_observation, rew, _, _ = self.env.step(ac)
            # Increment counter
            total_n_steps += 1
            if self.env.env._is_success(next_observation['achieved_goal'],
                                        g):
                print('REACHED GOAL!')
                break
            if self.args.render:
                self.env.render()
            # Get next obs
            obs_next = next_observation['observation']
            # GEt sim next obs
            set_sim_state_and_goal(self.planning_env,
                                   qpos.copy(),
                                   qvel.copy(),
                                   g.copy())
            next_observation_sim, _, _, _ = self.planning_env.step(ac)
            obs_sim_next = next_observation_sim['observation']
            # Store transition in real world memory
            transition = [obs.copy(), g.copy(), ac_ind, qpos.copy(
            ), qvel.copy(), obs_next.copy(), obs_sim_next.copy()]
            self.memory.store_real_world_transition(transition)

            # Update the dynamics
            dynamics_losses = []
            for _ in range(self.args.n_online_planning_updates):
                # Update dynamics
                loss = self._update_residual_dynamics()
                dynamics_losses.append(loss.item())

            # Update state value residual
            value_loss = self.plan_online_in_model(self.args.n_online_planning_updates,
                                                   initial_observation=copy.deepcopy(observation))
            # log
            logger.record_tabular('n_steps', total_n_steps)
            logger.record_tabular('dynamics_loss', np.mean(dynamics_losses))
            logger.record_tabular('residual_loss', value_loss)
            logger.dump_tabular()

            # Move to next iteration
            observation = copy.deepcopy(next_observation)

            if max_timesteps and total_n_steps >= max_timesteps:
                break

        return total_n_steps
