from datetime import datetime
import torch
import numpy as np
from mpi4py import MPI
import ray
import copy

import odium.utils.logger as logger
from odium.utils.mpi_utils.normalizer import normalizer
from odium.utils.simple_utils import multi_append
from odium.utils.simulation_utils import set_sim_state_and_goal, set_gripper_position_in_sim, set_object_position_in_sim, get_sim_state_and_goal

from odium.agents.ilc_rts_agent.dataset import Dataset, DynamicsDataset
from odium.agents.ilc_rts_agent.models import Residual, DynamicsResidual
from odium.agents.ilc_rts_agent.sampler import Sampler
from odium.agents.ilc_rts_agent.worker import Worker


class ilc_rts_agent:
    def __init__(self, args, env, planning_env, env_params, controller):
        self.args = args
        self.env = env
        self.planning_env = planning_env
        self.env_params = env_params
        self.controller = controller
        self.controller_heuristic_fn = controller.heuristic_obs_g
        self.extract_features_fn = planning_env.extract_features
        self.reward_fn = planning_env.compute_reward

        self.sampler = Sampler(args, self.reward_fn,
                               self.controller_heuristic_fn,
                               self.extract_features_fn)
        self.dataset = Dataset(args, env_params, self.sampler)
        self.dynamics_dataset = DynamicsDataset(args, env_params)
        self.residual = Residual(env_params)
        self.residual_target = Residual(env_params)
        self.dynamics_residual = DynamicsResidual(env_params)
        self.residual_optim = torch.optim.Adam(
            # self.residual_optim = torch.optim.SGD(
            self.residual.parameters(),
            lr=self.args.lr_residual,
            # momentum=0.9,
            weight_decay=self.args.l2_reg
        )
        self.dynamics_residual_optim = torch.optim.Adam(
            # self.dynamics_residual_optim = torch.optim.SGD(
            self.dynamics_residual.parameters(),
            lr=self.args.lr_model,
            # momentum=0.9,
            weight_decay=self.args.model_l2_reg
        )
        # TODO: Sync networks, if we want to use MPI
        self.residual_target.load_state_dict(self.residual.state_dict())

        self.f_norm = normalizer(
            size=env_params['num_features'],
        )
        self.pos_norm = normalizer(
            size=4,
        )

        self.dummy_sim_state = self.planning_env.reset()['sim_state']

        self.workers = [Worker.remote(args, env_params)
                        for i in range(args.num_ilc_workers)]

        self.n_planning_steps = 0
        self.n_real_steps = 0

        # Store start and goal states of real env for num_real_traj_eval trajectories
        self.eval_qpos, self.eval_qvel, self.eval_goals = [], [], []
        self.populate_sim_states_and_goals()

    def collect_trajectories(self, num_traj):
        '''
        This function collects trajectories based on the controller and learned residuals
        '''
        logger.debug("Rolling out")
        mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f = [
        ], [], [], [], [], [], [], [], []
        for traj in range(num_traj):
            ep_obs, ep_ag, ep_g, ep_actions, ep_s_h, ep_r, ep_qpos, ep_qvel, ep_f = [
            ], [], [], [], [], [], [], [], []
            # observation = self.planning_env.reset()
            observation = set_sim_state_and_goal(
                self.planning_env,
                self.eval_qpos[traj],
                self.eval_qvel[traj],
                self.eval_goals[traj],
            )
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            s_h = self.controller.heuristic_obs_g(obs, g)
            f = self.planning_env.extract_features(obs, g)
            for _ in range(self.env_params['max_timesteps']):
                qpos = observation['sim_state'].qpos
                qvel = observation['sim_state'].qvel
                ac, info = self.controller.act(observation)
                ac_ind = self.planning_env.discrete_actions[tuple(ac)]
                logger.debug('Heuristic', info['start_node_h'])
                logger.debug('Action', ac)
                observation_new, rew, _, _ = self.planning_env.step(ac)
                # Apply dynamics residual
                observation_new, rew = self.apply_dynamics_residual(
                    observation, ac, observation_new, rew)
                self.n_planning_steps += 1
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                if self.args.render:
                    self.planning_env.render()
                multi_append([ep_obs, ep_ag, ep_g, ep_actions, ep_s_h, ep_r, ep_qpos, ep_qvel, ep_f],
                             [obs.copy(), ag.copy(), g.copy(), ac_ind, s_h, rew, qpos.copy(), qvel.copy(), f.copy()])
                obs = obs_new.copy()
                ag = ag_new.copy()
                observation = observation_new
                s_h = self.controller.heuristic_obs_g(obs, g)
                f = self.planning_env.extract_features(obs, g)
            multi_append([ep_obs, ep_ag, ep_s_h, ep_f],
                         [obs.copy(), ag.copy(), s_h, f.copy()])
            multi_append([mb_obs, mb_ag, mb_actions, mb_g, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f],
                         [ep_obs, ep_ag, ep_actions, ep_g, ep_s_h, ep_r, ep_qpos, ep_qvel, ep_f])
        mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f = np.array(mb_obs), np.array(mb_ag), np.array(
            mb_g), np.array(mb_actions), np.array(mb_s_h), np.array(mb_r), np.array(mb_qpos), np.array(mb_qvel), np.array(mb_f)
        self.dataset.store_episode(
            [mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f])
        # Update normalizer
        self._update_normalizer(
            [mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f])

    def apply_dynamics_residual(self, observation, action, next_observation, rew):
        residual_pos = self.get_dynamics_residual(
            observation, np.array(action))
        next_obj_pos = next_observation['observation'][3:5]
        next_gripper_pos = next_observation['observation'][0:2]
        corrected_obj_pos = next_obj_pos + residual_pos[2:4]
        corrected_gripper_pos = next_gripper_pos + residual_pos[0:2]
        next_observation['observation'][0:2] = corrected_gripper_pos
        next_observation['observation'][3:5] = corrected_obj_pos
        next_observation['observation'][6:8] = corrected_obj_pos - \
            corrected_gripper_pos
        next_observation['achieved_goal'][0:2] = corrected_obj_pos
        # Set object position
        set_object_position_in_sim(self.planning_env, corrected_obj_pos)
        # Set gripper position
        set_gripper_position_in_sim(
            self.planning_env, corrected_gripper_pos, next_gripper_pos)
        rew = self.planning_env.compute_reward(
            next_observation['achieved_goal'], next_observation['desired_goal'], {})

        return next_observation, rew

    def learn(self):
        logger.info("Training")
        # ILC loop
        # 1. Train the model on real environment transitions
        # 2. Plan in the model to get the optimal policy, and the direction of improvement
        # 3. Do line search on the real environment to find the right step size
        initial_residual_parameters = copy.deepcopy(self.residual.state_dict())
        for epoch in range(self.args.n_epochs):
            # 0. Fix start and goals
            self.populate_sim_states_and_goals()
            # 1. Plan in the model to get the optimal policy
            logger.info("Improving policy in the model")
            residual_losses = []
            for _ in range(self.args.n_cycles):
                # Collect trajectories
                self.controller.reconfigure_heuristic(self.get_residual)
                self.controller.reconfigure_dynamics(
                    self.get_dynamics_residual)
                self.collect_trajectories(
                    self.args.num_rollouts_per_mpi)
                # Update residual
                logger.info("Updating")
                for _ in range(self.args.n_batches):
                    residual_loss = self._update_residual()
                    residual_losses.append(
                        residual_loss.detach().cpu().numpy())
                    logger.info('Residual Loss', residual_loss.item())

                self._update_target_network(
                    self.residual_target, self.residual)

            if not self.args.planning:
                # Get the direction of improvement
                logger.info("Computing direction of improvement")
                final_residual_parameters = copy.deepcopy(
                    self.residual.state_dict())
                gradient = {}
                for key in initial_residual_parameters.keys():
                    gradient[key] = final_residual_parameters[key] - \
                        initial_residual_parameters[key]

                # 2. Line search in the real world
                logger.info("Line search in the real world")
                logger.info("Evaluating initial policy in the real world")
                initial_real_value_estimate = self.evaluate_real_world(
                    initial_residual_parameters)
                logger.info("Initial cost-to-go", initial_real_value_estimate)
                alpha = 1.0
                while True:
                    logger.info("Alpha", alpha)
                    current_residual_parameters = {}
                    for key in initial_residual_parameters.keys():
                        current_residual_parameters[key] = initial_residual_parameters[key] + \
                            alpha * gradient[key]

                    current_real_value_estimate = self.evaluate_real_world(
                        current_residual_parameters)
                    logger.info("Current cost-to-go",
                                current_real_value_estimate)

                    if current_real_value_estimate < initial_real_value_estimate:
                        # Cost to go decreased - found an alpha
                        logger.info("Initial cost-to-go", initial_real_value_estimate,
                                    "Final cost-to-go", current_real_value_estimate)
                        initial_real_value_estimate = current_real_value_estimate
                        initial_residual_parameters = copy.deepcopy(
                            current_residual_parameters)
                        break
                    else:
                        # Decrease alpha
                        alpha *= 0.5

                    if alpha < self.args.alpha_threshold:
                        # If alpha is really really small
                        # Don't update the residual
                        logger.info(
                            "Alpha really small. Not updating residual")
                        logger.info("Best cost-to-go so far",
                                    initial_real_value_estimate)
                        break

                # Assign chosen residual parameters for the residual
                self.residual.load_state_dict(initial_residual_parameters)
                self.residual_target.load_state_dict(
                    initial_residual_parameters)

            logger.info("Evaluating")
            success_rate = self.eval_agent()

            if not self.args.planning:
                # 3. Train model on real world transitions collected so far
                logger.info("Training model residual using real world samples")
                model_losses = []
                for _ in range(self.args.n_model_batches):
                    model_loss = self._update_model()
                    model_losses.append(model_loss.detach().cpu().numpy())
                    logger.info('Model Loss', model_loss.item())
            else:
                model_losses = [0]

            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, Num planning steps: {}, Num real steps: {}, eval success rate is: {:.3f}'.format(
                    datetime.now(), epoch, self.n_planning_steps, self.n_real_steps, success_rate))
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('n_planning_steps',
                                      self.n_planning_steps)
                logger.record_tabular('n_real_steps', self.n_real_steps)
                logger.record_tabular('success_rate', success_rate)
                logger.record_tabular(
                    'residual_loss', np.mean(residual_losses))
                logger.record_tabular('model_loss', np.mean(model_losses))
                # logger.record_tabular(
                #     'cost-to-go', initial_real_value_estimate)
                logger.dump_tabular()

    def evaluate_real_world(self, residual_parameters):
        # TODO: Parallelize this function
        # Copy parameters to residual
        self.residual.load_state_dict(residual_parameters)
        self.controller.reconfigure_heuristic(self.get_residual)
        self.controller.reconfigure_dynamics(self.get_dynamics_residual)
        mb_obs, mb_actions, mb_qpos, mb_qvel, mb_returns = [], [], [], [], []
        mb_obs_model_next = []
        for traj in range(self.args.num_real_traj_eval):
            ep_obs, ep_actions, ep_qpos, ep_qvel = [], [], [], []
            ep_obs_model_next = []
            current_return = 0.
            observation = set_sim_state_and_goal(self.env,
                                                 self.eval_qpos[traj],
                                                 self.eval_qvel[traj],
                                                 self.eval_goals[traj])
            obs = observation['observation']
            for _ in range(self.env_params['max_timesteps']):
                qpos = observation['sim_state'].qpos.copy()
                qvel = observation['sim_state'].qvel.copy()
                goal = observation['desired_goal'].copy()
                ac, info = self.controller.act(observation)
                observation_new, rew, _, _ = self.env.step(ac)
                if self.args.render:
                    self.env.render()
                # Set model to the same state
                _ = set_sim_state_and_goal(
                    self.planning_env,
                    qpos,
                    qvel,
                    goal,
                )
                model_observation_next, _, _, _ = self.planning_env.step(ac)
                obs_model_next = model_observation_next['observation']
                self.n_real_steps += 1
                obs_new = observation_new['observation']
                multi_append([ep_obs, ep_actions, ep_qpos, ep_qvel, ep_obs_model_next], [
                             obs.copy(), ac.copy(), qpos.copy(), qvel.copy(), obs_model_next.copy()])
                current_return += -rew
                obs = obs_new.copy()
                observation = observation_new

            ep_obs.append(obs.copy())
            multi_append([mb_obs, mb_actions, mb_qpos, mb_qvel, mb_returns, mb_obs_model_next], [
                         ep_obs, ep_actions, ep_qpos, ep_qvel, current_return, ep_obs_model_next])

        mb_obs, mb_actions, mb_qpos, mb_qvel, mb_obs_model_next = np.array(mb_obs), np.array(
            mb_actions), np.array(mb_qpos), np.array(mb_qvel), np.array(mb_obs_model_next)
        self.dynamics_dataset.store_episode(
            [mb_obs, mb_actions, mb_qpos, mb_qvel, mb_obs_model_next])
        self._update_dynamics_normalizer(
            [mb_obs, mb_actions, mb_qpos, mb_qvel, mb_obs_model_next])
        return np.mean(mb_returns)

    def populate_sim_states_and_goals(self):
        self.eval_qpos.clear()
        self.eval_qvel.clear()
        self.eval_goals.clear()
        for _ in range(self.args.num_real_traj_eval):
            self.env.reset()
            qpos, qvel, goal = get_sim_state_and_goal(self.env)
            self.eval_qpos.append(qpos)
            self.eval_qvel.append(qvel)
            self.eval_goals.append(goal)

    def _update_model(self):
        transitions = self.dynamics_dataset.sample(self.args.batch_size)

        obs, obs_next = transitions['obs'], transitions['obs_next']
        ac = transitions['actions']
        ac_ind = np.array(
            [[i, self.planning_env.discrete_actions[tuple(a)]] for i, a in enumerate(ac)])

        real_pos, real_next_pos = np.concatenate(
            [obs[:, 0:2], obs[:, 3:5]], axis=1), np.concatenate([obs_next[:, 0:2], obs_next[:, 3:5]], axis=1)
        # Get the model next object pos
        model_obs_next = np.array(
            transitions['obs_model_next'])
        model_next_pos = np.concatenate(
            [model_obs_next[:, 0:2], model_obs_next[:, 3:5]], axis=1)

        # Compute target residuals
        target_residuals = real_next_pos - model_next_pos
        target_residuals = torch.as_tensor(
            target_residuals, dtype=torch.float32)
        # Compute predicted residuals
        real_pos_norm = self.pos_norm.normalize(real_pos)
        ac_input = np.zeros(
            (self.args.batch_size, self.env_params['num_actions']))
        ac_input[ac_ind[:, 0], ac_ind[:, 1]] = 1
        inputs = np.concatenate([real_pos_norm, ac_input], axis=1)
        inputs_tensor = torch.as_tensor(inputs, dtype=torch.float32)
        predicted_residuals = self.dynamics_residual(inputs_tensor)

        # model_loss = (predicted_residuals - target_residuals).pow(2).mean()
        # Using the L2 norm instead of MSE
        model_loss = torch.norm(predicted_residuals -
                                target_residuals, dim=1).mean()

        self.dynamics_residual_optim.zero_grad()
        model_loss.backward()
        self.dynamics_residual_optim.step()

        return model_loss

    def _preproc_inputs(self, obs, g):
        f = self.planning_env.extract_features(obs, g)
        f_norm = self.f_norm.normalize(f)
        # concatenate the stuffs
        inputs = f_norm
        inputs = torch.as_tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def _preproc_dynamics_inputs(self, obs, ac):
        pos = np.concatenate([obs[0:2], obs[3:5]])
        pos_norm = self.pos_norm.normalize(pos)
        ac_input = np.zeros(self.env_params['num_actions'])
        ac_ind = self.planning_env.discrete_actions[tuple(ac)]
        ac_input[ac_ind] = 1
        inputs = np.concatenate([pos_norm, ac_input])
        inputs_tensor = torch.as_tensor(
            inputs, dtype=torch.float32).unsqueeze(0)
        return inputs_tensor

    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        mb_s_h_next = mb_s_h[:, 1:]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       's_h': mb_s_h,
                       's_h_next': mb_s_h_next,
                       'r': mb_r,
                       'qpos': mb_qpos,
                       'qvel': mb_qvel,
                       'f': mb_f,
                       }
        transitions = self.sampler.sample(
            buffer_temp, num_transitions)
        self.f_norm.update(transitions['f'])
        # recompute the stats
        self.f_norm.recompute_stats()

    def _update_dynamics_normalizer(self, episode_batch):
        mb_obs, mb_actions, mb_qpos, mb_qvel, mb_obs_model_next = episode_batch
        # FIXME: Using object position for now
        obj_pos = np.concatenate([mb_obs[:, 0:2], mb_obs[:, 3:5]], axis=1)
        self.pos_norm.update(obj_pos)
        self.pos_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def _update_residual(self):
        transitions = self.dataset.sample(self.args.batch_size)

        qpos, qvel = transitions['qpos'], transitions['qvel']
        obs, g, ag = transitions['obs'], transitions['g'], transitions['ag']
        f = transitions['f']
        targets = []

        num_per_worker = self.args.batch_size // self.args.num_ilc_workers
        count = 0
        residual_target_id = ray.put(self.residual_target.state_dict())
        residual_dynamics_id = ray.put(self.dynamics_residual.state_dict())
        results = []
        for i in range(self.args.num_ilc_workers):
            self.workers[i].set_residual.remote(residual_target_id)
            self.workers[i].set_dynamics_residual.remote(residual_dynamics_id)
            self.workers[i].set_feature_normalizer.remote(
                self.f_norm.mean, self.f_norm.std)
            self.workers[i].set_dynamics_normalizer.remote(
                self.pos_norm.mean, self.pos_norm.std)
            if i == self.args.num_ilc_workers - 1:
                # The last worker takes the remaining load
                num_per_worker = self.args.batch_size - count
            result = self.workers[i].act_batch.remote(obs[count:count+num_per_worker],
                                                      g[count:count +
                                                        num_per_worker],
                                                      ag[count:count +
                                                         num_per_worker],
                                                      qpos[count:count +
                                                           num_per_worker],
                                                      qvel[count:count+num_per_worker])
            results.append(result)
            count += num_per_worker

        results = ray.get(results)
        targets = [item for sublist in results for item in sublist]

        targets = np.array(targets).reshape(-1, 1)
        targets = torch.as_tensor(targets, dtype=torch.float32)
        f_norm = self.f_norm.normalize(f)
        inputs_norm = f_norm
        inputs_norm_tensor = torch.as_tensor(inputs_norm, dtype=torch.float32)
        h = transitions['s_h']
        h_tensor = torch.as_tensor(h, dtype=torch.float32).unsqueeze(-1)

        target_residual_tensor = targets - h_tensor
        # Clip target residual tensor to avoid value function less than zero
        target_residual_tensor = torch.max(targets - h_tensor, -h_tensor)
        # Clip target residual tensor to avoid value function greater than horizon
        target_residual_tensor = torch.min(
            target_residual_tensor, self.env_params['max_timesteps'] - h_tensor)

        residual_tensor = self.residual(inputs_norm_tensor)
        residual_loss = (residual_tensor -
                         target_residual_tensor).pow(2).mean()

        self.residual_optim.zero_grad()
        residual_loss.backward()
        self.residual_optim.step()

        return residual_loss

    def get_residual(self, observation):
        obs = observation['observation']
        g = observation['desired_goal']
        inputs_tensor = self._preproc_inputs(obs, g)
        with torch.no_grad():
            residual_tensor = self.residual(inputs_tensor)
            residual = residual_tensor.detach().cpu().numpy().squeeze()
        return residual

    def get_dynamics_residual(self, observation, ac):
        obs = observation['observation']
        inputs_tensor = self._preproc_dynamics_inputs(obs, ac)
        with torch.no_grad():
            dynamics_residual_tensor = self.dynamics_residual(inputs_tensor)
            dynamics_residual = dynamics_residual_tensor.detach().cpu().numpy().squeeze()

        return dynamics_residual

    def get_residual_target(self, observation):
        obs = observation['observation']
        g = observation['desired_goal']
        inputs_tensor = self._preproc_inputs(obs, g)
        with torch.no_grad():
            residual_tensor = self.residual_target(inputs_tensor)
            residual = residual_tensor.detach().cpu().numpy().squeeze()
        return residual

    def eval_agent(self):
        # TODO: Parallelize eval agent
        total_success_rate = []
        self.controller.reconfigure_heuristic(self.get_residual)
        self.controller.reconfigure_dynamics(self.get_dynamics_residual)
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            for _ in range(self.env_params['max_timesteps']):
                ac, _ = self.controller.act(observation)
                observation, _, _, info = self.env.step(ac)
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(
            local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
