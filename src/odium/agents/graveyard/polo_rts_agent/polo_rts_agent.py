from datetime import datetime
import torch
import numpy as np
from mpi4py import MPI
import ray

import odium.utils.logger as logger
from odium.utils.mpi_utils.normalizer import normalizer
from odium.utils.simple_utils import multi_append

from odium.agents.polo_rts_agent.dataset import Dataset
from odium.agents.polo_rts_agent.models import Residual
from odium.agents.polo_rts_agent.sampler import Sampler
from odium.agents.polo_rts_agent.worker import Worker


class polo_rts_agent:
    def __init__(self, args, env, env_params, controller):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.controller = controller
        self.controller_heuristic_fn = controller.heuristic_obs_g
        self.extract_features_fn = env.extract_features
        self.reward_fn = env.compute_reward

        self.sampler = Sampler(args, self.reward_fn,
                               self.controller_heuristic_fn,
                               self.extract_features_fn)
        self.dataset = Dataset(args, env_params, self.sampler)
        self.residual = Residual(env_params)
        self.residual_target = Residual(env_params)
        # self.residual_optim = torch.optim.Adam(
        self.residual_optim = torch.optim.SGD(
            self.residual.parameters(),
            lr=self.args.lr_residual,
            momentum=0.9,
            weight_decay=self.args.l2_reg
        )
        # TODO: Sync networks, if we want to use MPI
        self.residual_target.load_state_dict(self.residual.state_dict())

        self.f_norm = normalizer(
            size=env_params['num_features']
        )

        self.dummy_sim_state = self.env.reset()['sim_state']

        self.workers = [Worker.remote(args, env_params)
                        for i in range(args.num_polo_workers)]

    def collect_trajectories(self, num_traj):
        '''
        This function collects trajectories based on the controller and learned residuals
        '''
        logger.debug("Rolling out")
        n_steps = 0
        mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f = [
        ], [], [], [], [], [], [], [], []
        for traj in range(num_traj):
            ep_obs, ep_ag, ep_g, ep_actions, ep_s_h, ep_r, ep_qpos, ep_qvel, ep_f = [
            ], [], [], [], [], [], [], [], []
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            s_h = self.controller.heuristic_obs_g(obs, g)
            f = self.env.extract_features(obs, g)
            for _ in range(self.env_params['max_timesteps']):
                qpos = observation['sim_state'].qpos
                qvel = observation['sim_state'].qvel
                ac, info = self.controller.act(observation)
                ac_ind = self.env.discrete_actions[tuple(ac)]
                logger.debug('Heuristic', info['start_node_h'])
                logger.debug('Action', ac)
                observation_new, rew, _, _ = self.env.step(ac)
                n_steps += 1
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                if self.args.render:
                    self.env.render()
                multi_append([ep_obs, ep_ag, ep_g, ep_actions, ep_s_h, ep_r, ep_qpos, ep_qvel, ep_f],
                             [obs.copy(), ag.copy(), g.copy(), ac_ind, s_h, rew, qpos.copy(), qvel.copy(), f.copy()])
                obs = obs_new.copy()
                ag = ag_new.copy()
                observation = observation_new
                s_h = self.controller.heuristic_obs_g(obs, g)
                f = self.env.extract_features(obs, g)
            multi_append([ep_obs, ep_ag, ep_s_h, ep_f],
                         [obs.copy(), ag.copy(), s_h, f.copy()])
            multi_append([mb_obs, mb_ag, mb_actions, mb_g,
                          mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f],
                         [ep_obs, ep_ag, ep_actions, ep_g, ep_s_h, ep_r, ep_qpos, ep_qvel, ep_f])
        mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f = np.array(mb_obs), np.array(mb_ag), np.array(
            mb_g), np.array(mb_actions), np.array(mb_s_h), np.array(mb_r), np.array(mb_qpos), np.array(mb_qvel), np.array(mb_f)
        self.dataset.store_episode(
            [mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f])
        # Update normalizer
        self._update_normalizer(
            [mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f])
        return n_steps

    def learn(self):
        logger.info("Training")
        n_steps = 0
        # best_success_rate = 0.

        for epoch in range(self.args.n_epochs):
            residual_losses = []
            for _ in range(self.args.n_cycles):
                # Collect trajectories
                self.controller.reconfigure_heuristic(self.get_residual)
                n_steps += self.collect_trajectories(
                    self.args.num_rollouts_per_mpi)
                # Update residual
                logger.debug("Updating")
                for _ in range(self.args.n_batches):
                    residual_loss = self._update_residual()
                    residual_losses.append(
                        residual_loss.detach().cpu().numpy())
                    logger.debug('Loss', residual_loss.item())

                self._update_target_network(
                    self.residual_target, self.residual)

                if self.args.clear_buffer:
                    self.dataset.clear()

            success_rate = self.eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, Num steps: {}, eval success rate is: {:.3f}'.format(
                    datetime.now(), epoch, n_steps, success_rate))
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('n_steps', n_steps)
                logger.record_tabular('success_rate', success_rate)
                logger.record_tabular(
                    'residual_loss', np.mean(residual_losses))
                logger.dump_tabular()

    def _preproc_inputs(self, obs, g):
        f = self.env.extract_features(obs, g)
        f_norm = self.f_norm.normalize(f)
        inputs = f_norm
        inputs = torch.as_tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

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

        # Compute targets by setting sim state and initiating search
        num_per_worker = self.args.batch_size // self.args.num_polo_workers
        count = 0
        residual_target_id = ray.put(self.residual_target.state_dict())
        results = []
        for i in range(self.args.num_polo_workers):
            self.workers[i].set_residual.remote(residual_target_id)
            self.workers[i].set_normalizer.remote(
                self.f_norm.mean, self.f_norm.std)
            if i == self.args.num_polo_workers - 1:
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

    def get_residual_target(self, observation):
        obs = observation['observation']
        g = observation['desired_goal']
        inputs_tensor = self._preproc_inputs(obs, g)
        with torch.no_grad():
            residual_tensor = self.residual_target(inputs_tensor)
            residual = residual_tensor.detach().cpu().numpy().squeeze()
        return residual

    def eval_agent(self):
        total_success_rate = []
        self.controller.reconfigure_heuristic(self.get_residual)
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
