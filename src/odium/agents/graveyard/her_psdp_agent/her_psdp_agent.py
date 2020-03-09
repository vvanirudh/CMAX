import os
from datetime import datetime
import torch
import numpy as np
from mpi4py import MPI
from odium.utils.mpi_utils.mpi_utils import sync_networks, sync_grads, sync_all_networks
from odium.agents.her_psdp_agent.replay_buffer import residual_replay_buffer
from odium.agents.her_psdp_agent.models import residualactor, critic, residualcritic
from odium.utils.mpi_utils.normalizer import normalizer
from odium.agents.her_psdp_agent.her_sampler import residual_her_sampler
import odium.utils.logger as logger

"""
ddpg with HER (MPI-version)
PSDP version

"""


class her_psdp_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.T = self.env_params['max_timesteps']
        # create the network
        # Create T actors
        self.actor_networks = [residualactor(
            env_params) for _ in range(self.T)]
        self.critic_networks = [residualcritic(
            env_params) for _ in range(self.T)]
        # sync the networks across the cpus
        sync_all_networks(self.actor_networks)
        sync_all_networks(self.critic_networks)
        # build up the target network
        # if use gpu
        if self.args.cuda:
            _ = [self.actor_networks[i].cuda() for i in range(self.T)]
            _ = [self.critic_networks[i].cuda() for i in range(self.T)]
        # create the optimizer
        # Create T optimizers
        self.actor_optims = [torch.optim.Adam(self.actor_networks[i].parameters(
        ), lr=self.args.lr_actor) for i in range(self.T)]
        self.critic_optims = [torch.optim.Adam(self.critic_networks[i].parameters(
        ), lr=self.args.lr_critic) for i in range(self.T)]
        # her sampler
        self.her_module = residual_her_sampler(
            self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = residual_replay_buffer(
            self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(
            size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(
            size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(
                self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        logger.info("initialized agent")

    def learn(self):
        """
        train the network

        """
        logger.info("Training..")
        n_psdp_iters = 0
        n_steps = 0
        best_success_rate = 0.
        epoch = 0
        success_rate = self._eval_agent()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[{}] epoch is: {}, Num steps: {}, eval success rate is: {:.3f}'.format(
                datetime.now(), epoch, n_steps, success_rate))
        # start to collect samples
        #assert self.args.n_cycles == self.T, "Number of cycles should be equal to horizon"
        #actor_losses, prev_actor_losses = [0.], [0.]
        #critic_losses, prev_critic_losses = [0.], [0.]
        current_t = self.T
        for epoch in range(self.args.n_epochs):
            # TODO: Burn-in critic?
            #prev_actor_losses = actor_losses
            #prev_critic_losses = critic_losses
            actor_losses = []
            critic_losses = []
            if epoch % 10 == 0:
                current_t = current_t - 1
                logger.info(
                    "Training residual policy at time step {}".format(current_t))
            # TODO: Update actors one at a time by monitoring corresponding critic loss
            # Once the critic has been sufficiently trained, then we can start training the actor
            # at that time-step before moving onto the next time-step
            for _ in range(self.args.n_cycles):
                # current_t -= 1
                # if (current_t + 1) % 10 == 0:
                #     logger.info(
                #         "Training residual policy at time step {}".format(current_t))
                # for current_t in range(self.T-1, -1, -1):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        n_steps += 1
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            if t == current_t:
                                # Use untrained residual policy
                                pi = self.actor_networks[t](input_tensor)
                                action = self._select_actions(pi)
                            # elif t > current_t:
                            else:
                                # Use current trained policy
                                # If it has not been trained, it will predict zeros as
                                # a result of our initialization
                                pi = self.actor_networks[t](input_tensor)
                                action = pi.cpu().numpy().squeeze()
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer(
                    [mb_obs, mb_ag, mb_g, mb_actions], current_t)
                for _ in range(self.args.n_batches):
                    # train the network
                    critic_loss, actor_loss = self._update_network(current_t)
                    critic_losses.append(critic_loss.detach().numpy())
                    actor_losses.append(actor_loss.detach().numpy())
                # soft update
                # self._soft_update_target_network(
                #     self.actor_target_networks[current_t], self.actor_networks[current_t])
            # FIX: No target network updates
            # self._soft_update_target_network(
            #     self.critic_target_network, self.critic_network)
            # self._hard_update_target_network(
            #     self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, Current time step : {}, Num steps: {}, eval success rate is: {:.3f}'.format(
                    datetime.now(), epoch, current_t, n_steps, success_rate))
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('n_steps', n_steps)
                logger.record_tabular('success_rate', success_rate)
                logger.record_tabular('actor_loss', np.mean(actor_losses))
                logger.record_tabular('critic_loss', np.mean(critic_losses))
                logger.dump_tabular()
                if success_rate > best_success_rate:
                    logger.info("Better success rate... Saving policy")
                    # torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()],
                    #            self.model_path + '/model.pt')
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std] + [
                               self.actor_networks[t].state_dict() for t in range(self.T)], self.model_path + '/model.pt')
                    best_success_rate = success_rate

    def change_actor_lrs(self, new_lr):
        for t in range(self.T):
            for param_group in self.actor_optims[t].param_groups:
                param_group['lr'] = new_lr

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi, coin_flipping=False):
        action = pi.cpu().numpy().squeeze()
        noise_eps = self.args.noise_eps
        random_eps = self.args.random_eps
        if coin_flipping:
            deterministic_actions = np.random.random() < 0.5
            if deterministic_actions:
                noise_eps = 0.
                random_eps = 0.
        # add the gaussian
        action += noise_eps * \
            self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(
            action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'],
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, random_eps,
                                     1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch, t):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(
            buffer_temp, num_transitions, t)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # hard update
    def _hard_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # update the network
    def _update_network(self, t):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size, t)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(
            o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(
            inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(
            transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            # FIX: Not use target network here
            # actions_next = self.actor_target_networks[t](
            #     inputs_next_norm_tensor)
            if t < self.T - 1:
                actions_next = self.actor_networks[t +
                                                   1](inputs_next_norm_tensor)
                q_next_value = self.critic_networks[t+1](
                    inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
            else:
                actions_next = torch.zeros(
                    inputs_next_norm_tensor.shape[0], self.env_params['action'])
                q_next_value = 0.0

            # FIX: Removing discounting
            # target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = r_tensor + q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            # clip_return = 1 / (1 - self.args.gamma)
            # FIX: Clipping based on horizon
            clip_return = self.T
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_networks[t](
            inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_networks[t](inputs_norm_tensor)
        actor_loss = - \
            self.critic_networks[t](inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * \
            (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optims[t].zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_networks[t])
        self.actor_optims[t].step()
        # update the critic_network
        self.critic_optims[t].zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_networks[t])
        self.critic_optims[t].step()

        return critic_loss, actor_loss

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_networks[t](input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(
            local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
