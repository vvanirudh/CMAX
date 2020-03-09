import copy
import pickle
import ray
import numpy as np
import torch

from odium.utils.env_utils.make_env import make_env
from odium.utils.controller_utils.get_controller import get_controller
from odium.utils.simple_utils import multi_append
from odium.utils.simulation_utils import set_sim_state_and_goal, apply_dynamics_residual

from odium.agents.fetch_agents.approximators import StateValueResidual, DynamicsResidual, get_state_value_residual, get_next_observation, KNNDynamicsResidual, GPDynamicsResidual
from odium.agents.fetch_agents.normalizer import FeatureNormalizer
from odium.agents.fetch_agents.discrepancy_utils import get_discrepancy_neighbors, apply_discrepancy_penalty


@ray.remote
class InternalRolloutWorker:
    def __init__(self, args, env_params, worker_id=0):
        # Save args
        self.args, self.env_params = args, env_params
        # Env
        self.env_id = args.planning_env_id
        self.env = make_env(env_name=args.env_name,
                            env_id=self.env_id,
                            discrete=True,
                            reward_type=args.reward_type)
        # Set environment seed
        self.env.seed(args.seed + worker_id)
        # Make deterministic, if you have to
        if args.deterministic:
            self.env.make_deterministic()
        # Controller
        self.controller = get_controller(env_name=args.env_name,
                                         num_expansions=args.n_expansions,
                                         # NOTE: Controller can only use internal model
                                         env_id=args.planning_env_id,
                                         discrete=True,
                                         reward_type=args.reward_type,
                                         seed=args.seed+worker_id)
        # State value residual
        self.state_value_residual = StateValueResidual(env_params)
        # KDTrees
        self.kdtrees = [None for _ in range(self.env_params['num_actions'])]
        # Normalizers
        self.features_normalizer = FeatureNormalizer(env_params)
        # Dynamics model
        if self.args.agent == 'mbpo':
            self.residual_dynamics = DynamicsResidual(env_params)
        elif self.args.agent == 'mbpo_knn':
            self.residual_dynamics = [KNNDynamicsResidual(
                args, env_params) for _ in range(self.env_params['num_actions'])]
        else:
            self.residual_dynamics = [GPDynamicsResidual(
                args, env_params) for _ in range(self.env_params['num_actions'])]
            # Flags
        self.kdtrees_set = False
        self.residual_dynamics_set = False

    def set_worker_params(self,
                          value_residual_state_dict,
                          feature_norm_dict,
                          kdtrees_serialized=None,
                          residual_dynamics_state_dict=None):
        # Load all worker parameters
        self.state_value_residual.load_state_dict(value_residual_state_dict)
        self.features_normalizer.load_state_dict(feature_norm_dict)
        # Reconfigure controller heuristic function
        self.controller.reconfigure_heuristic(
            lambda obs: get_state_value_residual(obs,
                                                 self.preproc_inputs,
                                                 self.state_value_residual))
        if kdtrees_serialized:
            self.kdtrees_set = True
            self.kdtrees = pickle.loads(kdtrees_serialized)
            self.controller.reconfigure_discrepancy(
                lambda obs, ac: get_discrepancy_neighbors(obs,
                                                          ac,
                                                          self.construct_4d_point,
                                                          self.kdtrees,
                                                          self.args.neighbor_radius))
        if residual_dynamics_state_dict:
            self.residual_dynamics_set = True
            if self.args.agent == 'mbpo':
                self.residual_dynamics.load_state_dict(
                    residual_dynamics_state_dict)
            elif self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp':
                self.residual_dynamics = pickle.loads(
                    residual_dynamics_state_dict)
            else:
                raise NotImplementedError
            self.controller.reconfigure_residual_dynamics(
                self.get_residual_dynamics)
        return

    def set_num_expansions(self, num_expansions):
        self.controller.reconfigure_num_expansions(num_expansions)
        return

    def preproc_inputs(self, obs, g):
        # Prepare input for the state value residual
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

        s_tensor = torch.as_tensor(s, dtype=torch.float32).view(1, -1)
        a_tensor = torch.as_tensor(ac_ind, dtype=torch.long).view(1, -1)

        return s_tensor, a_tensor

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

    def construct_4d_point(self, obs, ac):
        # Concatenate 2D gripper pos and 2D object pos
        pos = np.concatenate([obs[0:2], obs[3:5]]).reshape(1, -1)
        ac_ind = self.env.discrete_actions[tuple(ac)]
        return pos, ac_ind

    def rollout(self, rollout_length=None, initial_state=None):
        self.env.reset()
        if initial_state:
            # Load initial state if given
            qpos = initial_state['qpos'].copy()
            qvel = initial_state['qvel'].copy()
            goal = initial_state['goal'].copy()
            set_sim_state_and_goal(self.env, qpos, qvel, goal)

        # Data structures
        n_steps = 0
        ep_obs, ep_ag, ep_g, ep_actions, ep_heuristic = [], [], [], [], []
        ep_reward, ep_qpos, ep_qvel, ep_features = [], [], [], []
        ep_penetration = []

        # Start rollout
        observation = self.env.get_obs()
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        features = self.env.extract_features(obs, g)
        heuristic = self.controller.heuristic_obs_g(obs, g)
        if rollout_length is None:
            if self.args.offline:
                rollout_length = self.env_params['offline_max_timesteps']
            else:
                rollout_length = self.env_params['max_timesteps']
        for _ in range(rollout_length):
            qpos = observation['sim_state'].qpos
            qvel = observation['sim_state'].qvel
            ac, _ = self.controller.act(observation)
            ac_ind = self.env.discrete_actions[tuple(ac)]
            observation_new, rew, _, info = self.env.step(ac)
            penetration = info['penetration']
            n_steps += 1
            if self.kdtrees_set:
                assert self.args.agent == 'rts'
                rew = apply_discrepancy_penalty(observation, ac, rew,
                                                self.controller.discrepancy_fn)
            elif self.residual_dynamics_set:
                assert self.args.agent == 'mbpo' or self.args.agent == 'mbpo_knn' or self.args.agent == 'mbpo_gp'
                next_observation, rew = apply_dynamics_residual(self.env,
                                                                self.get_residual_dynamics,
                                                                observation,
                                                                info,
                                                                ac,
                                                                observation_new)
                next_observation['sim_state'] = copy.deepcopy(
                    self.env.env.sim.get_state())
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']
            multi_append([ep_obs, ep_ag, ep_g, ep_actions, ep_heuristic, ep_reward,
                          ep_qpos, ep_qvel, ep_features, ep_penetration],
                         [obs.copy(), ag.copy(), g.copy(), ac_ind, heuristic, rew,
                          qpos.copy(), qvel.copy(), features.copy(), penetration])
            obs = obs_new.copy()
            ag = ag_new.copy()
            observation = copy.deepcopy(observation_new)
            heuristic = self.controller.heuristic_obs_g(obs, g)
            features = self.env.extract_features(obs, g)

        multi_append([ep_obs, ep_ag, ep_heuristic, ep_features],
                     [obs.copy(), ag.copy(), heuristic, features.copy()])

        return ep_obs, ep_ag, ep_g, ep_actions, ep_heuristic, ep_reward, ep_qpos, ep_qvel, ep_features, ep_penetration, n_steps

    def do_rollouts(self, num_rollouts=1, rollout_length=None, initial_state=None):
        # Data structures
        mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic = [], [], [], [], []
        mb_reward, mb_qpos, mb_qvel, mb_features = [], [], [], []
        mb_penetration = []
        mb_n_steps = 0
        for _ in range(num_rollouts):
            ep_obs, ep_ag, ep_g, ep_actions, ep_heuristic, ep_reward, ep_qpos, ep_qvel, ep_features, ep_penetration, n_steps = self.rollout(
                rollout_length, initial_state)
            multi_append([mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
                          mb_reward, mb_qpos, mb_qvel, mb_features, mb_penetration],
                         [ep_obs, ep_ag, ep_g, ep_actions, ep_heuristic,
                          ep_reward, ep_qpos, ep_qvel, ep_features, ep_penetration])
            mb_n_steps += n_steps

        return [mb_obs, mb_ag, mb_g, mb_actions, mb_heuristic,
                mb_reward, mb_qpos, mb_qvel, mb_features, mb_n_steps, mb_penetration]

    def lookahead(self, obs, g, ag, qpos, qvel):
        observation = {}
        observation['observation'] = obs.copy()
        observation['desired_goal'] = g.copy()
        observation['achieved_goal'] = ag.copy()
        observation['sim_state'] = copy.deepcopy(self.env.env.sim.get_state())
        observation['sim_state'].qpos[:] = qpos.copy()
        observation['sim_state'].qvel[:] = qvel.copy()

        _, info = self.controller.act(observation)
        return info

    def lookahead_batch(self, obs, g, ag, qpos, qvel):
        infos = []
        batch_size = obs.shape[0]
        for i in range(batch_size):
            info = self.lookahead(obs[i], g[i], ag[i], qpos[i], qvel[i])
            infos.append(info)

        return infos

    def evaluate(self, num_rollouts=1):
        total_success_rate = []
        total_return = []
        for _ in range(num_rollouts):
            per_success_rate = []
            current_return = 0
            observation = self.env.reset()
            for _ in range(self.env_params['offline_max_timesteps']):
                ac, _ = self.controller.act(observation)
                observation, rew, _, info = self.env.step(ac)
                if self.args.render:
                    self.env.render()
                per_success_rate.append(info['is_success'])
                current_return += rew
            total_success_rate.append(per_success_rate)
            total_return.append(current_return)
        return [total_success_rate, total_return]


def set_workers_num_expansions(list_of_remote_workers, num_expansions):
    results = []
    for remote_worker in list_of_remote_workers:
        results.append(remote_worker.set_num_expansions.remote(num_expansions))

    ray.get(results)
    return
