import numpy as np
import ray
import torch
from odium.utils.controller_utils.get_controller import get_controller
from odium.utils.env_utils.make_env import make_env
from odium.agents.ilc_rts_agent.models import Residual, DynamicsResidual
from odium.utils.mpi_utils.normalizer import normalizer


@ray.remote
class Worker:
    def __init__(self, args, env_params):
        self.controller = get_controller(args.env_name, env_id=args.planning_env_id, discrete=True,
                                         num_expansions=args.offline_num_expansions, reward_type=args.reward_type)
        self.residual = Residual(env_params)
        self.dynamics_residual = DynamicsResidual(env_params)
        self.env = make_env(args.env_name, args.planning_env_id,
                            discrete=True, reward_type=args.reward_type)
        self.f_norm = normalizer(env_params['num_features'])
        self.pos_norm = normalizer(size=4)
        self.dummy_sim_state = self.env.reset()['sim_state']
        self.env_params = env_params

    def set_residual(self, residual_target_state_dict):
        self.residual.load_state_dict(residual_target_state_dict)
        self.controller.reconfigure_heuristic(self.get_residual)

    def set_dynamics_residual(self, residual_dynamics_state_dict):
        self.dynamics_residual.load_state_dict(residual_dynamics_state_dict)
        self.controller.reconfigure_dynamics(self.get_dynamics_residual)

    def set_feature_normalizer(self, mean, std):
        self.f_norm.load(mean, std)

    def set_dynamics_normalizer(self, mean, std):
        self.pos_norm.load(mean, std)

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

    def _preproc_inputs(self, obs, g):
        f = self.env.extract_features(obs, g)
        f_norm = self.f_norm.normalize(f)

        inputs = f_norm
        inputs = torch.as_tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def _preproc_dynamics_inputs(self, obs, ac):
        pos = np.concatenate([obs[0:2], obs[3:5]])
        pos_norm = self.pos_norm.normalize(pos)
        ac_input = np.zeros(self.env_params['num_actions'])
        ac_ind = self.env.discrete_actions[tuple(ac)]
        ac_input[ac_ind] = 1
        inputs = np.concatenate([pos_norm, ac_input])
        inputs_tensor = torch.as_tensor(
            inputs, dtype=torch.float32).unsqueeze(0)
        return inputs_tensor

    def act(self, obs, g, ag, qpos, qvel):
        observation = {}
        observation['observation'] = obs.copy()
        observation['desired_goal'] = g.copy()
        observation['achieved_goal'] = ag.copy()
        c_qpos, c_qvel = qpos.copy(), qvel.copy()
        observation['sim_state'] = self.dummy_sim_state
        observation['sim_state'] = observation['sim_state']._replace(
            qpos=c_qpos)
        observation['sim_state'] = observation['sim_state']._replace(
            qvel=c_qvel)

        _, info = self.controller.act(observation)
        return info['best_node_f']

    def act_batch(self, obs, g, ag, qpos, qvel):
        values = []
        batch_size = obs.shape[0]
        for i in range(batch_size):
            value = self.act(obs[i], g[i], ag[i], qpos[i], qvel[i])
            values.append(value)

        return values
