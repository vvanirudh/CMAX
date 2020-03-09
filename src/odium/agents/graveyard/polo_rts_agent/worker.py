import ray
import torch
from odium.utils.controller_utils.get_controller import get_controller
from odium.utils.env_utils.make_env import make_env
from odium.agents.polo_rts_agent.models import Residual
from odium.utils.mpi_utils.normalizer import normalizer


@ray.remote
class Worker:
    def __init__(self, args, env_params):
        self.controller = get_controller(args.env_name, env_id=args.env_id, discrete=True,
                                         num_expansions=args.offline_num_expansions, reward_type=args.reward_type)
        self.residual = Residual(env_params)
        self.env = make_env(args.env_name, args.env_id,
                            discrete=True, reward_type=args.reward_type)
        self.f_norm = normalizer(env_params['num_features'])
        self.dummy_sim_state = self.env.reset()['sim_state']

    def set_residual(self, residual_target_state_dict):
        self.residual.load_state_dict(residual_target_state_dict)
        self.controller.reconfigure_heuristic(self.get_residual)

    def set_normalizer(self, mean, std):
        self.f_norm.load(mean, std)

    def get_residual(self, observation):
        obs = observation['observation']
        g = observation['desired_goal']
        inputs_tensor = self._preproc_inputs(obs, g)
        with torch.no_grad():
            residual_tensor = self.residual(inputs_tensor)
            residual = residual_tensor.detach().cpu().numpy().squeeze()
        return residual

    def _preproc_inputs(self, obs, g):
        f = self.env.extract_features(obs, g)
        f_norm = self.f_norm.normalize(f)

        inputs = f_norm
        inputs = torch.as_tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

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
