import numpy as np
from mpi4py import MPI
import odium.utils.logger as logger


class hardcoded_agent:
    def __init__(self, args, env, env_params):
        # Assuming that the env is going to be a residual env
        self.args = args
        self.env = env
        self.env_params = env_params

    def eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            self.env.reset()
            for _ in range(self.env_params['max_timesteps']):
                # convert the actions
                actions = np.zeros(self.env_params['action'])
                observation_new, _, _, info = self.env.step(actions)
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(
            local_success_rate, op=MPI.SUM)
        success_rate = global_success_rate / MPI.COMM_WORLD.Get_size()
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.record_tabular('success_rate', success_rate)
            logger.dump_tabular()
        return success_rate
