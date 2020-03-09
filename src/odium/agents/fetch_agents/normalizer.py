import numpy as np
from odium.utils.mpi_utils.normalizer import normalizer


class Normalizer:
    def __init__(self, norm):
        '''
        Base class
        '''
        self.norm = norm

    def normalize(self, x):
        return self.norm.normalize(x)

    def update(self, x):
        return self.norm.update(x)

    def recompute_stats(self):
        return self.norm.recompute_stats()

    def get_mean(self):
        return self.norm.mean.copy()

    def get_std(self):
        return self.norm.std.copy()

    def state_dict(self):
        return {'mean': self.get_mean(),
                'std': self.get_std()}

    def load_state_dict(self, state_dict):
        self.norm.mean = state_dict['mean'].copy()
        self.norm.std = state_dict['std'].copy()


class FeatureNormalizer(Normalizer):
    def __init__(self, env_params):
        '''
        Create a feature normalizer
        '''
        # Save args
        self.env_params = env_params
        # Normalizer
        f_norm = normalizer(size=env_params['num_features'])
        super(FeatureNormalizer, self).__init__(f_norm)

    def update_normalizer(self, episode_batch, sampler):
        mb_obs, mb_ag, mb_g, mb_actions, mb_s_h, mb_r, mb_qpos, mb_qvel, mb_f, mb_penetration = episode_batch
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
                       'heuristic': mb_s_h,
                       'heuristic_next': mb_s_h_next,
                       'r': mb_r,
                       'features': mb_f,
                       'penetration': mb_penetration,
                       }
        buffer_temp['qpos'] = mb_qpos
        buffer_temp['qvel'] = mb_qvel
        transitions = sampler.sample(
            buffer_temp, num_transitions)
        self.update(transitions['features'])
        # recompute the stats
        self.recompute_stats()
        return True


class DynamicsNormalizer(Normalizer):
    def __init__(self, env_params):
        '''
        Create dynamics normalizer
        '''
        # Save args
        self.env_params = env_params
        # Normalizer
        dyn_norm = normalizer(size=4)
        super(DynamicsNormalizer, self).__init__(dyn_norm)

    def update_normalizer(self, episode_batch):
        mb_obs, mb_actions, mb_qpos, mb_qvel, mb_obs_model_next = episode_batch
        obj_pos = np.concatenate([mb_obs[:, 0:2], mb_obs[:, 3:5]], axis=1)
        self.update(obj_pos)
        self.recompute_stats()
        return True
