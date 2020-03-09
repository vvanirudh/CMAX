import threading
import numpy as np


def rolling_sum(arr, n):
    ret = np.cumsum(np.flip(arr, -1), axis=1)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return np.flip(ret, 1)


class Dataset:
    def __init__(self, args, env_params, sampler):
        self.env_params = env_params
        self.sampler = sampler
        self.buffer_size = args.buffer_size
        self.T = env_params['max_timesteps']
        self.size = self.buffer_size // self.T
        self.n_transitions_stored = 0

        self.current_size = 0

        self.buffers = {
            'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
            'g': np.empty([self.size, self.T, self.env_params['goal']]),
            # FIX: Storing action indices
            'actions': np.empty([self.size, self.T]),
            's_h': np.empty([self.size, self.T + 1]),
            'r': np.empty([self.size, self.T]),
            'qpos': np.empty([self.size, self.T, self.env_params['qpos']]),
            'qvel': np.empty([self.size, self.T, self.env_params['qvel']]),
            'f': np.empty([self.size, self.T + 1, self.env_params['num_features']])
        }

        self.lock = threading.Lock()

    def store_episode(self, episode_batch):
        obs, ag, g, actions, s_h, r, qpos, qvel, f = episode_batch
        batch_size = obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            self.buffers['obs'][idxs] = obs
            self.buffers['ag'][idxs] = ag
            self.buffers['g'][idxs] = g
            self.buffers['actions'][idxs] = actions
            self.buffers['s_h'][idxs] = s_h
            self.buffers['r'][idxs] = r
            self.buffers['qpos'][idxs] = qpos
            self.buffers['qvel'][idxs] = qvel
            self.buffers['f'][idxs] = f
            self.n_transitions_stored += self.T * batch_size

    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]

        # temp_buffers['obs_next'] = np.append(temp_buffers['obs'][:, self.lookahead:, :],
        #                                      np.tile(
        #                                          np.expand_dims(temp_buffers['obs'][:, -1, :],
        #                                                         axis=1),
        #     (1, self.lookahead - 1, 1)),
        #     axis=1)
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]

        # temp_buffers['r'] = rolling_sum(temp_buffers['r'], n=self.lookahead)

        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

        temp_buffers['s_h_next'] = temp_buffers['s_h'][:, 1:]

        # sample transitions
        transitions = self.sampler.sample(
            temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def clear(self):
        self.n_transitions_stored = 0

        self.current_size = 0

        with self.lock:
            for key in self.buffers.keys():
                self.buffers[key] = np.empty(self.buffers[key].shape)


class DynamicsDataset:
    def __init__(self, args, env_params):
        self.env_params = env_params
        self.args = args
        self.buffer_size = args.buffer_size
        self.T = env_params['max_timesteps']
        self.size = self.buffer_size // self.T
        self.n_transitions_stored = 0
        self.current_size = 0

        self.buffers = {
            'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
            'actions': np.empty([self.size, self.T, self.env_params['action']]),
            'qpos': np.empty([self.size, self.T, self.env_params['qpos']]),
            'qvel': np.empty([self.size, self.T, self.env_params['qvel']]),
            'obs_model_next': np.empty([self.size, self.T, self.env_params['obs']])
        }

        self.lock = threading.Lock()

    def store_episode(self, episode_batch):
        obs, actions, qpos, qvel, obs_model_next = episode_batch
        batch_size = obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            self.buffers['obs'][idxs] = obs
            self.buffers['actions'][idxs] = actions
            self.buffers['qpos'][idxs] = qpos
            self.buffers['qvel'][idxs] = qvel
            self.buffers['obs_model_next'][idxs] = obs_model_next
            self.n_transitions_stored += self.T * batch_size

    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]

        episode_idxs = np.random.randint(0, self.current_size, batch_size)
        t_samples = np.random.randint(self.T, size=batch_size)
        transitions = {key: temp_buffers[key][episode_idxs, t_samples].copy(
        ) for key in temp_buffers.keys()}

        transitions = {k: transitions[k].reshape(
            batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def clear(self):
        self.n_transitions_stored = 0

        self.current_size = 0

        with self.lock:
            for key in self.buffers.keys():
                self.buffers[key] = np.empty(self.buffers[key].shape)
