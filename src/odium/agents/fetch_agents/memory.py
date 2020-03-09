import threading
import numpy as np


class rts_memory:
    def __init__(self, args, env_params, sampler):
        '''
        args - arguments
        env_params - environment parameters
        sampler - memory sampler
        '''
        # Store arguments
        self.args, self.env_params, self.sampler = args, env_params, sampler

        # Get relevant args
        self.buffer_size = args.buffer_size
        if args.offline:
            # Offline
            self.T = env_params['offline_max_timesteps']
        else:
            # Online
            self.T = env_params['max_timesteps']
        self.size = self.buffer_size // self.T
        self.n_internal_model_transitions_stored, self.n_real_world_transitions_stored = 0, 0
        self.current_size_internal_model, self.current_size_real_world = 0, 0

        # Buffers
        self.internal_model_buffers = {
            'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
            'g': np.empty([self.size, self.T, self.env_params['goal']]),
            'actions': np.empty([self.size, self.T]),
            'heuristic': np.empty([self.size, self.T + 1]),
            'r': np.empty([self.size, self.T]),
            'qpos': np.empty([self.size, self.T, self.env_params['qpos']]),
            'qvel': np.empty([self.size, self.T, self.env_params['qvel']]),
            'features': np.empty([self.size, self.T + 1, self.env_params['num_features']]),
            'qvalues': np.empty([self.size, self.T + 1, self.env_params['num_actions']]),
            'penetration': np.empty([self.size, self.T]),
        }

        self.real_world_buffers = {
            'obs': np.empty([self.buffer_size, self.env_params['obs']]),
            'g': np.empty([self.buffer_size, self.env_params['goal']]),
            'actions': np.empty([self.buffer_size]),
            'qpos': np.empty([self.buffer_size, self.env_params['qpos']]),
            'qvel': np.empty([self.buffer_size, self.env_params['qvel']]),
            'obs_next': np.empty([self.buffer_size, self.env_params['obs']]),
            'obs_sim_next': np.empty([self.buffer_size, self.env_params['obs']])
        }

        # Lock
        self.lock = threading.Lock()

    def _get_storage_idx(self, current_size, total_size, inc=None):
        '''
        Taken directly from HER code
        '''
        inc = inc or 1
        if current_size+inc <= total_size:
            idx = np.arange(current_size, current_size+inc)
        elif current_size < total_size:
            overflow = inc - (total_size - current_size)
            idx_a = np.arange(current_size, total_size)
            idx_b = np.random.randint(0, current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, total_size, inc)
        new_current_size = min(total_size, current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx, new_current_size

    def clear(self, internal=True, real=True):
        # Clear buffers
        self.n_internal_model_transitions_stored, self.n_real_world_transitions_stored = 0, 0
        self.current_size_internal_model, self.current_size_real_world = 0, 0
        with self.lock:
            if internal:
                for key in self.internal_model_buffers.keys():
                    self.internal_model_buffers[key] = np.empty(
                        self.internal_model_buffers[key].shape)
            if real:
                for key in self.real_world_buffers.keys():
                    self.real_world_buffers[key] = np.empty(
                        self.real_world_buffers[key].shape)

        return True

    def store_internal_model_rollout(self, episode_batch, qvalue=False):
        # Store episode in buffers
        if not qvalue:
            obs, ag, g, actions, heuristic, r, qpos, qvel, features, penetration = episode_batch
        else:
            obs, ag, g, actions, heuristic, r, qpos, qvel, features, qvalues, penetration = episode_batch
        batch_size = obs.shape[0]
        with self.lock:
            idxs, self.current_size_internal_model = self._get_storage_idx(
                self.current_size_internal_model, self.size, inc=batch_size)
            self.internal_model_buffers['obs'][idxs] = obs
            self.internal_model_buffers['ag'][idxs] = ag
            self.internal_model_buffers['g'][idxs] = g
            self.internal_model_buffers['actions'][idxs] = actions
            self.internal_model_buffers['heuristic'][idxs] = heuristic
            self.internal_model_buffers['r'][idxs] = r
            self.internal_model_buffers['qpos'][idxs] = qpos
            self.internal_model_buffers['qvel'][idxs] = qvel
            self.internal_model_buffers['features'][idxs] = features
            if qvalue:
                self.internal_model_buffers['qvalues'][idxs] = qvalues
            self.internal_model_buffers['penetration'][idxs] = penetration
            self.n_internal_model_transitions_stored += self.T * batch_size

        return True

        # DEPRECATED
        # def store_real_world_rollout(self, episode_batch):
        #     # Store episode in buffers
        #     obs, actions, qpos, qvel, obs_sim_next = episode_batch
        #     batch_size = obs.shape[0]
        #     with self.lock:
        #         idxs, self.current_size_real_world = self._get_storage_idx(
        #             self.current_size_real_world, inc=batch_size)
        #         self.real_world_buffers['obs'][idxs] = obs
        #         self.real_world_buffers['actions'][idxs] = actions
        #         self.real_world_buffers['qpos'][idxs] = qpos
        #         self.real_world_buffers['qvel'][idxs] = qvel
        #         self.real_world_buffers['obs_sim_next'][idxs] = obs_sim_next
        #         self.n_real_world_transitions_stored += self.T * batch_size

        #     return True

    def store_real_world_transition(self, transition):
        # Store a single transition in buffers
        obs, g, action, qpos, qvel, obs_next, obs_sim_next = transition
        with self.lock:
            idxs, self.current_size_real_world = self._get_storage_idx(
                self.current_size_real_world, self.buffer_size, inc=1)
            self.real_world_buffers['obs'][idxs] = obs
            self.real_world_buffers['g'][idxs] = g
            self.real_world_buffers['actions'][idxs] = action
            self.real_world_buffers['obs_next'][idxs] = obs_next
            self.real_world_buffers['qpos'][idxs] = qpos
            self.real_world_buffers['qvel'][idxs] = qvel
            self.real_world_buffers['obs_sim_next'][idxs] = obs_sim_next
            self.n_real_world_transitions_stored += 1

        return True

    def sample_internal_world_memory(self, batch_size, qvalue=False):
        # Create a copy of buffers
        temp_buffers = {}
        with self.lock:
            for key in self.internal_model_buffers.keys():
                temp_buffers[key] = self.internal_model_buffers[key][:self.current_size_internal_model]

        # Create obs_next, ag_next, heuristic_next
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        temp_buffers['heuristic_next'] = temp_buffers['heuristic'][:, 1:]
        temp_buffers['features_next'] = temp_buffers['features'][:, 1:, :]
        if qvalue:
            temp_buffers['qvalues_next'] = temp_buffers['qvalues'][:, 1:, :]

        # Sample transitions
        transitions = self.sampler.sample(temp_buffers, batch_size)
        return transitions

    def sample_real_world_memory(self, batch_size=None):
        # Create a copy of buffers
        temp_buffers = {}
        with self.lock:
            for key in self.real_world_buffers.keys():
                temp_buffers[key] = self.real_world_buffers[key][:self.current_size_real_world]

        # Sample transitions uniformly at random
        if batch_size:
            transition_idxs = np.random.randint(
                0, self.current_size_real_world, batch_size)
        else:
            # If no batch size is given, return all transitions
            transition_idxs = np.arange(0, self.current_size_real_world)
            batch_size = self.current_size_real_world

        transitions = {key: temp_buffers[key][transition_idxs].copy(
        ) for key in temp_buffers.keys()}

        transitions = {k: transitions[k].reshape(
            batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
