import numpy as np


class Sampler:
    def __init__(self, args, reward_fn, heuristic_fn, extract_features_fn):
        self.replay_strategy = args.replay_strategy
        self.replay_k = args.replay_k
        self.args = args

        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + self.replay_k))
            # raise Exception('HER is not applicable for this setup!')
        else:
            self.future_p = 0
        self.reward_func = reward_fn
        self.heuristic_func = heuristic_fn
        self.extract_features_func = extract_features_fn

    def sample(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy(
        ) for key in episode_batch.keys()}
        # her idx
        if self.args.her:
            her_indexes = np.where(np.random.uniform(
                size=batch_size) < self.future_p)
            future_offset = np.random.uniform(
                size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            # replace go with achieved goal
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag
            # to get the params to re-compute reward
            transitions['r'] = np.expand_dims(self.reward_func(
                transitions['ag_next'], transitions['g'], None), 1)

            new_s_h = np.array([self.heuristic_func(
                transitions['obs'][i], transitions['g'][i]) for i in range(batch_size)])
            new_s_h_next = np.array([self.heuristic_func(
                transitions['obs_next'][i], transitions['g'][i]) for i in range(batch_size)])

            new_f = np.array([self.extract_features_func(transitions['obs'][i], transitions['g'][i]) for i in range(batch_size)])
            new_f_next = np.array([self.extract_features_func(transitions['obs_next'][i], transitions['g'][i]) for i in range(batch_size)])

            transitions['s_h'] = np.expand_dims(new_s_h, 1)
            transitions['s_h_next'] = np.expand_dims(new_s_h_next, 1)
            transitions['f'] = new_f
            transitions['f_next'] = new_f_next

        transitions = {k: transitions[k].reshape(
            batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
