import numpy as np


class rts_sampler:
    def __init__(self, args, reward_fn, heuristic_fn, extract_features_fn):
        # Store arguments
        self.args, self.reward_fn = args, reward_fn
        self.heuristic_fn, self.extract_features_fn = heuristic_fn, extract_features_fn

        # HER
        self.her = args.her
        self.future_p = 1 - (1. / (1 + self.args.replay_k))

    def sample(self, episode_batch, batch_size_in_transitions):
        # Get horizon length
        T = episode_batch['actions'].shape[1]
        # Get memory size
        rollout_batch_size = episode_batch['actions'].shape[0]
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(
            0, rollout_batch_size, size=batch_size_in_transitions)
        t_samples = np.random.randint(T, size=batch_size_in_transitions)
        # Pick the transitions
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Apply HER relabeling if needed
        if self.her:
            transitions = self.her_relabeling(transitions,
                                              batch_size_in_transitions,
                                              t_samples,
                                              episode_idxs,
                                              episode_batch)

        # Reshape transitions
        transitions = {k: transitions[k].reshape(batch_size_in_transitions,
                                                 *transitions[k].shape[1:])
                       for k in transitions.keys()}

        return transitions

    def her_relabeling(self,
                       transitions,
                       batch_size_in_transitions,
                       t_samples,
                       episode_idxs,
                       episode_batch):
        # Get horizon length
        T = episode_batch['actions'].shape[1]
        # Get transitions that will be relabeled
        her_indexes = np.where(np.random.uniform(
            size=batch_size_in_transitions) < self.future_p)
        # Get the offset for goals
        future_offset = np.random.uniform(
            size=batch_size_in_transitions) * (T - t_samples)
        future_offset = future_offset.astype(int)
        # Get future time steps
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # Replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # Recompute rewards
        transitions['r'] = np.expand_dims(
            self.reward_fn(
                transitions['ag_next'], transitions['g'], None, transitions['penetration']),
            1)
        # Recompute heuristics
        transitions['heuristic'] = np.array([self.heuristic_fn(transitions['obs'][i],
                                                               transitions['g'][i])
                                             for i in range(batch_size_in_transitions)])

        transitions['heuristic_next'] = np.array([self.heuristic_fn(transitions['obs_next'][i],
                                                                    transitions['g'][i])
                                                  for i in range(batch_size_in_transitions)])
        # Recompute features
        transitions['features'] = np.array([self.extract_features_fn(transitions['obs'][i],
                                                                     transitions['g'][i])
                                            for i in range(batch_size_in_transitions)])

        transitions['features_next'] = np.array([self.extract_features_fn(transitions['obs_next'][i],
                                                                          transitions['g'][i])
                                                 for i in range(batch_size_in_transitions)])

        return transitions
