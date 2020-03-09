import numpy as np


def get_discrepancy_neighbors(observation, action, construct_4d_point, kdtrees, neighbor_radius):
    obs = observation['observation']
    pos, ac_ind = construct_4d_point(obs, action)

    if kdtrees[int(ac_ind)] is None:
        return 0

    num_neighbors = kdtrees[int(ac_ind)].query_radius(
        pos.reshape(1, -1), neighbor_radius, count_only=True)

    return num_neighbors.squeeze()


def apply_discrepancy_penalty(observation, action, rew, discrepancy_fn):
    num_discrepancy_neighbors = discrepancy_fn(observation, np.array(action))
    if num_discrepancy_neighbors > 0:
        rew = -10000

    return rew


def apply_4d_discrepancy_penalty(observation, ac, cost, discrepancy_fn):
    num_discrepancy_neighbors = discrepancy_fn(observation, ac)
    if num_discrepancy_neighbors > 0:
        cost = 100

    return cost
