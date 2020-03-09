import numpy as np
import random
import torch

def multi_append(list_of_lists, list_of_values):
    for i, l in enumerate(list_of_lists):
        l.append(list_of_values[i])

def convert_to_np_array(list_of_lists):
    list_of_arrays = []
    for l in list_of_lists:
        list_of_arrays.append(np.array(l))

    return np.array(list_of_arrays)

def convert_to_list_of_np_arrays(list_of_lists):
    list_of_arrays = []
    for l in list_of_lists:
        list_of_arrays.append(np.array(l))

    return list_of_arrays

def multi_merge(list_of_lists, list_of_values):
    for i, l in enumerate(list_of_lists):
        l += list_of_values[i]

    return list_of_lists

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
