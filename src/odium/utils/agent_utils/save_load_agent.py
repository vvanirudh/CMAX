import torch


def save_agent(path,
               network_state_dict,
               optimizer_state_dict=None,
               normalizer_state_dict=None,
               knn_dynamics_residuals_serialized=None,
               dynamics_state_dict=None,
               dynamics_optimizer_state_dict=None,
               epoch=None,
               success_rate=None):
    # Save model
    save_dict = {}
    save_dict['network_state_dict'] = network_state_dict
    if optimizer_state_dict:
        save_dict['optimizer_state_dict'] = optimizer_state_dict
    if normalizer_state_dict:
        save_dict['normalizer_state_dict'] = normalizer_state_dict
    if knn_dynamics_residuals_serialized:
        save_dict['knn_dynamics_residuals_serialized'] = knn_dynamics_residuals_serialized
    if epoch:
        save_dict['epoch'] = epoch
    if success_rate:
        save_dict['success_rate'] = success_rate
    if dynamics_state_dict:
        save_dict['dynamics_state_dict'] = dynamics_state_dict
    if dynamics_optimizer_state_dict:
        save_dict['dynamics_optimizer_state_dict'] = dynamics_optimizer_state_dict

    torch.save(save_dict, path)
    return True


def load_agent(path):
    # Load model
    load_dict = torch.load(path)

    return load_dict, load_dict.keys()
