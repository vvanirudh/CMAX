import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np


class StateValueResidual(nn.Module):
    def __init__(self, env_params):
        '''
        Approximator for the residual on state value function
        '''
        super(StateValueResidual, self).__init__()
        # Save args
        self.env_params = env_params

        # Hidden layer size
        self.hidden = 64

        # Layers
        self.fc1 = nn.Linear(env_params['num_features'], self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.v_out = nn.Linear(self.hidden, 1)
        # Initialize last layer weights to be zero
        self.v_out.weight.data = torch.zeros(1, self.hidden)
        self.v_out.bias.data = torch.zeros(1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.v_out(x)

        return value


class StateActionValueResidual(nn.Module):
    def __init__(self, env_params):
        '''
        Approximator for the residual on state-action value function
        '''
        super(StateActionValueResidual, self).__init__()
        # Save args
        self.env_params = env_params

        # Hidden layer size
        self.hidden = 64

        # Layers
        self.fc1 = nn.Linear(env_params['num_features'], self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.q_out = nn.Linear(self.hidden, env_params['num_actions'])
        # Initialize last layer weights to be zero
        self.q_out.weight.data = torch.zeros(
            env_params['num_actions'], self.hidden)
        self.q_out.bias.data = torch.zeros(env_params['num_actions'])

    def forward(self, x):
        q_x = F.relu(self.fc1(x))
        q_x = F.relu(self.fc2(q_x))
        q_x = F.relu(self.fc3(q_x))
        q_values = self.q_out(q_x)
        return q_values


class Dynamics(nn.Module):
    def __init__(self, env_params):
        '''
        Approximator for the dynamics
        '''
        super(Dynamics, self).__init__()
        # Save args
        self.env_params = env_params

        # Hidden layer size
        self.hidden = 64

        # State size
        self.state_size = 2 + 2  # Object (x, y) and Gripper (x, y)
        # Action size
        self.action_size = env_params['num_actions']

        # Layers
        self.fc1 = nn.Linear(self.state_size, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.state_out = nn.Linear(
            self.hidden, self.action_size * self.state_size)
        self.state_out.weight.data = torch.zeros(
            self.action_size * self.state_size, self.hidden)
        self.state_out.bias.data = torch.zeros(
            self.action_size * self.state_size)

    def forward(self, s, a):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = self.state_out(x)
        # How to gather the right slices
        indices = torch.zeros(x.shape[0], self.state_size, dtype=torch.long)
        for row in range(x.shape[0]):
            indices[row] = torch.arange(
                a[row].item() * self.state_size, (a[row].item()+1) * self.state_size)
        x = torch.gather(x, 1, indices)
        return x


class DynamicsResidual(nn.Module):
    def __init__(self, env_params):
        '''
        Approximator for the residual on state-action transition dynamics
        '''
        super(DynamicsResidual, self).__init__()
        # Save args
        self.env_params = env_params

        # Hidden layer size
        self.hidden = 32

        # Input and output size
        self.input_dim, self.output_dim = 4 + self.env_params['num_actions'], 4

        # Layers
        self.fc1 = nn.Linear(
            self.input_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.dyn_out = nn.Linear(self.hidden, self.output_dim)
        # Initialize last layer weights to be zero
        self.dyn_out.weight.data = torch.zeros(
            self.output_dim, self.hidden)
        self.dyn_out.bias.data = torch.zeros(self.output_dim)

    def forward(self, s, a):
        one_hot_encoding = torch.zeros(
            a.shape[0], self.env_params['num_actions'])
        for row in range(a.shape[0]):
            one_hot_encoding[row, a[row]] = 1
        x = torch.cat([s, one_hot_encoding], dim=1)
        dyn_x = F.relu(self.fc1(x))
        dyn_x = F.relu(self.fc2(dyn_x))
        dyn_values = self.dyn_out(dyn_x)
        return dyn_values


class KNNDynamicsResidual:
    def __init__(self, args, env_params):
        # Save args
        self.args, self.env_params = args, env_params
        # Create the KNN model
        self.knn_model = RadiusNeighborsRegressor(radius=args.neighbor_radius,
                                                  weights='uniform')
        # Flag
        self.is_fit = False

    def fit(self, X, y):
        '''
        X should be the data matrix N x d, where each row is a 4D vector
        consisting of object pos and gripper pos
        y should be target matrix N x d, where each row is a 4D vector 
        consisting of next object pos and next gripper pos
        '''
        self.knn_model.fit(X, y)
        self.is_fit = True
        return self.loss(X, y)

    def predict(self, X):
        '''
        X should be the data matrix N x d, where each row is a 4D vector
        consisting of object pos and gripper pos
        '''
        ypred = np.zeros(X.shape)
        if not self.is_fit:
            # KNN model is not fit
            return ypred
        # Get neighbors of X
        neighbors = self.knn_model.radius_neighbors(X)
        # Check if any of the X doesn't have any neighbors by getting nonzero mask
        neighbor_mask = [x.shape[0] != 0 for x in neighbors[1]]
        # If none of X has any neighbors
        if X[neighbor_mask].shape[0] == 0:
            return ypred
        # Else, for the X that have neighbors use the KNN prediction
        ypred[neighbor_mask] = self.knn_model.predict(X[neighbor_mask])
        return ypred

    def get_num_neighbors(self, X):
        if not self.is_fit:
            return np.zeros(X.shape[0])
        neighbors = self.knn_model.radius_neighbors(X)
        num_neighbors = np.array([x.shape[0] for x in neighbors[1]])
        return num_neighbors

    def loss(self, X, y):
        ypred = self.predict(X)
        # Loss is just the mean distance between predictions and true targets
        loss = np.linalg.norm(ypred - y, axis=1).mean()
        return loss


class GPDynamicsResidual:
    def __init__(self, args, env_params):
        # Save args
        self.args, self.env_params = args, env_params

        # Create the GP model
        self.gp_model = GaussianProcessRegressor()

        # Flag
        self.is_fit = False

    def fit(self, X, y):
        self.gp_model.fit(X, y)
        self.is_fit = True
        return self.loss(X, y)

    def predict(self, X):
        ypred = np.zeros(X.shape)
        if not self.is_fit:
            return ypred

        ypred = self.gp_model.sample_y(X).squeeze()
        return ypred

    def loss(self, X, y):
        ypred = self.predict(X)
        loss = np.linalg.norm(ypred - y, axis=1).mean()
        return loss


class Policy(nn.Module):
    def __init__(self, env_params):
        '''
        Approximator for the stochastic policy
        '''
        super(Policy, self).__init__()
        # Save args
        self.env_params = env_params

        # Hidden layer size
        self.hidden = 256

        # Input and output size
        self.input_dim, self.output_dim = env_params['num_features'], env_params['num_actions']

        # Layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.policy_out = nn.Linear(self.hidden, self.output_dim)

    def forward(self, x):
        policy_x = F.relu(self.fc1(x))
        policy_x = F.relu(self.fc2(policy_x))
        policy_x = F.relu(self.fc3(policy_x))
        policy_values = self.policy_out(policy_x)
        return policy_values


def get_state_value_residual(observation, preproc_inputs, state_value_residual):
    obs = observation['observation']
    g = observation['desired_goal']
    inputs_tensor = preproc_inputs(obs, g)
    with torch.no_grad():
        residual_tensor = state_value_residual(inputs_tensor)
        residual = residual_tensor.detach().cpu().numpy().squeeze()

    return residual


def get_dynamics_residual(observation, action,
                          preproc_dynamics_inputs, dynamics_residual):
    obs = observation['observation']
    inputs_tensor = preproc_dynamics_inputs(obs, action)
    with torch.no_grad():
        dynamics_residual_tensor = dynamics_residual(inputs_tensor)
        dynamics_residual = dynamics_residual_tensor.detach().cpu().numpy().squeeze()

    return dynamics_residual


def get_knn_dynamics_residual(observation, action,
                              preproc_knn_dynamics_inputs, knn_dynamics_residuals):
    obs = observation['observation']
    pos, ac_ind = preproc_knn_dynamics_inputs(obs, action)
    # residual = knn_dynamics_residuals[int(ac_ind)].predict(pos)
    residual = knn_dynamics_residuals[int(ac_ind)].get_num_neighbors(pos)
    return residual.squeeze()


def get_next_observation(observation, ac, preproc_dynamics_inputs, residual_dynamics):
    obs = observation['observation']
    s_tensor, a_tensor = preproc_dynamics_inputs(obs, ac)
    obs_next = residual_dynamics(
        s_tensor, a_tensor).detach().cpu().numpy().squeeze()
    return obs_next
