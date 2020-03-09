import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, env_params):
        super(Residual, self).__init__()
        self.hidden = 256
        # self.fc1 = nn.Linear(env_params['obs']+env_params['goal'], self.hidden)
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


class ResidualAdvantage(nn.Module):
    def __init__(self, env_params):
        super(ResidualAdvantage, self).__init__()
        self.hidden = 256
        self.fc1 = nn.Linear(env_params['obs']+env_params['goal'], self.hidden)
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
