import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network


class actor(nn.Module):
    def __init__(self, env_params, residual=False):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        if residual:
            # Intialize last layer weights to be zero
            self.action_out.weight.data = torch.zeros(
                env_params['action'], 256)
            self.action_out.bias.data = torch.zeros(env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(
            env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class switch(nn.Module):
    def __init__(self, env_params, args):
        super(switch, self).__init__()
        self.dueling = args.dueling
        num_switch_actions = 2  # Hardcoded or learned
        self.fc1 = nn.Linear(
            env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        # 0 is hardcoded, 1 is learned
        self.q_out = nn.Linear(256, num_switch_actions)
        if self.dueling:
            self.v_fc1 = nn.Linear(
                env_params['obs'] + env_params['goal'], 256)
            self.v_fc2 = nn.Linear(256, 256)
            self.v_fc3 = nn.Linear(256, 256)
            self.v_out = nn.Linear(256, 1)

    def forward(self, x):
        # x = torch.cat([x, actions / self.max_action], dim=1)
        q_x = F.relu(self.fc1(x))
        q_x = F.relu(self.fc2(q_x))
        q_x = F.relu(self.fc3(q_x))
        q_value = self.q_out(q_x)
        if self.dueling:
            v_x = F.relu(self.v_fc1(x))
            v_x = F.relu(self.v_fc2(v_x))
            v_x = F.relu(self.v_fc3(v_x))
            v_value = self.v_out(v_x)
            # Center q values
            q_value_mean = torch.mean(q_value, dim=1, keepdim=True)
            q_value_center = q_value - q_value_mean
            # Add predicted value to advantage
            # TODO: This value prediction network can be shared with DDPG critic
            q_value = v_value + q_value_center
        return q_value


class critic_with_switch(nn.Module):
    def __init__(self, env_params):
        super(critic_with_switch, self).__init__()
        # NOTE: Dueling enabled by default
        self.max_action = env_params['action_max']
        self.num_switch_actions = 2  # Hardcoded or learned
        self.fc1 = nn.Linear(
            env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256 + env_params['action'], 256)
        self.fc2_switch = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc3_switch_value = nn.Linear(256, 128)
        self.fc3_switch_adv = nn.Linear(256, 128)
        self.q_out = nn.Linear(256, 1)
        self.value_switch = nn.Linear(128, 1)
        self.adv_switch = nn.Linear(128, self.num_switch_actions)

    def forward(self, x, actions, value=False):
        h_x = F.relu(self.fc1(x))
        # Critic
        critic_x = torch.cat([h_x, actions / self.max_action], dim=1)
        critic_x = F.relu(self.fc2(critic_x))
        critic_x = F.relu(self.fc3(critic_x))
        critic_out = self.q_out(critic_x)
        # Switch
        switch_x = F.relu(self.fc2_switch(h_x))
        switch_x_value = F.relu(self.fc3_switch_value(switch_x))
        switch_x_adv = F.relu(self.fc3_switch_adv(switch_x))
        switch_out_value = self.value_switch(switch_x_value)
        switch_out_adv = self.adv_switch(switch_x_adv)

        switch_out_adv_avg = torch.mean(switch_out_adv, dim=1, keepdim=True)
        switch_out_adv_centered = switch_out_adv - switch_out_adv_avg
        switch_out = switch_out_value + switch_out_adv_centered

        if value:
            return critic_out, switch_out, switch_out_value

        return critic_out, switch_out
