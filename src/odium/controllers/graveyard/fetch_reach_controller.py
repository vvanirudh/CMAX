import numpy as np

from odium.controllers.controller import Controller


class ReachController(Controller):
    def __init__(self, gain=2.0, discrete=False):
        assert isinstance(gain, float) or isinstance(gain, int)
        self.gain = gain
        self.discrete = discrete

    def act(self, obs):
        achieved_goal = obs['achieved_goal']
        goal = obs['desired_goal']

        action = np.zeros(4)
        action[:3] = self.gain * (goal - achieved_goal)
        return action
