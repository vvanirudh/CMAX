'''
Wrapper to expose the sim state of the env
ResidualWrapper to expose the residual env
'''
import copy
import numpy as np
import gym

from odium.envs.fetch.push_among_obstacles import FetchPushAmongObstaclesEnv


def get_env(env_name, discrete, reward_type, env_id=None):
    if env_name == 'FetchPushAmongObstacles-v1':
        # Custom defined env
        env = FetchPushAmongObstaclesEnv(env_id=env_id,
                                         reward_type=reward_type,
                                         discrete=discrete)

    else:
        # Gym env
        assert env_id is None, "Env ID not defined for gym environments"
        assert not discrete, "Discrete option not defined for gym environments"
        raise NotImplementedError("Not implemented for gym environments")
        env = gym.make(env_name, reward_type=reward_type)

    return env


class WrapperEnv(gym.Env):

    def __init__(self, env_name, env_id, discrete, reward_type):
        self.env = get_env(env_name, discrete, reward_type, env_id)
        if discrete:
            self.discrete_actions = self.env.discrete_actions
            self.num_discrete_actions = self.env.num_discrete_actions
            self.discrete_actions_list = self.env.discrete_actions_list
        self.discrete = discrete
        self.metadata = self.env.metadata
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self._max_episode_steps = self.env._max_episode_steps
        self.num_features = self.env.num_features

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        sim_state = copy.deepcopy(self.env.sim.get_state())
        obs['sim_state'] = sim_state
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        sim_state = copy.deepcopy(self.env.sim.get_state())
        obs['sim_state'] = sim_state
        return obs

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def compute_reward(self, *args, **kwargs):
        return self.env.compute_reward(*args, **kwargs)

    def get_sim_state(self):
        return self.env.sim.get_state()

    def make_deterministic(self):
        return self.env.make_deterministic()

    def extract_features(self, obs, g):
        return self.env.extract_features(obs, g)

    def get_obs(self):
        obs = self.env._get_obs()
        sim_state = copy.deepcopy(self.env.sim.get_state())
        obs['sim_state'] = sim_state
        return obs


def get_wrapper_env(env_name, env_id, discrete, reward_type):
    return WrapperEnv(env_name, env_id, discrete, reward_type)
