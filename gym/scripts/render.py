import gym
import numpy as np

env = gym.make('FetchPushWithObstacleEnv-v1')
env.seed(1)
env.reset()
i = 0
while True:
    if i % 80 == 0:
        action = np.array([0.5, 0, 0, 0])
    elif i % 40 == 0:
        action = np.array([-0.5, 0, 0, 0])
    elif i % 20 == 0:
        action = np.array([0, 0.1, 0, 0])
    elif i % 10 == 0:
        action = np.array([0, -0.1, 0, 0])
    obs, _, _, _ = env.step(action)
    print(obs['observation'][-9:])
    env.render()
    i += 1
