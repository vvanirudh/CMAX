import os
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_with_obstacle.xml')


class FetchPushWithObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    '''
    Changes made:
    1. New xml file
    2. Increased obj_range and target_range
    3. Increased size of table from default
    '''

    def __init__(self, reward_type='sparse', discrete=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'obstacle0:joint': [1.25, 0.53, 0.5, 1., 0., 0., 0.]
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, has_obstacle=True, discrete=discrete)
        utils.EzPickle.__init__(self)
