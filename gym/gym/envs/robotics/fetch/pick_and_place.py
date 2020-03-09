import os
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', discrete=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, discrete=discrete)
        utils.EzPickle.__init__(self)

    def _construct_discrete_actions(self):
        actions = {}
        count = 0
        # Null action
        #actions[(0, 0, 0, 0)] = count
        #count += 1

        for pos_x in range(1, self.n_bins, 1):
            actions[(pos_x, 0, 0, 0)] = count
            count += 1

        for pos_y in range(1, self.n_bins, 1):
            actions[(0, pos_y, 0, 0)] = count
            count += 1

        for pos_z in range(1, self.n_bins, 1):
            actions[(0, 0, pos_z, 0)] = count
            count += 1

        for gripper in [-1, 1]:
            actions[(0, 0, 0, gripper)] = count
            count += 1

        self.num_discrete_actions = count
        return actions
