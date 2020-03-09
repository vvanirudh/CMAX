import os
import numpy as np
from gym import utils
from odium.envs.fetch import fetch_push_env


def stable_divide(a, b):
    '''
    Ensuring numerical stability
    '''
    if b < 1e-7:
        return a / 1e-7

    return a / b


class FetchPushAmongObstaclesEnv(fetch_push_env.FetchPushEnv, utils.EzPickle):
    def __init__(self, env_id=1, reward_type='sparse', discrete=False):
        # TODO: Need to modify initial values
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        if env_id == 1:
            # Push env with obstacles
            MODEL_XML_PATH = os.path.join(
                'fetch', 'push_among_obstacles_6.xml')
            self.num_features = 15
        elif env_id == 2:
            # Push env but with negative cost for pushing into obstacle
            MODEL_XML_PATH = os.path.join(
                'fetch', 'push_among_obstacles_6.xml')
            self.num_features = 15
        # elif env_id == 3:
        #     MODEL_XML_PATH = os.path.join(
        #         'fetch', 'push_among_obstacles_3.xml')
        elif env_id == 4:
            # Push env with no obstacles
            MODEL_XML_PATH = os.path.join(
                'fetch', 'push_among_obstacles_4.xml')
            self.num_features = 15
        elif env_id == 5:
            # Push env with no obstacles but with a spherical object
            MODEL_XML_PATH = os.path.join(
                'fetch', 'push_among_obstacles_5.xml')
            self.num_features = 9
        else:
            raise NotImplementedError('This env_id is not implemented')

        fetch_push_env.FetchPushEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            env_id=env_id, discrete=discrete, n_bins=3
        )
        utils.EzPickle.__init__(self)

    def _construct_discrete_actions(self):
        actions = {}
        count = 0
        # Null action
        actions[(0, 0, 0, 0)] = count
        count += 1

        for pos_x in range(1, self.n_bins, 1):
            actions[(pos_x, 0, 0, 0)] = count
            count += 1

        for pos_y in range(1, self.n_bins, 1):
            actions[(0, pos_y, 0, 0)] = count
            count += 1

        self.num_discrete_actions = count
        return actions

    def make_deterministic(self):
        obs = self.reset()
        self.fixed_object_pos = obs['observation'][3:6]
        self.fixed_goal_pos = obs['desired_goal']
        self.randomize = False

    def extract_features(self, obs, g):
        '''
        Function to define features from observation
        observation is a dict with 3 entries:
        1. observation : consists of the actual observation
        2. achieved_goal : 3D position of the object
        3. desired_goal : 3D position of the goal

        observation is a 25-D vector consisting of:
        1. 3D Gripper position
        2. 3D object position
        3. 3D object relative position w.r.t. gripper (object_pos - grip_pos)
        4. 2D gripper state
        5. 3D object rotation
        6. 3D object positional velocity
        7. 3D object rotational velocity
        8. 3D Gripper positional velocity
        9. 2D Gripper velocity
        '''
        # Define basis w.r.t center of table
        cot = np.array([1.2, 0.75])
        # Absolute features
        abs_pos_obj_wrt_cot = stable_divide(obs[3:5] - cot,
                                            np.linalg.norm(
            obs[3:5] - cot))
        dist_pos_obj_wrt_cot = np.linalg.norm(obs[3:5] - cot)
        abs_pos_gripper_wrt_cot = stable_divide(obs[:2] - cot,
                                                np.linalg.norm(
                                                    obs[:2] - cot))
        dist_pos_gripper_wrt_cot = np.linalg.norm(obs[:2] - cot)
        # Define relative features
        # Define relative features
        # The 2D relative position of the object w.r.t gripper seems useful
        rel_pos_obj_wrt_gripper = stable_divide(
            obs[6:8], np.linalg.norm(obs[6:8]))
        dist_pos_obj_wrt_gripper = np.linalg.norm(obs[6:8])
        # The 2D relative position of the goal w.r.t object is the next useful thing
        rel_pos_goal_wrt_object = stable_divide(g[:2] -
                                                obs[3:5], np.linalg.norm(
                                                    g[:2] - obs[3:5]))
        dist_pos_goal_wrt_object = np.linalg.norm(g[:2] - obs[3:5])
        # 3D object orientation seems useful
        # TODO: Removing object orientation for now
        object_orientation = obs[11:14]
        # The 2D relative position of the goal w.r.t gripper is redundant but useful
        rel_pos_goal_wrt_gripper = stable_divide(g[:2] -
                                                 obs[:2], np.linalg.norm(g[:2] - obs[:2]))
        dist_pos_goal_wrt_gripper = np.linalg.norm(g[:2] - obs[:2])
        # TODO: Need features to capture the displacement to the nearest obstacle
        # if self.env_id == 1:
        #     obstacle_com = self.sim.data.get_site_xpos('obstacle1')
        #     rel_pos_obj_wrt_obstacle_com = stable_divide(
        #         obs[3:5] - obstacle_com[:2], np.linalg.norm(obs[3:5] - obstacle_com[:2]))
        #     dist_pos_obj_wrt_obstacle_com = np.linalg.norm(
        #         obs[3:5] - obstacle_com[:2])
        #     rel_pos_grip_wrt_obstacle_com = stable_divide(
        #         obs[:2] - obstacle_com[:2], np.linalg.norm(obs[:2] - obstacle_com[:2]))
        #     dist_pos_grip_wrt_obstacle_com = np.linalg.norm(
        #         obs[:2] - obstacle_com[:2])
        #     features = np.concatenate([rel_pos_obj_wrt_gripper,
        #                                [dist_pos_obj_wrt_gripper],
        #                                rel_pos_goal_wrt_object,
        #                                [dist_pos_goal_wrt_object],
        #                                rel_pos_goal_wrt_gripper,
        #                                [dist_pos_goal_wrt_gripper],
        #                                rel_pos_obj_wrt_obstacle_com,
        #                                [dist_pos_obj_wrt_obstacle_com],
        #                                rel_pos_grip_wrt_obstacle_com,
        #                                [dist_pos_grip_wrt_obstacle_com],
        #                                # object_orientation
        #                                ])
        # elif self.env_id == 4 or self.env_id == 5:
        #     # No obstacles
        features = np.concatenate([rel_pos_obj_wrt_gripper,
                                   [dist_pos_obj_wrt_gripper],
                                   rel_pos_goal_wrt_object,
                                   [dist_pos_goal_wrt_object],
                                   rel_pos_goal_wrt_gripper,
                                   [dist_pos_goal_wrt_gripper],
                                   abs_pos_obj_wrt_cot,
                                   [dist_pos_obj_wrt_cot],
                                   abs_pos_gripper_wrt_cot,
                                   [dist_pos_gripper_wrt_cot],
                                   # object_orientation
                                   ])
        # else:
        #     raise NotImplementedError()

        assert features.shape[0] == self.num_features

        return features
