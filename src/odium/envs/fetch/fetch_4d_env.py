import copy
import os
import os.path as osp
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.robotics import utils

import mujoco_py

DEFAULT_SIZE = 500


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class fetch_4d_env(gym.GoalEnv):
    def __init__(self, env_id, seed):

        # Parameters
        self.num_discrete_actions = 4
        self.cell_size = 0.02
        self.continuous_limits_high = np.array([1.6, 1.0])  # x, y
        self.continuous_limits_low = np.array([1.0, 0.4])  # x, y
        self.grid_size = 30
        self.env_id = env_id

        # Flags
        self.obj_x = 0

        # Load model
        if self.env_id == 0:
            model_path = osp.join(
                os.environ['HOME'],
                'workspaces/odium_ws/src/odium/src/odium/envs/assets/fetch/push_among_obstacles_4.xml'
            )
        elif self.env_id == 1:
            model_path = osp.join(
                os.environ['HOME'],
                'workspaces/odium_ws/src/odium/src/odium/envs/assets/fetch/push_among_obstacles_1.xml'
            )
        else:
            raise Exception('Invalid env_id')
        model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps=20)
        self.seed(seed)

        self.viewer = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        # Sample goal
        self.goal, self.goal_cell = self._sample_goal()

        # Construct spaces
        space_4D = spaces.Tuple((spaces.Discrete(self.grid_size),
                                 spaces.Discrete(self.grid_size),
                                 spaces.Discrete(self.grid_size),
                                 spaces.Discrete(self.grid_size)))
        space_2D = spaces.Tuple((spaces.Discrete(self.grid_size),
                                 spaces.Discrete(self.grid_size)))
        self.observation_space = spaces.Dict(dict(
            desired_goal=space_2D,
            achieved_goal=space_2D,
            observation=space_4D))
        self.action_space = spaces.Discrete(4)

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, ac):
        self._set_action(ac)
        for _ in range(10):
            self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = {'is_success': self._is_success(obs['achieved_goal'],
                                               self.goal_cell)}
        cost = self.compute_cost(obs['achieved_goal'], self.goal_cell)
        return obs, cost, done, info

    def reset(self):
        super(fetch_4d_env, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal, self.goal_cell = self._sample_goal()
        obs = self._get_obs()
        return obs

    # RENDERING STUFF
    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(
                    self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _reset_sim(self):
        '''
        This is executed before _sample_goal in the reset function
        '''
        self.sim.set_state(self.initial_state)
        obj_range_x, obj_range_y = self._get_obj_range()
        object_xpos = np.zeros(2)
        object_xpos[0] = self.np_random.uniform(obj_range_x[0], obj_range_x[1])
        object_xpos[1] = self.np_random.uniform(obj_range_y[0], obj_range_y[1])
        # Set the object pos in simulator
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()

        return True

    def _get_obs(self):
        # positions
        gripper_pos = self.sim.data.get_site_xpos('robot0:grip')[:2]
        object_pos = self.sim.data.get_site_xpos('object0')[:2]
        continuous_pos = np.concatenate([gripper_pos, object_pos])
        # Get grid cell
        grid_cell = self._continuous_to_grid(gripper_pos, object_pos)
        observation = {'observation': grid_cell.copy(),
                       'continuous_observation': continuous_pos.copy(),
                       'achieved_goal': grid_cell[2:].copy(),
                       'desired_goal': self.goal_cell.copy(),
                       'sim_state': copy.deepcopy(self.sim.get_state())}
        return observation

    def _set_action(self, ac):
        current_grid_cell = self._get_obs()['observation']
        pos_ctrl, gripper_ctrl = np.zeros(3), np.zeros(1)
        if ac == 0:
            if current_grid_cell[0] >= 1:
                pos_ctrl[0] = -self.cell_size
        elif ac == 1:
            if current_grid_cell[0] <= self.grid_size - 1:
                pos_ctrl[0] = self.cell_size
        elif ac == 2:
            if current_grid_cell[1] >= 1:
                pos_ctrl[1] = -self.cell_size
        elif ac == 3:
            if current_grid_cell[1] <= self.grid_size - 1:
                pos_ctrl[1] = self.cell_size
        else:
            raise Exception('Invalid action')

        rot_ctrl = [1.0, 0.0, 1.0, 0.0]
        # Force the gripper to be at table height
        gripper_pos = self.sim.data.get_site_xpos('robot0:grip')
        pos_ctrl[2] = self.height_offset-gripper_pos[2]
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        # Apply action in the simulator using IK
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _is_success(self, achieved_goal, desired_goal):
        return np.array_equal(achieved_goal, desired_goal)

    def _sample_goal(self):
        goal_range_x, goal_range_y = self._get_goal_range()
        goal = self.initial_gripper_xpos.copy()
        achieved_goal = self.sim.data.get_site_xpos('object0')[:2]
        goal[0] = self.np_random.uniform(goal_range_x[0], goal_range_y[1])
        goal[1] = self.np_random.uniform(goal_range_x[1], goal_range_y[1])
        goal_cell = self._continuous_to_2d_grid(goal[:2])
        while self._is_success(achieved_goal, goal_cell):
            goal[0] = self.np_random.uniform(goal_range_x[0], goal_range_y[1])
            goal[1] = self.np_random.uniform(goal_range_x[1], goal_range_y[1])
            goal_cell = self._continuous_to_2d_grid(goal)
        return goal, goal_cell

    def extract_features(self):
        pass

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        self.initial_gripper_xpos = self.sim.data.get_site_xpos(
            'robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

        return True

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos -
                        self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _step_callback(self):
        # Block the gripper, since we do not need it
        self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
        self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
        self.sim.forward()

    def compute_cost(self, achieved_goal, goal):
        if np.array_equal(achieved_goal, goal):
            return 0
        return 1

    def _continuous_to_grid(self, gripper_pos, object_pos):
        zero_adjusted_gripper_pos = gripper_pos - self.continuous_limits_low
        gripper_grid_cell = zero_adjusted_gripper_pos // self.cell_size
        gripper_grid_cell = np.maximum(
            0, np.minimum(gripper_grid_cell, self.grid_size-1))
        zero_adjusted_object_pos = object_pos - self.continuous_limits_low
        object_grid_cell = zero_adjusted_object_pos // self.cell_size
        object_grid_cell = np.maximum(
            0, np.minimum(object_grid_cell, self.grid_size-1))

        return np.concatenate([gripper_grid_cell, object_grid_cell])

    def _continuous_to_2d_grid(self, pos):
        zero_adjusted_pos = pos - self.continuous_limits_low
        grid_cell = zero_adjusted_pos // self.cell_size
        grid_cell = np.maximum(0, np.minimum(grid_cell, self.grid_size-1))

        return grid_cell

    def _grid_to_continuous(self, grid_cell):
        gripper_grid_cell = grid_cell[:2]
        object_grid_cell = grid_cell[2:]

        gripper_pos = gripper_grid_cell * self.cell_size + \
            self.continuous_limits_low + self.cell_size / 2.0
        object_pos = object_grid_cell * self.cell_size + \
            self.continuous_limits_low + self.cell_size / 2.0

        return gripper_pos, object_pos

    def _get_obj_range(self):
        if self.env_id == 0:
            # No obstacle
            obj_range_y = [0.45, 0.975]
            obj_range_x = [1.05, 1.5]
        elif self.env_id == 1:
            # One obsacle
            obj_range_y = [0.675, 0.825]
            toss = self.np_random.rand()
            if toss < 0.5:
                self.obj_x = 1
                obj_range_x = [1.05, 1.2]
            else:
                self.obj_x = 0
                obj_range_x = [1.3, 1.5]
        else:
            raise Exception('Invalid env_id')

        return obj_range_x, obj_range_y

    def _get_goal_range(self):
        if self.env_id == 0:
            # No obstacle
            target_range_x = [0.975, 1.425]
            target_range_y = [0.525, 0.975]
        elif self.env_id == 1:
            # One obstacle
            target_range_y = [0.675, 0.825]
            if self.obj_x == 1:
                target_range_x = [1.3, 1.425]
            else:
                target_range_x = [0.975, 1.2]
        else:
            raise Exception('Invalid env_id')

        return target_range_x, target_range_y
