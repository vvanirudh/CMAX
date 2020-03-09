import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, discrete, n_bins=3):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(
                __file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.discrete = discrete
        self.n_bins = n_bins

        self.goal = self._sample_goal()
        obs = self._get_obs()
        if not discrete:
            # Continuous
            self.action_space = spaces.Box(-1., 1.,
                                           shape=(n_actions,), dtype='float32')
        else:
            # Discrete
            self.action_space = spaces.MultiDiscrete(
                [self.n_bins for _ in range(n_actions-1)]+[2])
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf,
                                    shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf,
                                     shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf,
                                   shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        prev_obs = self._get_obs()
        prev_achieved_goal = prev_obs['achieved_goal'].copy()
        prev_gripper_pos = prev_obs['observation'][0:3].copy()
        if not self.discrete:
            # continuous
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)

        initial_qpos = self.sim.get_state().qpos.copy()
        initial_qvel = self.sim.get_state().qvel.copy()
        # if self.env_id == 1 and np.any(self.sim.data.sensordata != 0.0):
        #     # print('Object already touching the obstacle')
        self._set_action(action)
        penetration = False
        for _ in range(2):
            self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        if self.env_id == 1 or self.env_id == 2:
            # Check if object is penetrating the obstacle
            # obj_pos = self.sim.data.get_site_xpos('object0')
            # obstacle_size_x = 0.025 + 0.02
            # obstacle_size_y = 0.15 + 0.02
            # obstacle_boundaries_x = np.array([1.25 - obstacle_size_x, 1.25 +
            #                                   obstacle_size_x])
            # obstacle_boundaries_y = np.array(
            #     [0.75 - obstacle_size_y, 0.75 + obstacle_size_y])
            # # Get all 4 corners of the object
            # obj_size = 0.06
            # obj_corners = np.array([[obj_pos[0] + obj_size/2, obj_pos[1] + obj_size/2],
            #                         [obj_pos[0] + obj_size/2,
            #                          obj_pos[1] - obj_size/2],
            #                         [obj_pos[0] - obj_size/2,
            #                          obj_pos[1] - obj_size/2],
            #                         [obj_pos[0] + obj_size/2, obj_pos[1] + obj_size/2]])
            # # Check if any of the corner is within the obstacle boundaries
            # for corner in obj_corners:
            #     if corner[0] >= obstacle_boundaries_x[0] and corner[0] <= obstacle_boundaries_x[1]:
            #         if corner[1] >= obstacle_boundaries_y[0] and corner[1] <= obstacle_boundaries_y[1]:
            #             penetration = True
            #             # print('DETECTED PENETRATION')
            #             break

            touch_sensor_data = self.sim.data.sensordata
            if np.any(touch_sensor_data != 0.0):
                # print('Touch sensor activated')
                penetration = True

            if penetration:
                # Set initial state
                # print('resetting state')
                current_state = self.sim.get_state()
                current_state.qpos[:] = initial_qpos.copy()
                current_state.qvel[:] = initial_qvel.copy()
                self.sim.set_state(current_state)
                self.sim.forward()
                obs = self._get_obs()
        curr_gripper_pos = obs['observation'][0:3].copy()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'prev_achieved_goal': prev_achieved_goal,
            'prev_gripper_pos': prev_gripper_pos,
            'curr_gripper_pos': curr_gripper_pos,
            'penetration': penetration,
        }
        reward = self.compute_reward(
            obs['achieved_goal'], self.goal, info, penetration)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

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

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def extract_features(self, obs, g):
        raise NotImplementedError

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
