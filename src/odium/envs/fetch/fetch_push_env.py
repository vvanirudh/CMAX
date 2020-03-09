import numpy as np

from gym.envs.robotics import rotations, utils
from odium.envs import robot_env


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchPushEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
            env_id, discrete, n_bins=3
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.env_id = env_id
        self.discrete = discrete
        self.n_bins = n_bins

        if self.discrete:
            self.num_discrete_actions = None
            self.discrete_actions = self._construct_discrete_actions()
            self.discrete_actions_list = list(self.discrete_actions.keys())

        self.randomize = True
        self.fixed_object_pos = None
        self.fixed_goal_pos = None

        # Dummy
        self.obj_x = 0  # 0 means below, 1 means above
        self.obj_y = 0  # 0 means left, 1 means middle and 2 means right
        self.obj_inside = False  # True means inside the bug trap, else outside

        super(FetchPushEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos, discrete=discrete, n_bins=n_bins)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info, penetration=None):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            reward = -(d > self.distance_threshold).astype(np.float32)
            # if self.env_id == 2:
            #     if isinstance(reward, np.ndarray):
            #         np.putmask(reward, penetration, -100)
            #     else:
            #         reward = reward if not penetration else -100
            return reward
        else:
            # Dense reward
            # block_position = achieved_goal.copy()
            # block_width = 0.1
            # block_to_goal_angle = np.arctan2(
            #     goal[0] - block_position[0], goal[1] - block_position[1])
            # target_gripper_position = block_position.copy()
            # target_gripper_position[0] += -1. * \
            #     np.sin(block_to_goal_angle) * block_width / 2.0
            # target_gripper_position[1] += -1. * \
            #     np.cos(block_to_goal_angle) * block_width / 2.0
            # target_gripper_position[2] += 0.005

            # g_d = goal_distance(info['curr_gripper_pos'],
            #                     target_gripper_position)
            # return -(d + g_d)
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        if not self.discrete:
            # Continuous
            assert action.shape == (4,)
            action = action.copy()  # ensure that we don't change the action outside of this scope
            pos_ctrl, gripper_ctrl = action[:3], action[3]

            pos_ctrl *= 0.05  # limit maximum change in position
            # fixed rotation of the end effector, expressed as a quaternion
            rot_ctrl = [1., 0., 1., 0.]
            gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
            assert gripper_ctrl.shape == (2,)
            if self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        else:
            # Discrete
            assert action.shape == (4,)
            action = action.copy()
            pos_discrete_ctrl, gripper_discrete_ctrl = action[:3], action[3]
            assert np.all(pos_discrete_ctrl >= 0)
            scale = 0.04
            # Convert discrete to continuous control
            # 0 - nothing
            # 1 - (n_bins - 1) / 2 - +scale
            # (n_bins+1) / 2 - (n_bins - 1) - -scale
            pos_ctrl = np.zeros_like(pos_discrete_ctrl, dtype=np.float32)
            for ac in range(3):
                if pos_discrete_ctrl[ac] == 0:
                    pos_ctrl[ac] = 0.
                elif pos_discrete_ctrl[ac] > 0 and pos_discrete_ctrl[ac] <= (self.n_bins - 1) / 2:
                    pos_ctrl[ac] = scale * pos_discrete_ctrl[ac]
                else:
                    pos_ctrl[ac] = -scale * \
                        (pos_discrete_ctrl[ac] - (self.n_bins - 1)/2)

            # Fixed rotation of the end effector, expressed as a quaternion
            rot_ctrl = [1., 0., 1., 0.]
            # -1 means close the gripper, 0 means do nothing, 1 means open the gripper
            gripper_ctrl = np.array(
                [gripper_discrete_ctrl, gripper_discrete_ctrl])
            if self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Force gripper to be at table height
        obs = self._get_obs()
        gripper_pos = obs['observation'][0:3]
        pos_ctrl[2] = self.height_offset - gripper_pos[2]
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        # Apply action to simulation
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def get_action_index(self, action):
        # Defines a fixed ordering over discrete actions
        if self.discrete:
            return self.discrete_actions[action]
        else:
            raise Exception('Seeking action index for a continuous action')

    def _construct_discrete_actions(self):
        raise NotImplementedError(
            'Construct discrete actions function not defined')

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(
                self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(
                0)
        gripper_state = robot_qpos[-2:]
        # change to a scalar if the gripper is made symmetric
        gripper_vel = robot_qvel[-2:] * dt

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(
            ), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

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

    def _get_obj_range(self):
        if self.env_id == 4 or self.env_id == 5:
            # No obstacles
            # obj_range_y = [0.525, 0.975]
            obj_range_y = [0.55, 0.95]
            # obj_range_x = [0.975, 1.425]
            obj_range_x = [1.0, 1.4]
        elif self.env_id == 1:
            # One obstacle
            # obj_range_y = [0.525, 0.975]
            # obj_range_y = [0.55, 0.95]
            obj_range_y = [0.675, 0.825]
            toss = self.np_random.rand()
            if toss < 0.5:
                self.obj_x = 1
                # obj_range_x = [1.1, 1.2]  # Changed from 0.975 to 1.0
                obj_range_x = [1.05, 1.1]
            else:
                self.obj_x = 0
                # obj_range_x = [1.3, 1.4]
                obj_range_x = [1.35, 1.4]
        elif self.env_id == 2:
            # Two obstacles
            obj_range_x = [0.975, 1.425]
            toss = self.np_random.rand()
            if toss < 0.33:
                self.obj_y = 0
                obj_range_y = [0.525, 0.65]
            elif toss < 0.66:
                self.obj_y = 1
                obj_range_y = [0.65, 0.85]
            else:
                self.obj_y = 2
                obj_range_y = [0.85, 0.975]
        elif self.env_id == 3:
            # Three obstacles as a bug trap
            toss = self.np_random.rand()
            if toss < 0.5:
                self.obj_inside = True
                obj_range_x = [1.1, 1.25]
                obj_range_y = [0.65, 0.85]
            else:
                self.obj_inside = False
                obj_range_x = [0.975, 1.425]
                obj_range_y = [0.525, 0.975]
        else:
            raise NotImplementedError('Other env_id not implemented')

        return obj_range_x, obj_range_y

    def _reset_sim(self):
        '''
        This is executed before _sample_goal in the reset function
        '''
        self.sim.set_state(self.initial_state)

        obj_range_x, obj_range_y = self._get_obj_range()

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            #     # object_xpos = self.initial_gripper_xpos[:2] + \
            #     #     self.np_random.uniform(-self.obj_range,
            #     #                            self.obj_range, size=2)
            if self.randomize:
                object_xpos[0] = self.np_random.uniform(
                    obj_range_x[0], obj_range_x[1])
                object_xpos[1] = self.np_random.uniform(
                    obj_range_y[0], obj_range_y[1])
            else:
                object_xpos[0] = self.fixed_object_pos[0]
                object_xpos[1] = self.fixed_object_pos[1]
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _get_target_range(self):
        if self.env_id == 4 or self.env_id == 5:
            # target_range_x = [0.975, 1.425]
            target_range_x = [1.0, 1.4]
            # target_range_y = [0.525, 0.975]
            target_range_y = [0.55, 0.95]
        elif self.env_id == 1:
            # target_range_y = [0.525, 0.975]
            # target_range_y = [0.55, 0.95]
            target_range_y = [0.675, 0.825]
            if self.obj_x == 1:
                # target_range_x = [1.3, 1.425]
                target_range_x = [1.35, 1.4]
            else:
                target_range_x = [1.05, 1.1]  # Changed from 0.975 to 1.0
        elif self.env_id == 2:
            target_range_x = [0.975, 1.425]
            if self.obj_y == 0:
                target_range_y = [0.65, 0.975]
            elif self.obj_y == 2:
                target_range_y = [0.525, 0.85]
            else:
                target_range_y = [0.525, 0.975]
        elif self.env_id == 3:
            if self.obj_inside:
                target_range_x = [0.975, 1.425]
                target_range_y = [0.525, 0.975]
            else:
                # Target should be inside
                target_range_x = [1.1, 1.25]
                target_range_y = [0.65, 0.85]
        else:
            raise NotImplementedError('Other env_id not implemented')

        return target_range_x, target_range_y

    def _sample_goal(self):
        target_range_x, target_range_y = self._get_target_range()

        if self.has_object:
            # goal = self.initial_gripper_xpos[:3] + \
            #     self.np_random.uniform(-self.target_range,
            #                            self.target_range, size=3)
            # goal += self.target_offset
            goal = self.initial_gripper_xpos[:3]
            if self.randomize:
                goal[0] = self.np_random.uniform(
                    target_range_x[0], target_range_x[1])
                goal[1] = self.np_random.uniform(
                    target_range_y[0], target_range_y[1])
                goal[2] = self.height_offset
            else:
                goal = self.fixed_goal_pos
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)

            object_xpos = self.sim.data.get_site_xpos('object0')
            if self._is_success(object_xpos, goal):
                # Initial state success
                # Resample
                return self._sample_goal()
        else:
            goal = self.initial_gripper_xpos[:3] + \
                self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        # gripper_target = np.array(
        #     [-0.45, 0.25, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos(
            'robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchPushEnv, self).render(mode, width, height)

    def make_deterministic(self):
        raise NotImplementedError()

    def extract_features(self, obs, g):
        raise NotImplementedError()

    def get_obs(self):
        return self._get_obs()

    def set_goal(self, goal):
        self.goal = goal.copy()
        return True
