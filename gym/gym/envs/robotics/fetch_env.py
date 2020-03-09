import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
            has_obstacle=False, immovable_obstacles=False, movable_obstacles=False,
            has_static_obstacles=False, num_static_obstacles=2, discrete=False, n_bins=3
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
        self.discrete = discrete
        self.n_bins = n_bins

        if self.discrete:
            self.num_discrete_actions = None
            self.discrete_actions = self._construct_discrete_actions()

        self.has_obstacle = has_obstacle
        self.has_immovable_obstacle = immovable_obstacles
        self.has_movable_obstacle = movable_obstacles

        self.num_static_obstacles = num_static_obstacles

        if self.has_obstacle:
            assert not self.has_immovable_obstacle
            assert not self.has_movable_obstacle

        if self.has_immovable_obstacle:
            assert self.has_movable_obstacle
            assert not self.has_object
            assert not self.has_obstacle

        if self.has_movable_obstacle:
            assert self.has_immovable_obstacle
            assert not self.has_object
            assert not self.has_obstacle

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos, discrete=discrete, n_bins=n_bins)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
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

        # Apply action to simulation.
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
        if self.has_obstacle:
            obstacle_pos = self.sim.data.get_site_xpos('obstacle0')
            # gripper state
            obstacle_grip_rel_pos = obstacle_pos - grip_pos
            # object state
            obstacle_obj_rel_pos = obstacle_pos - object_pos
            obs = np.concatenate([obs, obstacle_pos.ravel(
            ), obstacle_grip_rel_pos.ravel(), obstacle_obj_rel_pos.ravel()])

        # TODO: Need to add identifiers for immovable and movable obstacles
        if self.has_immovable_obstacle:
            immovable_obstacle_pos = self.sim.data.get_site_xpos('immovable0')
            immovable_obstacle_grip_rel_pos = immovable_obstacle_pos - grip_pos
            obs = np.concatenate(
                [obs, immovable_obstacle_pos.ravel(), immovable_obstacle_grip_rel_pos.ravel()])

        if self.has_movable_obstacle:
            movable_obstacle_pos = self.sim.data.get_site_xpos('movable0')
            movable_obstacle_grip_rel_pos = movable_obstacle_pos - grip_pos
            obs = np.concatenate(
                [obs, movable_obstacle_pos.ravel(), movable_obstacle_grip_rel_pos.ravel()])

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

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.obj_range,
                                           self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        # Randomize start position of obstacle.
        if self.has_obstacle:
            obstacle_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(obstacle_xpos - self.initial_gripper_xpos[:2]) < 0.1 or (self.has_object and np.linalg.norm(object_xpos - obstacle_xpos) < 0.1):
                obstacle_xpos = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.obj_range,
                                           self.obj_range, size=2)
            obstacle_qpos = self.sim.data.get_joint_qpos('obstacle0:joint')
            assert obstacle_qpos.shape == (7,)
            obstacle_qpos[:2] = obstacle_xpos
            obstacle_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos('obstacle0:joint', obstacle_qpos)

        # Immovable obstacles
        if self.has_immovable_obstacle:
            immovable_obstacle_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(immovable_obstacle_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                immovable_obstacle_xpos = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.obj_range,
                                           self.obj_range, size=2)
            immovable_obstacle_qpos = self.sim.data.get_joint_qpos(
                'immovable0:joint')
            assert immovable_obstacle_qpos.shape == (7,)
            immovable_obstacle_qpos[:2] = immovable_obstacle_xpos
            immovable_obstacle_qpos[2] = self.immovable_height_offset
            self.sim.data.set_joint_qpos(
                'immovable0:joint', immovable_obstacle_qpos)

        # Movable obstacles
        if self.has_movable_obstacle:
            movable_obstacle_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(movable_obstacle_xpos - self.initial_gripper_xpos[:2]) < 0.1 or (self.has_immovable_obstacle and np.linalg.norm(immovable_obstacle_xpos - movable_obstacle_xpos) < 0.1):
                movable_obstacle_xpos = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.obj_range,
                                           self.obj_range, size=2)
            movable_obstacle_qpos = self.sim.data.get_joint_qpos(
                'movable0:joint')
            assert movable_obstacle_qpos.shape == (7,)
            movable_obstacle_qpos[:2] = movable_obstacle_xpos
            movable_obstacle_qpos[2] = self.movable_height_offset
            self.sim.data.set_joint_qpos(
                'movable0:joint', movable_obstacle_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + \
                self.np_random.uniform(-self.target_range,
                                       self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + \
                self.np_random.uniform(-0.15, 0.15, size=3)
            if self.has_immovable_obstacle:
                goal = self.initial_gripper_xpos[:3] + \
                    self.np_random.uniform(-self.target_range,
                                           self.target_range, size=3)
                goal[2] = self.height_offset

        # Checking if this is an easy problem, if so skip it
        if np.linalg.norm(goal[:2] - self.initial_gripper_xpos[:2]) < 0.1:
            # TODO: Get rid of recursion
            return self._sample_goal()
        if self.has_object:
            object_xpos = self.sim.data.get_joint_qpos('object0:joint')[:2]
            if np.linalg.norm(goal[:2] - object_xpos) < 0.1:
                # TODO: Get rid of recursion
                return self._sample_goal()

        if self.has_obstacle:
            obstacle_qpos = self.sim.data.get_joint_qpos('obstacle0:joint')
            obstacle_xpos = obstacle_qpos[:2]
            distance_to_goal = np.linalg.norm(obstacle_xpos - goal[:2])
            if distance_to_goal < 0.1:
                # Goal is too close to obstacle (or) far from obstacle
                # TODO: Get rid of recursion
                return self._sample_goal()
            else:
                if self.has_object:
                    # Object should be closer to the obstacle than it is from goal
                    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
                    object_xpos = object_qpos[:2]
                    vector_from_goal_to_object = (
                        object_xpos - goal[:2])
                    vector_from_obstacle_to_goal = (goal[:2] - obstacle_xpos)
                    if np.inner(vector_from_goal_to_object, vector_from_obstacle_to_goal) > 0:
                        # Goal and obstacle on either side of object
                        # TODO: Get rid of recursion
                        return self._sample_goal()

        if self.has_immovable_obstacle:
            immovable_obstacle_qpos = self.sim.data.get_joint_qpos(
                'immovable0:joint')
            immovable_obstacle_xpos = immovable_obstacle_qpos[:2]
            distance_to_goal = np.linalg.norm(
                immovable_obstacle_xpos - goal[:2])
            if distance_to_goal < 0.05:
                # Goal is too close to immovable obstacle
                # TODO: Get rid of recursion
                return self._sample_goal()
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
        if self.has_immovable_obstacle:
            gripper_target = np.array(
                [-0.6, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
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
        if self.has_immovable_obstacle:
            self.immovable_height_offset = self.sim.data.get_site_xpos('immovable0')[
                2]
        if self.has_movable_obstacle:
            self.movable_height_offset = self.sim.data.get_site_xpos('movable0')[
                2]
            self.height_offset = self.movable_height_offset
        # Set gripper to the table
        # self.initial_gripper_xpos[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
