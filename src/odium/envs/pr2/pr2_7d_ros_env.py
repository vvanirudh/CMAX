'''
This file contains the ros interface for the PR2 7D env
'''
import os
import os.path as osp
import sys
import copy
import rospy
import numpy as np
from urdfpy import URDF

import moveit_commander
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from tf import transformations
from geometry_msgs.msg import PoseStamped, Pose

from pr2_controllers_msgs.msg import *
import actionlib


class pr2_7d_ros_env:
    def __init__(self, args, table=True):
        self.args = args
        # Setup essentials
        # Init
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moveit_node', anonymous=True)
        # Setup scene
        self.scene = moveit_commander.PlanningSceneInterface()
        # Initialize robot
        self.robot = moveit_commander.RobotCommander()
        # Get group
        self.group = self.robot.get_group('left_arm')
        self.base_group = self.robot.get_group('base')
        self.right_group = self.robot.get_group('right_arm')
        base_link_pose = self.base_group.get_current_pose()
        self.base_tf = self._construct_matrix_from_pose(
            base_link_pose.pose)
        # Set planning time
        self.group.set_planning_time(10.0)
        # Set goal tolerance
        self.group.set_goal_tolerance(0.005)
        # Setup joint limits
        self._setup_joint_limits()

        # import ipdb
        # ipdb.set_trace()

        # Setup RVIZ
        self._setup_env()

        # Setup URDF
        self._setup_urdf()
        # Read args
        self.goal_tolerance = args.goal_tolerance
        self.grid_size = args.grid_size

        self.grid_limits_arr = self.true_joint_limits[:, 1] - \
            self.true_joint_limits[:, 0]

        # Grid size
        self.grid_size_arr = self.grid_limits_arr / self.grid_size
        # Start cell and goal pose
        self.start_cell = self._continuous_to_grid(
            self.start_joint_config)

        self.table_dimensions = [0.264, 2.0, 0.76]

        self.end_effector_roll_link_shift = 0.15

        self.table_x = 0.53 + self.end_effector_roll_link_shift
        self.table_y = 0.52
        self.table_z = self.table_dimensions[2]/2

        if table:
            self._setup_table()

        # Define a goal pose that we are trying to reach
        # np.array([0.51, 0.71, 1.07])
        # np.array([0.409, 0.719, 0.813])
        # np.array([0.603, 0.4, 0.927])
        # self.goal_pose = np.array([0.41, 0.72, 1.09])
        # self.goal_pose = np.array([0.41, 0.72, 1.05]) # broken joint 1 values
        # broken joint 3 values

        if not args.goal_6d:
            # self.goal_pose = np.array(
            #     [0.625019786729, 0.470264266838, 0.970789956906])
            self.goal_pose = np.array(
                [0.625019786729, 0.470264266838, 0.850789956906])
        else:
            goal_pose = Pose()
            # pose before moving the robot 0.285323119211
            goal_pose.position.x = 0.451613551774
            goal_pose.position.y = 0.522707661949
            goal_pose.position.z = 0.989203613931 - 0.05
            goal_pose.orientation.x = -0.0586202806291
            goal_pose.orientation.y = 0.0220664705909
            goal_pose.orientation.z = 0.029858076886
            goal_pose.orientation.w = 0.99758970966
            self.goal_pose = self._construct_xyzrpy_from_pose(goal_pose)

        self.broken_joint = args.broken_joint
        self.broken_joint_index = args.broken_joint_index

        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path_search',
            DisplayTrajectory, queue_size=1)

        self.discrepancy_publisher = rospy.Publisher(
            'discrepancy_marker', Marker, queue_size=1)

        self.goal_publisher = rospy.Publisher(
            'goal_marker', Marker, queue_size=1)
        rospy.sleep(1)
        self._publish_goal_marker()

    def _setup_table(self):

        # Remove objects, if already present
        self.scene.remove_world_object("table")

        # Add objects
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        # Table
        # TODO: Clean this
        p.pose.position.x = self.table_x
        p.pose.position.y = self.table_y
        p.pose.position.z = self.table_z

        self.scene.add_box("table", p, tuple(self.table_dimensions))
        print('TABLE ADDED')
        rospy.sleep(1)

    def _construct_matrix_from_pose(self, pose):
        translation = np.array(
            [pose.position.x, pose.position.y, pose.position.z])
        translation_matrix = transformations.translation_matrix(translation)

        quaternion = np.array(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        quaternion_matrix = transformations.quaternion_matrix(quaternion)

        pose_matrix = np.dot(translation_matrix, quaternion_matrix)

        return pose_matrix

    def _construct_pose_from_matrix(self, matrix):
        translation = transformations.translation_from_matrix(matrix)
        quaternion = transformations.quaternion_from_matrix(matrix)

        pose = Pose()
        pose.position.x = translation[0]
        pose.position.y = translation[1]
        pose.position.z = translation[2]
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose

    def _construct_xyzrpy_from_matrix(self, matrix):
        translation = transformations.translation_from_matrix(matrix)
        euler = transformations.euler_from_matrix(matrix)

        pose = np.concatenate([translation, euler])

        return pose

    def _construct_xyzrpy_from_pose(self, pose):
        translation = [pose.position.x, pose.position.y, pose.position.z]
        euler = transformations.euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                                       pose.orientation.w])

        pose = np.concatenate([translation, euler])

        return pose

    def _setup_env(self):
        # Move to a fixed start joint configuration

        print("setting up the env")

        # joint_values_dict = {}
        # # dictionary with integer keys
        # # joint_values_dict = {1: 'apple', 2: 'ball'}
        # joint_values_dict = {'l_shoulder_pan_joint': 0.0, 'l_shoulder_lift_joint': 0.0, 'l_upper_arm_roll_joint': 0.0,
        #                      'l_elbow_flex_joint': -1.13565, 'l_forearm_roll_joint': 0.0, 'l_wrist_flex_joint': -1.05, 'l_wrist_roll_joint': 0.0}

        # self.group.set_joint_value_target(joint_values_dict)

        # self.start_joint_config = np.array(
        #     [0.0, 0.0, 0.0, -1.13565, 0.0, -1.05, 0.0])  # broken joint 1 values
        # self.start_joint_config = np.array([1.6282268354481173, -0.09153170679901994, 1.4139693066002512, -0.14604685462423772, -142.62529016056308, -0.41611121567400045, -31.745293364141144]) # broken joint 3 values

        # this is for the grasp broken joint experiment - breaking joint 1
        self.start_joint_config = [2.1350349873513714, -0.10845069141991091 - 0.1, 0.8469546405563337, -
                                   0.985861751360082, -9.070481080934837, -0.09734813282433619, -0.17842386576278058]
        # self.start_joint_config = [1.5212772990576058, 0.08950142864451305, 1.3907178567766179, -0.9294011739205063, 0.2658652116247761, -1.0513247149280436, 0.011623097232247526]

        # self.start_joint_config = self.group.get_random_joint_values()

        self._move_to_joint_config(self.start_joint_config)

        # Move right arm away
        self.right_arm_config = [-1.8872953364408611, 0.36233243738497545, 1.767231654375791e-05, -
                                 1.683313474303413, 0.643046973963422, -1.2695123792044125, 1.024420504378357]

        self.right_group.set_joint_value_target(self.right_arm_config)
        plan = self.right_group.plan()
        self.right_group.execute(plan)

        return True

    def set_start_and_goal(self, start_config, goal_config):
        self.start_joint_config = start_config.copy()
        self.start_cell = self._continuous_to_grid(self.start_joint_config)

        if not self._move_to_joint_config(self.start_joint_config):
            raise Exception('Could not reset to start joint config')

        self.goal_cell = self._continuous_to_grid(goal_config)
        self.goal_pose = self.get_pose(self.goal_cell)

        self._publish_goal_marker()

        return True

    def sample_valid_start_and_goal(self, joint_index=1):
        # Sample start
        start_config = self.group.get_random_joint_values()
        start_cell = self._continuous_to_grid(start_config)

        while (not self._move_to_joint_config(start_config, execute=False)):
            start_config = self.group.get_random_joint_values()
            start_cell = self._continuous_to_grid(start_config)

        while True:
            # Sample goal
            goal_config = self.group.get_random_joint_values()
            goal_cell = self._continuous_to_grid(goal_config)
            goal_pose = self.get_pose(goal_cell)
            # Check joint index
            if goal_cell[joint_index] != start_cell[joint_index]:
                continue

            # Check if its already at goal
            start_pose = self.get_pose(start_cell)
            if np.linalg.norm(start_pose - goal_pose) < self.goal_tolerance:
                continue

            # Finally, check if start config is reachable from goal config
            if not self._move_to_joint_config(goal_config, execute=True):
                continue
            if not self._move_to_joint_config(start_config, execute=False):
                continue

            break

        return np.array(start_config), start_cell, np.array(goal_config), goal_cell, goal_pose

    def _publish_goal_marker(self):

        goal_marker = Marker()
        goal_marker.header.frame_id = "/odom_combined"
        goal_marker.type = goal_marker.SPHERE
        goal_marker.scale.x = self.goal_tolerance
        goal_marker.scale.y = self.goal_tolerance
        goal_marker.scale.z = self.goal_tolerance
        goal_marker.color.r, goal_marker.color.g, goal_marker.color.b, goal_marker.color.a = 0, 1, 0, 0.5
        goal_marker.pose.orientation.w = 1
        goal_marker.pose.position.x = self.goal_pose[0]
        goal_marker.pose.position.y = self.goal_pose[1]
        goal_marker.pose.position.z = self.goal_pose[2]

        self.goal_publisher.publish(goal_marker)

    def _publish_discrepancy_marker(self, text):

        marker = Marker()
        marker.header.frame_id = "/odom_combined"
        marker.type = marker.TEXT_VIEW_FACING
        marker.text = text
        marker.scale.z = 0.1
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 2.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1, 0, 0, 1
        marker.lifetime = rospy.Duration.from_sec(0.5)

        self.discrepancy_publisher.publish(marker)

    def _setup_urdf(self):
        # Parse the URDF
        # TODO: Need to fix this absolute path
        self.urdf = URDF.load(osp.join(os.environ['HOME'],
                                       # 'ws_moveit/src/odium/urdf/pr2.urdf'))
                                       'workspaces/odium_ws/src/odium/urdf/pr2.urdf'))
        self.link_names = [link.name for link in self.urdf.links]
        self.jmap = {}
        for j in self.robot.get_joint_names():
            if j != 'world_joint':
                self.jmap[j] = self.robot.get_joint(j).value()

        # self.group.get_end_effector_link()
        self.end_link = 'l_gripper_motor_slider_link'
        self.cache_fk = {}
        return True

    def _get_joint_names(self):
        return ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint',
                'l_elbow_flex_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint']

    def _setup_joint_limits(self):

        # TRUE JOINT LIMITS
        self.true_joint_limits = np.zeros((7, 2))
        self.true_joint_limits[0, :] = [-0.57,  2.13]
        self.true_joint_limits[1, :] = [-0.36,  1.29]
        self.true_joint_limits[2, :] = [-0.65,  2.2]
        self.true_joint_limits[3, :] = [-2.13, -0.15]
        self.true_joint_limits[4, :] = [-np.pi + 0.001, np.pi]
        self.true_joint_limits[5, :] = [-2.02, -0.1]
        self.true_joint_limits[6, :] = [-np.pi + 0.001, np.pi]

        # FAKE JOINT LIMITS
        self.joint_limits = self.true_joint_limits.copy()

    def _move_to_joint_config(self, joint_config, execute=True):
        self.group.set_joint_value_target(joint_config)
        plan = self.group.plan()
        if execute:
            success_execute = self.group.execute(plan)
            self.group.clear_pose_targets()
            return success_execute
        return (len(plan.joint_trajectory.points) != 0)

    def _continuous_to_grid(self, pose):
        wrapped_pose = copy.deepcopy(pose)
        alpha_4 = np.round(wrapped_pose[4]/(2*np.pi))
        alpha_6 = np.round(wrapped_pose[6]/(2*np.pi))
        wrapped_pose[4] = wrapped_pose[4] - alpha_4 * 2 * np.pi
        wrapped_pose[6] = wrapped_pose[6] - alpha_6 * 2 * np.pi
        zero_adjusted_grid = np.array(
            wrapped_pose) - self.true_joint_limits[:, 0]
        grid_cell = np.array(zero_adjusted_grid //
                             np.array(self.grid_size_arr), dtype=np.int32)
        grid_cell = np.maximum(0, np.minimum(grid_cell, self.grid_size-1))
        return grid_cell

    def _grid_to_continuous(self, grid_cell):
        joint_config = grid_cell*self.grid_size_arr + \
            self.grid_size_arr/2.0 + self.true_joint_limits[:, 0]
        return joint_config

    def _get_end_effector_link_pose(self, joint_config):
        if tuple(joint_config) in self.cache_fk:
            return self.cache_fk[tuple(joint_config)]
        else:
            self.cache_fk[tuple(joint_config)] = self._compute_fk(self._get_joint_names(),
                                                                  joint_config,
                                                                  self.end_link)
            return self.cache_fk[tuple(joint_config)]

    def _compute_fk(self, joint_names, joint_config, end_link):
        for i in range(len(joint_names)):
            self.jmap[joint_names[i]] = joint_config[i]

        fk = self.urdf.link_fk(cfg=self.jmap)
        link_index = self.link_names.index(end_link)

        transform = fk[self.urdf.links[link_index]]

        end_link_tf = transform.dot(self.base_tf)

        if not self.args.goal_6d:
            end_link_pose = self._construct_pose_from_matrix(end_link_tf)
            end_link_pose_vector = np.array(
                [end_link_pose.position.x, end_link_pose.position.y, end_link_pose.position.z])
        else:
            end_link_pose_vector = self._construct_xyzrpy_from_matrix(
                end_link_tf)
        return end_link_pose_vector

    def get_pose(self, grid_cell):
        # Get joint config
        joint_config = self._grid_to_continuous(grid_cell)
        pose = self._get_end_effector_link_pose(joint_config)
        return pose

    def move_to_cell(self, commanded_grid_cell):
        if self.broken_joint:
            current_grid_cell = self.get_current_grid_cell()
            commanded_grid_cell[self.broken_joint_index] = current_grid_cell[self.broken_joint_index]
        joint_config = self._grid_to_continuous(commanded_grid_cell)
        self._move_to_joint_config(joint_config)
        self.group.clear_pose_targets()
        current_joint_config = self.group.get_current_joint_values()
        current_grid_cell = self._continuous_to_grid(current_joint_config)
        self._publish_goal_marker()
        return current_grid_cell

    def check_goal(self, grid_cell):
        pose = self.get_pose(grid_cell)
        if np.linalg.norm(pose - self.goal_pose) < self.goal_tolerance:
            return True
        return False

    def get_current_grid_cell(self):
        joint_config = self.group.get_current_joint_values()
        return self._continuous_to_grid(joint_config)

    def get_current_joint_config(self):
        joint_config = self.group.get_current_joint_values()
        return joint_config

    def get_joint_trajectory_msg(self, path_config):
        trajectory = JointTrajectory()
        trajectory.header = self.group.get_current_pose().header
        trajectory.joint_names = self._get_joint_names()
        time_from_start = 0
        for config in path_config:
            trajectory.points.append(
                self.get_joint_trajectory_point_msg(config, time_from_start))
            time_from_start += 0.5

        return trajectory

    def get_joint_trajectory_point_msg(self, config, time_from_start):
        point = JointTrajectoryPoint()
        point.positions = config
        point.time_from_start = rospy.Duration.from_sec(time_from_start)
        return point

    def get_robot_trajectory_msg(self, path_config):
        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory = self.get_joint_trajectory_msg(
            path_config)
        return robot_trajectory

    def visualize_path(self, path):
        '''
        path will be a list of grid cells
        '''
        current_state = self.robot.get_current_state()
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = current_state
        path_config = [self._grid_to_continuous(cell) for cell in path]
        display_trajectory.trajectory.append(self.get_robot_trajectory_msg(
            path_config))

        self.display_trajectory_publisher.publish(display_trajectory)

    def open_right_gripper(self):
        right_gripper_client = actionlib.SimpleActionClient(
            "l_gripper_controller/gripper_action", Pr2GripperCommandAction)
        print("waiting for server")
        right_gripper_client.wait_for_server()
        print("server active")
        goal = Pr2GripperCommandGoal()

        # goal.command.position = 0.0
        # goal.command.max_effort = 50.0

        goal.command.position = 0.08
        goal.command.max_effort = -1.0
        return right_gripper_client.send_goal_and_wait(goal, rospy.Duration(30.0), rospy.Duration(5.0))

    def rotate_gripper(self):
        pose_target = Pose()
        pose_target.orientation.w = 1.0
        pose_target.position.x = self.group.get_current_pose().pose.position.x
        pose_target.position.y = self.group.get_current_pose().pose.position.y
        pose_target.position.z = self.group.get_current_pose().pose.position.z
        self.group.set_pose_target(pose_target)

        # Now, we call the planner to compute the plan
        # and visualize it if successful
        # Note that we are just planning, not asking move_group
        # to actually move the robot
        plan1 = self.group.plan()
        self.group.execute(plan1)

    def close_right_gripper(self):
        right_gripper_client = actionlib.SimpleActionClient(
            "l_gripper_controller/gripper_action", Pr2GripperCommandAction)
        print("waiting for server")
        right_gripper_client.wait_for_server()
        print("server active")
        goal = Pr2GripperCommandGoal()

        goal.command.position = 0.0
        goal.command.max_effort = 50.0

        # goal.command.position = 0.08
        # goal.command.max_effort = -1.0
        return right_gripper_client.send_goal_and_wait(goal, rospy.Duration(30.0), rospy.Duration(5.0))

    def goto_postmanip_pose(self):
        waypoints = []
        # start with the current pose
        waypoints.append(self.group.get_current_pose().pose)
        # first orient gripper and move forward (+x)
        wpose = Pose()  # geometry_msgs.msg.Pose()
        wpose.orientation.w = 1.0
        wpose.position.x = waypoints[0].position.x - 0.1
        wpose.position.y = waypoints[0].position.y
        wpose.position.z = waypoints[0].position.z
        waypoints.append(copy.deepcopy(wpose))
        # # second move down
        # wpose.position.z -= 0.10
        # waypoints.append(copy.deepcopy(wpose))
        # # third move to the side
        # wpose.position.y += 0.05
        # waypoints.append(copy.deepcopy(wpose))
        # We want the cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in cartesian
        # translation.  We will specify the jump threshold as 0.0, effectively
        # disabling it.
        (plan3, fraction) = self.group.compute_cartesian_path(
            waypoints,   # waypoints to follow
            0.01,        # eef_step
            0.0)         # jump_threshold
        self.group.execute(plan3)
