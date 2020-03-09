import sys
import rospy
import numpy as np
import warnings
import copy
import math

import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import Grasp, PlaceLocation
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectoryPoint
from nav_msgs.msg import Path

from pr2_controllers_msgs.msg import Pr2GripperCommandGoal, Pr2GripperCommandAction
import actionlib


class pr2_3d_ros_env:
    def __init__(self, args):
        self.args = args

        # Dimensions of the obstacle
        # These are the dimensions of the old obstacle : [0.17, 0.07, 0.24]
        self.obstacle_dimensions = [0.15, 0.12, 0.375]

        # Dimensions of the table
        self.table_dimensions = [0.264, 2.0, 0.76]

        # Dimensions of pick-place object
        self.box_dimensions = [0.12, 0.04, 0.20]

        # Account for shift between end_effector and roll_link
        self.end_effector_roll_link_shift = 0.15
        self.end_effector_length_y = 0.13

        # Account for placing objects and obstacles further away on the table
        self.transform_table_object = 0.13/2

        # Obstacle values
        self.dist_obstacle_box = 0.23

        # Positions of different objects in the scene
        # If any position is not mentioned here, it means
        # that it is defined relative to 
        # Position of box 
        self.box_x = 0.53 #(23) 0.43 (22) + 0.1 
        # that it is defined relative to
        # Position of box
        self.box_y = 0.65
        self.box_z = 0.845
        # Define fixed positions
        # This is the fixed X plane in which the end-effector moves
        # self.fixed_x_plane = 0.43
        # This is the Y where the box will be placed at start and needs to be picked from
        self.pick_x = self.box_x
        self.pick_y = self.box_y
        # This is the Y where the box needs to be placed at the end
        self.place_x = self.box_x + 0.1
        self.place_y = 0.05

        # Define the grid limits in continuous space
        self.robo_x_at_zero_grid = 0.30  # old value = 0.20
        self.robo_y_at_zero_grid = -0.055
        self.robo_z_at_zero_grid = self.box_z + 0.01  # old value = 0.76
        self.robo_x_at_end_grid = self.box_x + 0.01
        self.robo_y_at_end_grid = self.box_y + 0.01  # old value = 0.75
        self.robo_z_at_end_grid = 1.3

        # Position of the table
        self.table_x = self.box_x + self.end_effector_roll_link_shift
        self.table_y = 0.52
        self.table_z = self.table_dimensions[2]/2

        # Position of the obstacle
        self.obstacle_box_centre_offset = 0.015
        self.obstacle_x = self.box_x - self.obstacle_box_centre_offset
        self.obstacle_y = self.box_y - self.dist_obstacle_box
        self.obstacle_z = self.table_dimensions[2] + \
            self.obstacle_dimensions[2]/2

        # Define number of grid cells
        self.grid_y_lim_cells = args.grid_size
        self.grid_z_lim_cells = args.grid_size
        self.grid_x_lim_cells = args.grid_size

        # Grid size
        self.grid_x_size = (self.robo_x_at_end_grid -
                            self.robo_x_at_zero_grid) / self.grid_x_lim_cells
        self.grid_y_size = (self.robo_y_at_end_grid -
                            self.robo_y_at_zero_grid) / self.grid_y_lim_cells
        self.grid_z_size = (self.robo_z_at_end_grid -
                            self.robo_z_at_zero_grid) / self.grid_z_lim_cells

        # Define start and goal grid cells
        self.start_pose = Pose()
        self.start_pose.position.x = self.box_x
        self.start_pose.position.y = self.box_y
        self.start_pose.position.z = self.robo_z_at_zero_grid
        self.start_cell = self._continuous_to_grid(self.start_pose)
        self.goal_pose = Pose()
        self.goal_pose.position.x = self.place_x
        self.goal_pose.position.y = self.place_y
        self.goal_pose.position.z = 0.85
        self.goal_cell = self._continuous_to_grid(self.goal_pose)

        # Setup RVIZ
        self._setup_env()

        # Go to pre-grasp pose
        self.goto_pregrasp_pose()

        # Pick the box to start
        self._pick_box()

        self.display_trajectory_publisher = rospy.Publisher(
            '/path_3d',
            Path, queue_size=1)

        self.discrepancy_publisher = rospy.Publisher(
            'discrepancy_marker', Marker, queue_size=1)
        self.discrepancy_id = 0

        self.goal_publisher = rospy.Publisher(
            'goal_marker', Marker, queue_size=1)
        rospy.sleep(1)
        self._publish_goal_marker()

    def _continuous_to_grid(self, pose):
        rob_x_continuous, rob_y_continuous, rob_z_continuous = pose.position.x, pose.position.y, pose.position.z

        # HACK: Clip the continuous values to be within the grid
        rob_x_continuous = min(
            max(rob_x_continuous, self.robo_x_at_zero_grid), self.robo_x_at_end_grid)
        rob_y_continuous = min(
            max(rob_y_continuous, self.robo_y_at_zero_grid), self.robo_y_at_end_grid)
        rob_z_continuous = min(
            max(rob_z_continuous, self.robo_z_at_zero_grid), self.robo_z_at_end_grid)

        x_shifted = rob_x_continuous - self.robo_x_at_zero_grid
        y_shifted = rob_y_continuous - self.robo_y_at_zero_grid
        z_shifted = rob_z_continuous - self.robo_z_at_zero_grid

        x_grid_cell = x_shifted//self.grid_x_size
        y_grid_cell = y_shifted//self.grid_y_size
        z_grid_cell = z_shifted//self.grid_z_size

        x_grid_cell = max(0, min(x_grid_cell, self.args.grid_size-1))
        y_grid_cell = max(0, min(y_grid_cell, self.args.grid_size-1))
        z_grid_cell = max(0, min(z_grid_cell, self.args.grid_size-1))

        return np.array([x_grid_cell, y_grid_cell, z_grid_cell], dtype=np.int32)

    def _grid_to_continuous(self, grid_cell):

        x_grid, y_grid, z_grid = grid_cell

        calc_x_cont = (x_grid)*self.grid_x_size + \
            self.grid_x_size/2.0 + self.robo_x_at_zero_grid

        calc_y_cont = (y_grid)*self.grid_y_size + \
            self.grid_y_size/2.0 + self.robo_y_at_zero_grid

        calc_z_cont = (z_grid)*self.grid_z_size + \
            self.grid_z_size/2.0 + self.robo_z_at_zero_grid

        pose = Pose()
        pose.position.y, pose.position.z = calc_y_cont, calc_z_cont
        pose.position.x = calc_x_cont
        pose.orientation.w = 1.0
        return pose

    def _setup_env(self):
        # Init
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moveit_node', anonymous=True)
        # Setup scene
        self.scene = moveit_commander.PlanningSceneInterface()
        # Initialize robot
        self.robot = moveit_commander.RobotCommander()
        # Get group
        self.group = self.robot.get_group('left_arm')
        self.gripper_group = self.robot.get_group('left_gripper')
        self.group.set_planning_time(30.0)
        # Initialize display trajectory publisher
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory)
        rospy.sleep(1)

        # Remove objects, if already present
        self.scene.remove_world_object("box")
        self.scene.remove_world_object("table")
        self.scene.remove_world_object("obstacle")

        # debugging
        # self.open_right_gripper()
        # import ipdb
        # ipdb.set_trace()
        # self._place_box()

        # Add objects
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        # Table
        # TODO: Clean this
        p.pose.position.x = self.table_x
        p.pose.position.y = self.table_y
        p.pose.position.z = self.table_z
        p.pose.orientation.w = 1.0
        # self.table_dimensions = [0.264, 2.0, 0.76]
        self.scene.add_box("table", p, tuple(self.table_dimensions))
        print('TABLE ADDED')
        rospy.sleep(1)
        # Box
        # TODO: Clean this
        p.pose.position.x = self.box_x + \
            self.end_effector_roll_link_shift + self.transform_table_object
        p.pose.position.y = self.box_y
        p.pose.position.z = self.box_z
        p.pose.orientation.w = 1.0
        # This was the dimensions of the light box
        # self.scene.add_box("box", p, (0.09, 0.04, 0.17))
        self.scene.add_box("box", p, tuple(self.box_dimensions))
        print('BOX ADDED')
        rospy.sleep(1)
        # Obstacle
        # TODO: Clean this
        p.pose.position.x = self.obstacle_x + \
            self.end_effector_roll_link_shift + self.transform_table_object
        p.pose.position.y = self.obstacle_y
        p.pose.position.z = self.obstacle_z
        p.pose.orientation.w = 1.0
        self.scene.add_box("obstacle", p, tuple(self.obstacle_dimensions))
        print('OBSTACLE ADDED')
        rospy.sleep(1)

        return True

    def _pick_box(self):
        # Construct the grasp
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        # TODO: Clean this
        p.pose.position.x = self.box_x
        p.pose.position.y = self.box_y
        p.pose.position.z = self.box_z  # old value = 0.85
        p.pose.orientation.w = 1.0
        g = Grasp()
        g.grasp_pose = p

        # Construct pre-grasp
        g.pre_grasp_approach.direction.vector.x = 1.0
        g.pre_grasp_approach.direction.header.frame_id = "l_wrist_roll_link"
        g.pre_grasp_approach.min_distance = 0.2
        g.pre_grasp_approach.desired_distance = 0.4

        g.pre_grasp_posture = self._open_gripper(g.pre_grasp_posture)
        g.grasp_posture = self._closed_gripper(g.grasp_posture)

        grasps = []
        grasps.append(g)

        self.group.set_support_surface_name("table")
        self.group.set_planner_id("RRTkConfigDefault")

        print("EXECUTING PICK")
        self.group.pick("box", grasps)
        rospy.sleep(3)
        return True

    def place(self):
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = self.place_x
        p.pose.position.y = self.place_y
        p.pose.position.z = self.box_z
        # p.pose.orientation.x = 0
        # p.pose.orientation.y = 0
        # p.pose.orientation.z = 0
        # p.pose.orientation.w = 1.0

        g = PlaceLocation()
        g.place_pose = p

        g.pre_place_approach.direction.vector.z = -1.0
        g.post_place_retreat.direction.vector.x = -1.0
        g.post_place_retreat.direction.header.frame_id = self.robot.get_planning_frame()
        g.pre_place_approach.direction.header.frame_id = "l_wrist_roll_link"
        g.pre_place_approach.min_distance = 0.0
        g.pre_place_approach.desired_distance = 0.0
        g.post_place_retreat.min_distance = 0.0
        g.post_place_retreat.desired_distance = 0.05

        g.post_place_posture = self._open_gripper(g.post_place_posture)

        self.group = self.robot.get_group("left_arm")
        self.group.set_support_surface_name("table")

        # group.set_path_constraints(constr)
        self.group.set_planner_id("RRTConnectkConfigDefault")

        self.group.place("box", g)

    def _place_box(self):
        print("PLACING BOX")
        # Only executed once the robot has reached the goal cell
        # TODO: This one simply opens the gripper and moves away
        # open_joint_values = [0.132, 0.132, 0.0, 0.0, 0.132, 0.132, 0.0]
        open_joint_values = [0.44611993634859004, 0.44611993634859004, 0.0,
                             0.0, 0.44611993634859004, 0.44611993634859004, 0.07729435420649784]
        close_joint_values = [0.011423711526516417, 0.011423711526516417, 0.0,
                              0.0, 0.011423711526516417, 0.011423711526516417, 0.001702811613909296]
        self.gripper_group.set_joint_value_target(open_joint_values)

        # joint_values_dict = {}
        # dictionary with integer keys
        # joint_values_dict = {1: 'apple', 2: 'ball'}

        # joint_values_dict = {"l_gripper_joint": 1.0, "l_gripper_motor_screw_joint": 1.0,
        #     "l_gripper_r_finger_joint": 0.477, "l_gripper_l_finger_joint": 0.477,
        #     "l_gripper_l_finger_tip_joint": 0.477, "l_gripper_r_finger_tip_joint":0.477}

        # joint_values_dict = {"l_gripper_joint": 1.0}

        # my_dict = dict({1:'apple', 2:'ball'})
        # self.gripper_group.set_joint_value_target(joint_values_dict)

        plan2 = self.gripper_group.plan()
        # self.gripper_group.execute(plan2)
        self.gripper_group.go(wait=True)

        self.gripper_group.clear_pose_targets()

        return True

    def _open_gripper(self, posture):
        posture.joint_names = []
        posture.joint_names.append("l_gripper_joint")
        posture.joint_names.append("l_gripper_motor_screw_joint")
        posture.joint_names.append("l_gripper_r_finger_joint")
        posture.joint_names.append("l_gripper_l_finger_joint")
        posture.joint_names.append("l_gripper_l_finger_tip_joint")
        posture.joint_names.append("l_gripper_r_finger_tip_joint")

        positions = [1, 1.0, 0.477, 0.477, 0.477, 0.477]
        posture.points = []
        posture.points.append(JointTrajectoryPoint())
        posture.points[0].positions = positions

        return posture

    def _closed_gripper(self, posture):
        posture.joint_names = []
        posture.joint_names.append("l_gripper_joint")
        posture.joint_names.append("l_gripper_motor_screw_joint")
        posture.joint_names.append("l_gripper_r_finger_joint")
        posture.joint_names.append("l_gripper_l_finger_joint")
        posture.joint_names.append("l_gripper_l_finger_tip_joint")
        posture.joint_names.append("l_gripper_r_finger_tip_joint")

        positions = [0, 0.0, 0.132, 0.132, 0.132, 0.132]
        posture.points = []
        posture.points.append(JointTrajectoryPoint())
        posture.points[0].positions = positions

        return posture

    def move_to_cell(self, commanded_grid_cell):
        # Get current grid cell
        current_pose = self.group.get_current_pose().pose
        # Get the corresponding grid cell
        current_grid_cell = self._continuous_to_grid(current_pose)
        # Check if commanded grid cell is one of the 4 adjacent neighbors of current grid cell
        displacement = current_grid_cell - commanded_grid_cell
        if np.count_nonzero(displacement == 0) < 2:
            # None of either y or z is 0 -> incorrect
            warnings.warn(
                'Trying to execute the incorrect displacement'+np.array2string(displacement))
            return current_grid_cell
        if np.abs(displacement[displacement != 0]) != 1:
            warnings.warn(
                'Trying to execute the incorrect displacement'+np.array2string(displacement))
            return current_grid_cell

        # If the commanded grid cell is one of the neighbors, move to it
        commanded_pose = self._grid_to_continuous(
            commanded_grid_cell)
        self.move_to_pose(commanded_pose)
        # Read the current pose and get its grid cell
        current_pose = self.group.get_current_pose().pose
        current_grid_cell = self._continuous_to_grid(current_pose)

        return current_grid_cell

    def move_to_pose(self, pose):
        # Set the pose as the target
        self.group.set_pose_target(pose)
        # waypoints = []
        # waypoints.append(self.group.get_current_pose().pose)
        # waypoints.append(pose)
        # Plan
        plan = self.group.plan()
        # plan, fraction = self.group.compute_cartesian_path(
        #     waypoints, 0.01, 0.0)
        # Visualize, if needed
        if self.args.visualize:
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectiry.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)
            # Wait, while plan is visualized
            rospy.sleep(1)
        # Move to the pose
        # self.group.go(wait=True)
        self.group.execute(plan)
        # Add a sleep of 1 second
        # rospy.sleep(1)
        return True

    def get_current_pose(self):
        return self.group.get_current_pose().pose

    def get_current_grid_cell(self):
        current_pose = self.group.get_current_pose().pose
        current_grid_cell = self._continuous_to_grid(current_pose)

        return current_grid_cell

    def check_goal(self, cell):
        if np.array_equal(cell, self.goal_cell):
            # self._place_box()
            # self.place()
            self.open_right_gripper()
            self.goto_postmanip_pose()
            return True
        return False

    def get_obstacle_cells(self):
        # TODO: This is incorrect. Need to get the obstacle pose in terms of the gripper roller link pose
        # Given obstacle_x, obstacle_y, obstacle_z and its dimensions
        # Figure out which grid cells it occupies
        lower_corner = Pose()
        lower_corner.position.x = self.obstacle_x - \
            self.obstacle_dimensions[0]/2.0 - self.box_dimensions[0]/2
        lower_corner.position.y = self.obstacle_y - \
            self.obstacle_dimensions[1]/2.0 - self.end_effector_length_y/2
        lower_corner.position.z = self.obstacle_z - \
            self.obstacle_dimensions[2]/2.0

        # Get its grid cell
        lower_corner_cell = self._continuous_to_grid(lower_corner)

        upper_corner = Pose()
        upper_corner.position.x = self.obstacle_x + \
            self.obstacle_dimensions[0]/2.0 + self.box_dimensions[0]/2
        upper_corner.position.y = self.obstacle_y + \
            self.obstacle_dimensions[1]/2.0 + self.end_effector_length_y/2
        upper_corner.position.z = self.obstacle_z + \
            self.obstacle_dimensions[2]/2.0 + self.box_dimensions[2]/2

        # Get its grid cell
        upper_corner_cell = self._continuous_to_grid(upper_corner)

        # Find the range of cells to be marked
        list_of_cells = set()
        for x in range(lower_corner_cell[0], upper_corner_cell[0]+1):
            for y in range(lower_corner_cell[1], upper_corner_cell[1]+1):
                for z in range(lower_corner_cell[2], upper_corner_cell[2]+1):
                    list_of_cells.add((x, y, z))

        return list_of_cells

    def _move_to_joint_config(self, joint_config):
        self.group.set_joint_value_target(joint_config)
        plan2 = self.group.plan()
        # self.group.execute(plan2)
        self.group.go(wait=True)
        self.group.clear_pose_targets()
        return True

    def goto_pregrasp_pose(self):
        print("EXECUTING PRE-GRASP POSE")
        start_pose = np.array([2.135449520438156, 0.5712695157243824, 1.6208270326174046, -
                               1.6959620906962858, -24.666634193675453, -0.5400246201176523, -23.598161057651897])
        # self.start_joint_config = np.array(
        #     [0.0, 0.0, 0.0, -1.13565, 0.0, -1.05, 0.0])
        self.start_joint_config = start_pose
        self._move_to_joint_config(self.start_joint_config)

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

    def _publish_discrepancy_marker(self, discrepancy_cell):

        discrepancy_location = self._grid_to_continuous(discrepancy_cell)

        discrepancy_marker = Marker()
        discrepancy_marker.header.frame_id = "/odom_combined"
        discrepancy_marker.type = discrepancy_marker.SPHERE
        discrepancy_marker.scale.x = self.grid_z_size
        discrepancy_marker.scale.y = self.grid_z_size
        discrepancy_marker.scale.z = self.grid_z_size
        discrepancy_marker.color.r, discrepancy_marker.color.g, discrepancy_marker.color.b, discrepancy_marker.color.a = 1, 0, 0, 0.5
        discrepancy_marker.pose.orientation.w = 1
        discrepancy_marker.pose.position.x = discrepancy_location.position.x + self.end_effector_roll_link_shift
        discrepancy_marker.pose.position.y = discrepancy_location.position.y
        discrepancy_marker.pose.position.z = discrepancy_location.position.z
        discrepancy_marker.id = self.discrepancy_id
        self.discrepancy_id += 1

        self.discrepancy_publisher.publish(discrepancy_marker)

    def _publish_goal_marker(self):

        goal_marker = Marker()
        goal_marker.header.frame_id = "/odom_combined"
        goal_marker.type = goal_marker.SPHERE
        goal_marker.scale.x = 0.05
        goal_marker.scale.y = 0.05
        goal_marker.scale.z = 0.05
        goal_marker.color.r, goal_marker.color.g, goal_marker.color.b, goal_marker.color.a = 0, 0, 1, 0.8
        goal_marker.pose.orientation.w = 1
        goal_marker.pose.position.x = self.goal_pose.position.x + self.end_effector_roll_link_shift
        goal_marker.pose.position.y = self.goal_pose.position.y
        goal_marker.pose.position.z = self.goal_pose.position.z
        goal_marker.id = 2

        self.goal_publisher.publish(goal_marker)

    def visualize_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = "/odom_combined"
        poses = [self._grid_to_continuous(cell) for cell in path]
        for pose in poses:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "/odom_combined"
            pose_stamped.pose.position.x = pose.position.x + self.end_effector_roll_link_shift
            pose_stamped.pose.position.y = pose.position.y
            pose_stamped.pose.position.z = pose.position.z
            path_msg.poses.append(pose_stamped)

        self.display_trajectory_publisher.publish(path_msg)
