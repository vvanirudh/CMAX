import sys
import rospy
import numpy as np
import warnings

import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import Grasp
from trajectory_msgs.msg import JointTrajectoryPoint


class pr2_2d_ros_env:
    def __init__(self, args):
        self.args = args
        # Define the grid limits in continuous space
        self.robo_y_at_zero_grid = -0.055
        self.robo_z_at_zero_grid = 0.85
        self.robo_y_at_end_grid = 0.75
        self.robo_z_at_end_grid = 1.3
        # Define number of grid cells
        self.grid_y_lim_cells = args.grid_size
        self.grid_z_lim_cells = args.grid_size
        # Define fixed positions
        # This is the fixed X plane in which the end-effector moves
        # self.fixed_x_plane = 0.5  # HACK: Changed from the value 0.43
        self.fixed_x_plane = 0.43
        # This is the Y where the box will be placed at start and needs to be picked from
        self.pick_y = self.box_y = 0.65
        # This is the Y where the box needs to be placed at the end
        self.place_y = 0.05
        # Grid size
        self.grid_y_size = (self.robo_y_at_end_grid -
                            self.robo_y_at_zero_grid) / self.grid_y_lim_cells
        self.grid_z_size = (self.robo_z_at_end_grid -
                            self.robo_z_at_zero_grid) / self.grid_z_lim_cells
        # Define start and goal grid cells
        self.start_pose = Pose()
        self.start_pose.position.x = self.fixed_x_plane
        self.start_pose.position.y = self.box_y
        self.start_pose.position.z = self.robo_z_at_zero_grid
        self.start_cell = self._continuous_to_grid(self.start_pose)
        self.goal_pose = Pose()
        self.goal_pose.position.x = self.fixed_x_plane
        self.goal_pose.position.y = self.place_y
        self.goal_pose.position.z = self.robo_z_at_zero_grid
        self.goal_cell = self._continuous_to_grid(self.goal_pose)

        # Setup RVIZ
        self._setup_env()

        # Pick the box to start
        self._pick_box()

    def _continuous_to_grid(self, pose):
        rob_y_continuous, rob_z_continuous = pose.position.y, pose.position.z

        # HACK: Clip the continuous values to be within the grid
        rob_y_continuous = min(
            max(rob_y_continuous, self.robo_y_at_zero_grid), self.robo_y_at_end_grid)
        rob_z_continuous = min(
            max(rob_z_continuous, self.robo_z_at_zero_grid), self.robo_z_at_end_grid)

        y_shifted = rob_y_continuous - self.robo_y_at_zero_grid
        z_shifted = rob_z_continuous - self.robo_z_at_zero_grid

        y_grid_cell = y_shifted//self.grid_y_size
        z_grid_cell = z_shifted//self.grid_z_size

        return np.array([y_grid_cell, z_grid_cell], dtype=np.int32)

    def _grid_to_continuous(self, grid_cell):

        y_grid, z_grid = grid_cell

        calc_y_cont = (y_grid)*self.grid_y_size + \
            self.grid_y_size/2.0 + self.robo_y_at_zero_grid

        calc_z_cont = (z_grid)*self.grid_z_size + \
            self.grid_z_size/2.0 + self.robo_z_at_zero_grid

        pose = Pose()
        pose.position.y, pose.position.z = calc_y_cont, calc_z_cont
        pose.position.x = self.fixed_x_plane
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
        # Initialize display trajectory publisher
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory)
        rospy.sleep(1)

        # Remove objects, if already present
        self.scene.remove_world_object("box")
        self.scene.remove_world_object("table")

        # Add objects
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        # Table
        p.pose.position.x = self.fixed_x_plane + 0.15
        p.pose.position.y = 0.52
        p.pose.position.z = 0.38
        p.pose.orientation.w = 1.0
        self.scene.add_box("table", p, (0.264, 2.0, 0.76))
        print('TABLE ADDED')
        rospy.sleep(1)
        # Box
        p.pose.position.x = self.fixed_x_plane + 0.15
        p.pose.position.y = self.box_y
        p.pose.position.z = 0.845
        p.pose.orientation.w = 1.0
        self.scene.add_box("box", p, (0.09, 0.04, 0.17))
        print('BOX ADDED')
        rospy.sleep(1)

        return True

    def _pick_box(self):
        # Construct the grasp
        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = self.fixed_x_plane
        p.pose.position.y = self.box_y
        p.pose.position.z = 0.85
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

    def _place_box(self):
        # Only executed once the robot has reached the goal cell
        # TODO: This one simply opens the gripper and moves away
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
        if not np.any(displacement == 0):
            # None of either y or z is 0 -> incorrect
            warnings.warn(
                'Trying to execute the incorrect displacement', displacement)
            return current_grid_cell
        if not np.any(np.abs(displacement) == 1):
            # None of either y or z is +1/-1 -> incorrect
            warnings.warn(
                'Trying to execute the incorrect displacement', displacement)
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
        # self.group.set_pose_target(pose)
        waypoints = []
        waypoints.append(self.group.get_current_pose().pose)
        waypoints.append(pose)
        # Plan
        # plan = self.group.plan()
        plan, fraction = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0)
        # Visualize, if needed
        if self.args.visualize:
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)
            # Wait, while plan is visualized
            rospy.sleep(1)
        # Move to the pose
        # self.group.go(wait=True)
        self.group.execute(plan)
        # Add a sleep of 1 second
        rospy.sleep(1)
        return True

    def get_current_pose(self):
        return self.group.get_current_pose().pose

    def get_current_grid_cell(self):
        current_pose = self.group.get_current_pose().pose
        current_grid_cell = self._continuous_to_grid(current_pose)

        return current_grid_cell

    def check_goal(self, cell):
        if np.array_equal(cell, self.goal_cell):
            self._place_box()
            return True
        return False
