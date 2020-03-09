import sys
import math
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Grasp, PlaceLocation, Constraints, OrientationConstraint
from trajectory_msgs.msg import JointTrajectoryPoint


def open_gripper(posture):
    posture.joint_names = []
    posture.joint_names.append("r_gripper_joint")
    posture.joint_names.append("r_gripper_motor_screw_joint")
    posture.joint_names.append("r_gripper_l_finger_joint")
    posture.joint_names.append("r_gripper_r_finger_joint")
    posture.joint_names.append("r_gripper_r_finger_tip_joint")
    posture.joint_names.append("r_gripper_l_finger_tip_joint")

    positions = [1, 1.0, 0.477, 0.477, 0.477, 0.477]
    posture.points = []
    posture.points.append(JointTrajectoryPoint())
    posture.points[0].positions = positions

    return posture


def closed_gripper(posture):
    posture.joint_names = []
    posture.joint_names.append("r_gripper_joint")
    posture.joint_names.append("r_gripper_motor_screw_joint")
    posture.joint_names.append("r_gripper_l_finger_joint")
    posture.joint_names.append("r_gripper_r_finger_joint")
    posture.joint_names.append("r_gripper_r_finger_tip_joint")
    posture.joint_names.append("r_gripper_l_finger_tip_joint")

    positions = [0, 0.0, 0.002, 0.002, 0.002, 0.002]
    posture.points = []
    posture.points.append(JointTrajectoryPoint())
    posture.points[0].positions = positions

    return posture


def pick(robot):
    # Construct the grasp
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()
    p.pose.position.x = 0.34
    p.pose.position.y = -0.7
    p.pose.position.z = 0.5
    p.pose.orientation.w = 1.0
    g = Grasp()
    g.grasp_pose = p

    # Construct pre-grasp
    g.pre_grasp_approach.direction.vector.x = 1.0
    g.pre_grasp_approach.direction.header.frame_id = "r_wrist_roll_link"
    g.pre_grasp_approach.min_distance = 0.2
    g.pre_grasp_approach.desired_distance = 0.4

    # Construct post-grasp
    g.post_grasp_retreat.direction.header.frame_id = robot.get_planning_frame()
    g.post_grasp_retreat.direction.vector.z = 1.0
    g.post_grasp_retreat.min_distance = 0.1
    g.post_grasp_retreat.desired_distance = 0.25

    g.pre_grasp_posture = open_gripper(g.pre_grasp_posture)
    g.grasp_posture = closed_gripper(g.grasp_posture)

    grasps = []
    grasps.append(g)

    group = robot.get_group("right_arm")
    group.set_support_surface_name("table")

    group.pick("part", grasps)


def place(robot):
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()
    p.pose.position.x = 0.7
    p.pose.position.y = 0.0
    p.pose.position.z = 0.5
    p.pose.orientation.x = 0
    p.pose.orientation.y = 0
    p.pose.orientation.z = 0
    p.pose.orientation.w = 1.0

    g = PlaceLocation()
    g.place_pose = p

    g.pre_place_approach.direction.vector.z = -1.0
    g.post_place_retreat.direction.vector.x = -1.0
    g.post_place_retreat.direction.header.frame_id = robot.get_planning_frame()
    g.pre_place_approach.direction.header.frame_id = "r_wrist_roll_link"
    g.pre_place_approach.min_distance = 0.1
    g.pre_place_approach.desired_distance = 0.2
    g.post_place_retreat.min_distance = 0.1
    g.post_place_retreat.desired_distance = 0.25

    g.post_place_posture = open_gripper(g.post_place_posture)

    group = robot.get_group("right_arm")
    group.set_support_surface_name("table")

    # Add path constraints
    constr = Constraints()
    constr.orientation_constraints = []
    ocm = OrientationConstraint()
    ocm.link_name = "r_wrist_roll_link"
    ocm.header.frame_id = p.header.frame_id
    ocm.orientation.x = 0.0
    ocm.orientation.y = 0.0
    ocm.orientation.z = 0.0
    ocm.orientation.w = 1.0
    ocm.absolute_x_axis_tolerance = 0.2
    ocm.absolute_y_axis_tolerance = 0.2
    ocm.absolute_z_axis_tolerance = math.pi
    ocm.weight = 1.0
    constr.orientation_constraints.append(ocm)

    # group.set_path_constraints(constr)
    group.set_planner_id("RRTConnectkConfigDefault")

    group.place("part", g)


def sample_pick():
    # Initialize moveit commander
    moveit_commander.roscpp_initialize(sys.argv)
    # Initialize ros node
    rospy.init_node("moveit_py_demo", anonymous=True)
    # Initialize scene
    scene = moveit_commander.PlanningSceneInterface()
    # Initialize robot
    robot = moveit_commander.RobotCommander()
    # Get group
    group = robot.get_group("right_arm")
    group.set_planning_time(45.0)
    # Wait for rviz to initialize
    print('Waiting for RVIZ to initialize')
    rospy.sleep(10)

    # Remove objects if present
    scene.remove_world_object("part")
    scene.remove_world_object("pole")
    scene.remove_world_object("table")
    # Add objects to the scene
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()
    # Pole
    p.pose.position.x = 0.7
    p.pose.position.y = -0.4
    p.pose.position.z = 0.85
    p.pose.orientation.w = 1.0
    scene.add_box("pole", p, (0.3, 0.1, 1.0))

    # Table
    p.pose.position.y = -0.2
    p.pose.position.z = 0.175
    scene.add_box("table", p, (0.5, 1.5, 0.35))

    # Part
    p.pose.position.x = 0.6
    p.pose.position.y = -0.7
    p.pose.position.z = 0.5
    scene.add_box("part", p, (0.15, 0.1, 0.3))

    rospy.sleep(1)

    # Pick
    pick(robot)

    rospy.sleep(1)

    # Place
    place(robot)

    rospy.spin()
    moveit_commander.roscpp_shutdown()
