#!/usr/bin/env python
import rospy
import rospkg
import roslaunch
import numpy as np
import sys

import moveit_commander
from moveit_msgs.msg import PlanningScene
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, JointState
import sensor_msgs.point_cloud2 as pc2
from ctypes import cast,POINTER,pointer,c_float,c_uint32
from gazebo_msgs.srv import GetWorldProperties, GetModelState
from gazebo_msgs.msg import ContactsState

from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher


class UR5LabEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all UR5LabEnv environments.
    """
    

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new UR5LabEnv environment.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Make sure simulation is unpaused: without that, the stream of data doesnt flow.
        2) If the simulation was already running for some reason, reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        Actuators Topic List:
        * /gripper_controller/command
        
        Additional Data:
        * moveit_commander
        * gripper contact sensor
        """
        rospy.loginfo("Start UR5LabEnv INIT...")
        
        # We launch the ROSlaunch that spawns the robot into the world
        self.launch = ROSLauncher(rospackage_name="ur_gazebo",
                    launch_file_name="ur5_lab.launch",
                    ros_ws_abspath=ros_ws_abspath)
        
        # Internal Vars
        self.controllers_list = []
        
        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(UR5LabEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")

        rospy.logdebug("UR5LabEnv unpausing")
        self.gazebo.unpauseSim()
                
        self._check_all_systems_ready()
        
        rospy.Subscriber("/joint_states", JointState, self._joints_state_callback)
        self.last_contact_r = 0
        self.last_contact_l = 0
        
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        ros_pack = rospkg.RosPack()
        ros_pack = ros_pack.get_path('ur5_moveit_config')
        ros_pack += '/launch/move_group.launch'
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [ros_pack])
        self.launch.start()
        rospy.loginfo("move_group started")
        
        self._move = MoveIt()
        
        rospy.logdebug("UR5LabEnv pausing")
        rospy.logdebug("Finished UR5LabEnv INIT...")
        
    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def close(self):
        self.launch.launch.shutdown()
        super().close()
        
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("UR5LabEnv check_all_systems_ready...")
        self._check_all_publishers_ready()
        self._check_all_sensors_ready()
        rospy.logdebug("END UR5LabEnv _check_all_systems_ready...")
        return True

    def _check_all_publishers_ready(self):
        rospy.logdebug("START ALL PUBLISHER")
        rospy.logdebug("All Publishers READY")
    
    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_joint_states_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _JointState_numpy(self, data):
        #print(data)
        data.name = list(data.name)
        data.position = list(data.position)
        finger_index = data.name.index('finger_joint')
        manipulator_index = [data.name.index('shoulder_pan_joint'), data.name.index('shoulder_lift_joint'),
                                data.name.index('elbow_joint'), data.name.index('wrist_1_joint'),
                                data.name.index('wrist_2_joint'), data.name.index('wrist_3_joint')]
                                
        manipulator_joints_position = [data.position[i] for i in manipulator_index]
                
        gripper_joints_position = data.position[finger_index]
        return manipulator_joints_position, gripper_joints_position
        
    def _check_joint_states_ready(self): #check published joint names and rearrange them for the move joint_goal
        self.joint_states = None
        self.gripper_states = None
        rospy.logdebug("Waiting for /joint_states to be READY...")
        while self.joint_states is None and not rospy.is_shutdown():
            try:
                self.joint_states, self.gripper_states = self._JointState_numpy(rospy.wait_for_message("/joint_states", JointState, timeout=2.0))
                rospy.logdebug("Current /joint_states READY=>")

            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joint_states
        
    def _joints_state_callback(self, data):
        self.joint_states, self.gripper_states = self._JointState_numpy(data)
        return 
        
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time 
        we reset at the start of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """
        Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """
        Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """
        Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def get_joint_states(self): 
        self.gazebo.unpauseSim()
        #joint = self._move.robot_joints()
        joint = self.joint_states
        return joint 
    
    def get_pc(self):
        return np.asarray(self.pc.points)
    
    def get_ee_pose(self): 
        """
        Returns the endeffector pose
        We make sure the simulation is unpaused because this calss is a service call.
        """
        rospy.logdebug("START getting endeffector pose")
        self.gazebo.unpauseSim()
        gripper_pose = self._move.ee_pose()
        rospy.logdebug("END getting endeffector pose")
        return gripper_pose   
        
    def get_ee_rpy(self): 
        rospy.logdebug("START getting endeffector rpy")
        self.gazebo.unpauseSim()
        gripper_rpy = self._move.ee_rpy()
        rospy.logdebug("END getting endeffector rpy")
        return gripper_rpy
        
    def move_ee(self, desired_pose): 
        """
        Moves the endeffector to the pose given,
        relative pose to world frame
        Currently it is only accepting position
        """
        pose_goal = Pose()
        pose_goal.position.x = desired_pose[0]
        pose_goal.position.y = desired_pose[1]
        pose_goal.position.z = desired_pose[2]
        pose_goal.orientation.x = desired_pose[3]
        pose_goal.orientation.y = desired_pose[4]
        pose_goal.orientation.z = desired_pose[5]
        pose_goal.orientation.w = desired_pose[6]
        
        
        result = self._move.ee_traj(pose_goal)
        return result
           
    def move_joints(self, joint_goal, wait=True): 
        """
        Moves the joints of the robot directly
        """
        result = self._move.joint_traj(joint_goal, wait)
        return result
    
    def check_pose(self, pose):
        pos = Pose()
        pos.position.x = pose[0]
        pos.position.y = pose[1]
        pos.position.z = pose[2]
        pos.orientation.x = pose[3]
        pos.orientation.y = pose[4]
        pos.orientation.z = pose[5]
        pos.orientation.w = pose[6]
        
        return self._move.check_pos(pos)[0]
    
    def get_observation(self):
        obs = self._get_obs()
        return obs

class MoveIt(object):
    def __init__(self):
        rospy.logdebug("===== MoveIt")
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")
        self.move_group.set_num_planning_attempts(1)
        self.move_group.allow_replanning(False)
        self.eef_link = self.move_group.get_end_effector_link()
        self.joint_index()
        
        rospy.logdebug("============")

    def ee_traj(self, goal):
        self.move_group.clear_pose_targets()
        self.move_group.set_pose_target(goal)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return plan

    def joint_traj(self, positions_array, wait=True):
        self.move_group.clear_pose_targets()   
        positions_array = [positions_array[i] for i in self.joint_name_index]
        result = self.move_group.go(positions_array, wait=wait)
        self.move_group.stop()
        return result
        
    def robot_joints(self):
        return self.move_group.get_current_joint_values()
        
    def check_pos(self, pose):
        return self.move_group.plan(pose)
    
    def joint_index(self):
        active_joints = self.move_group.get_active_joints()
        self.joint_name_index = [active_joints.index('shoulder_pan_joint'), active_joints.index('shoulder_lift_joint'),
                                active_joints.index('elbow_joint'), active_joints.index('wrist_1_joint'),
                                active_joints.index('wrist_2_joint'), active_joints.index('wrist_3_joint')]
        return
        
    def ee_pose(self):
        gripper_pose = self.move_group.get_current_pose().pose
        return gripper_pose

    def ee_rpy(self, request):
        gripper_rpy = self.move_group.get_current_rpy()
        return gripper_rpy
            
