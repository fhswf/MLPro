#!/usr/bin/env python
from gym import utils
import copy
import rospy
from gym import spaces
from openai_ros.robot_envs import ur5_lab_env
from gym.envs.registration import register
import numpy as np
from sensor_msgs.msg import JointState
from openai_ros.openai_ros_common import ROSLauncher
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import os

class UR5LabSimpleTask(ur5_lab_env.UR5LabEnv, utils.EzPickle):
    def __init__(self):
        # The box action version of simple_task
        # Load Params from the desired Yaml file relative to this TaskEnvironment
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/ur5_lab/config",
                               yaml_file_name="simple_task_v1.yaml")
                               
        # This is the path where the simulation files are

        ros_ws_abspath = rospy.get_param("/ur5_lab/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="ur_gazebo",
                    launch_file_name="ur5_lab_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        super(UR5LabSimpleTask, self).__init__(ros_ws_abspath)

        rospy.logdebug("Entered UR5LabSimpleTask Env")
        self.get_params()
        
        observations_high_range = np.array(
            self.upper_array_observations)
        observations_low_range = np.array(
            self.lower_array_observations)

        self.observation_space = spaces.Box(observations_low_range, observations_high_range)

        self.n_actions = len(self.upper_array_observations)
        self.action_space = spaces.Box(observations_low_range, observations_high_range)


    def get_params(self):
        # get configuration parameters

        self.n_max_iterations = rospy.get_param('/ur5_lab/max_iterations')

        init_pos_dict = rospy.get_param('/ur5_lab/init_pos')
        self.init_pos = [init_pos_dict["shoulder_pan_joint"],
                         init_pos_dict["shoulder_lift_joint"],
                         init_pos_dict["elbow_joint"],
                         init_pos_dict["wrist_1_joint"],
                         init_pos_dict["wrist_2_joint"],
                         init_pos_dict["wrist_3_joint"],
                         ]


        goal_pos_dict = rospy.get_param('/ur5_lab/goal_pos')
        self.goal_pos = [goal_pos_dict["elbow_joint"],
                         goal_pos_dict["shoulder_lift_joint"],
                         goal_pos_dict["wrist_1_joint"]]

        self.position_delta = rospy.get_param('/ur5_lab/position_delta')
        self.reached_goal_reward = rospy.get_param(
            '/ur5_lab/reached_goal_reward')

        upper_limit_dict = rospy.get_param('/ur5_lab/upper_joint_limits')
        self.upper_array = [upper_limit_dict["shoulder_pan_joint"],
                         upper_limit_dict["shoulder_lift_joint"],
                         upper_limit_dict["elbow_joint"],
                         upper_limit_dict["wrist_1_joint"],
                         upper_limit_dict["wrist_2_joint"],
                         upper_limit_dict["wrist_3_joint"],
                         ]
                         
        lower_limit_dict = rospy.get_param('/ur5_lab/lower_joint_limits')
        self.lower_array = [lower_limit_dict["shoulder_pan_joint"],
                         lower_limit_dict["shoulder_lift_joint"],
                         lower_limit_dict["elbow_joint"],
                         lower_limit_dict["wrist_1_joint"],
                         lower_limit_dict["wrist_2_joint"],
                         lower_limit_dict["wrist_3_joint"],
                         ]
                         
        self.upper_array_observations = [self.upper_array[1],self.upper_array[2],self.upper_array[3]]
        self.lower_array_observations = [self.lower_array[1], self.lower_array[2], self.lower_array[3]]

        self.n_observations = len(self.upper_array_observations)


    def _set_init_pose(self):
        # Check because it seems its not being used
        rospy.logdebug("Init Pos:")
        rospy.logdebug(self.init_pos)

        # INIT POSE
        rospy.logdebug("Moving To Init Pose ")
        self.move_joints(self.init_pos)
        self.last_action = "INIT"

        return True

    def _init_env_variables(self):
        rospy.logdebug("Init Env Variables...")
        self.iterations_done = 0
        rospy.logdebug("Init Env Variables...END")

    def _set_action(self, action):      
        gripper_target = self.get_joint_states()
        action = list(action)
        
        gripper_target[1] = float(action[0])
        gripper_target[2] = float(action[1])
        gripper_target[3] = float(action[2])
        
        self.movement_result = self.move_joints(gripper_target)

        rospy.logwarn("END Set Action ==>" + str(action) +
                      ", NAME=" + str(self.last_action))
        
    def _get_obs(self):
        joints_position = self.get_joint_states()
        obs_joints_position = [joints_position[1],joints_position[2],joints_position[3]]

        rospy.logdebug("OBSERVATIONS====>>>>>>>"+str(obs_joints_position))

        return np.array(obs_joints_position)

    def _is_done(self, observations):
        done = np.allclose(a=observations,
                            b=self.goal_pos,
                            atol=0.2)

        self.iterations_done += 1

        if self.iterations_done >= self.n_max_iterations:
            done = True

        return done

    def _compute_reward(self, observations, done):
        """
        We punish each step that it passes without achieveing the goal.
        and punish for every movement_result that is False
        Punishes differently if it reached a position that is imposible to move to.
        Rewards getting to a position close to the goal.
        """
        #implement punishment for false self.movement_result
        reward = 1.0 / (np.linalg.norm(np.array(observations) - np.array(self.goal_pos)))
        if done:
            reward += self.reached_goal_reward
        rospy.logwarn(">>>REWARD>>>"+str(reward))

        return reward
