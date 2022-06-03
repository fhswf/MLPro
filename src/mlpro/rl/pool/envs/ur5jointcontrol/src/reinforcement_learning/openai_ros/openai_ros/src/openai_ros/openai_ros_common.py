#!/usr/bin/env python
from click import launch
import gym
from .task_envs.task_envs_list import RegisterOpenAI_Ros_Env
import roslaunch
import rospy
import rospkg
import os
import sys
import subprocess


def StartOpenAI_ROS_Environment(task_and_robot_environment_name, max_episode_steps=10000):
    """
    It Does all the stuff that the user would have to do to make it simpler
    for the user.
    This means:
    0) Registers the TaskEnvironment wanted, if it exists in the Task_Envs.
    2) Checks that the workspace of the user has all that is needed for launching this.
    Which means that it will check that the robot spawn launch is there and the worls spawn is there.
    4) Launches the world launch and the robot spawn.
    5) It will import the Gym Env and Make it.
    """
    rospy.logwarn("Env: {} will be imported".format(
        task_and_robot_environment_name))
    result = RegisterOpenAI_Ros_Env(task_env=task_and_robot_environment_name,
                                    max_episode_steps=max_episode_steps)

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
        env = gym.make(task_and_robot_environment_name)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env


class ROSLauncher(object):
    def __init__(self, rospackage_name, launch_file_name, launch_arguments={}, use_popen=False, ros_ws_abspath="/home/user/simulation_ws"):

        self._rospackage_name = rospackage_name
        self._launch_file_name = launch_file_name
        self._launch_arguments = launch_arguments

        self.rospack = rospkg.RosPack()

        # Check Package Exists
        try:
            pkg_path = self.rospack.get_path(rospackage_name)
            rospy.logdebug("Package FOUND...")
        except rospkg.common.ResourceNotFound:
            rospy.logwarn("Package NOT FOUND...")

        # Now we check that the Package path is inside the ros_ws_abspath
        # This is to force the system to have the packages in that ws, and not in another.
        if ros_ws_abspath in pkg_path:
            rospy.logdebug("Package FOUND in the correct WS!")
        else:
            rospy.logwarn("Package FOUND in "+pkg_path +
                          ", BUT not in the ws="+ros_ws_abspath+"...")

        # If the package was found then we launch
        if pkg_path:
            rospy.loginfo(
                ">>>>>>>>>>Package found in workspace-->"+str(pkg_path))
            launch_dir = os.path.join(pkg_path, "launch")
            path_launch_file_name = os.path.join(launch_dir, launch_file_name)

            # Convert Launch argument from dictionary to proper argument
            launch_args = []
            for key, val in self._launch_arguments.items():
                launch_args.append(str(key)+":="+str(val))
        

            rospy.logwarn("path_launch_file_name=="+str(path_launch_file_name))
            if use_popen:
                source_env_command = "source "+ros_ws_abspath+"/devel/setup.bash"
                roslaunch_command = "roslaunch  {0} {1} {2}".format(rospackage_name, launch_file_name, launch_args)
                command = source_env_command+roslaunch_command
                rospy.logwarn("Launching command="+str(command))

                p = subprocess.Popen(['/bin/bash', '-c', source_env_command, "&&", roslaunch_command], shell=True)
                
            else:
                self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
                roslaunch.configure_logging(self.uuid)
                self.launch = roslaunch.parent.ROSLaunchParent(
                    self.uuid, [(path_launch_file_name, launch_args)])
                self.launch.start()
            


            rospy.loginfo(">>>>>>>>>STARTED Roslaunch-->" +
                          str(self._launch_file_name))
        else:
            assert False, "No Package Path was found for ROS apckage ==>" + \
                str(rospackage_name)
