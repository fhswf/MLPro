## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.pool.envs.ur5jointcontrol
## -- Module  : urjointcontrol.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-13  0.0.0     WB       Creation
## -- 2021-09-13  1.0.0     WB       Released first version
## -- 2021-09-13  1.0.1     WB       Instantiated without WrEnvGym
## -- 2021-09-23  1.0.2     WB       Increased C_LATENCY
## -- 2021-10-05  1.0.3     SY       Update following new attributes done and broken in State
## -- 2021-12-03  1.0.4     DA       Refactoring
## -- 2021-12-20  1.0.5     DA       Replaced 'done' by 'success'
## -- 2021-12-20  1.0.6     WB       Update 'success' and 'broken' rule
## -- 2021-12-21  1.0.7     DA       Class UR5JointControl: renamed method reset() to _reset()
## -- 2022-01-20  1.0.8     MRD      Use the gym wrapper to wrap the ur5 environment
## -- 2022-03-04  1.1.0     WB       Adds the ability to control gripper
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2022-03-04)

This module provides an environment with multivariate state and action spaces 
based on the Gym-based environment 'UR5RandomTargetTask-v0'. 
"""

from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
import mlpro
import numpy as np
import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import subprocess


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class UR5JointControl(WrEnvGYM2MLPro):
    """
    This environment multivariate space and action spaces by duplicating the
    Gym-based environment 'UR5RandomTargetTask-v0'. 
    """

    C_NAME = 'UR5JointControl'
    C_LATENCY = timedelta(0, 5, 0)
    C_INFINITY = np.finfo(np.float32).max

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_seed=0, p_visualize=False, p_logging=True):
        """
        Parameters:
            p_logging       Boolean switch for logging
        """
        roscore = subprocess.Popen('roscore')
        rospy.init_node('ur5_lab_training_start', anonymous=True, log_level=rospy.WARN)

        LoadYamlFileParamsTest(rospackage_name="ur5_lab",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="ur5_simple_task_param.yaml")

        ros_ws_path = mlpro.rl.pool.envs.ur5jointcontrol.__file__.replace("/__init__.py", "")
        rospy.set_param('ros_ws_path', ros_ws_path)
        rospy.set_param('visualize', p_visualize)

        # Init OpenAI_ROS ENV
        task_and_robot_environment_name = rospy.get_param('/ur5_lab/task_and_robot_environment_name')

        max_step_episode = rospy.get_param('/ur5_lab/max_iterations')

        env = StartOpenAI_ROS_Environment(task_and_robot_environment_name, max_step_episode)
        env.seed(p_seed)

        super().__init__(p_gym_env=env)

    ## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        obs = p_state.get_values()
        close = np.allclose(a=obs[:3],
                            b=obs[3:],
                            atol=0.1)

        if close:
            self._state.set_terminal(True)

        return close
