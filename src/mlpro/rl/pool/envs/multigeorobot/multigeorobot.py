## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : multigeorobot.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-19  0.0.0     MRD      Creation
## -- 2021-12-19  1.0.0     MRD      Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-12-19)

This module provides an environment for multi geometry robot.
"""

from mlpro.rl.models import *
import rospy
import subprocess

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest

class MultiGeo(Environment):
    """
    This module provides an environment for multi geometry robot.
    """

    C_NAME      = 'MultiGeo'
    C_LATENCY   = timedelta(0,5,0)
    C_INFINITY  = np.finfo(np.float32).max  

    def __init__(self, p_logging=True):
        roscore = subprocess.Popen('roscore')
        rospy.init_node('multi_geo_robot_training', anonymous=True, log_level=rospy.WARN)
                    
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)

        # Init OpenAI_ROS ENV
        task_and_robot_environment_name = rospy.get_param(
        '/multi_geo_robot/task_and_robot_environment_name')

        max_step_episode = rospy.get_param(
        '/multi_geo_robot/max_iterations')

        self.env = StartOpenAI_ROS_Environment(task_and_robot_environment_name, max_step_episode)
        
        self.reset()