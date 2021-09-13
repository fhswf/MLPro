## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : urjointcontrol
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-13  0.0.0     WB       Creation
## -- 2021-09-13  1.0.0     WB       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-13)

This module provides an environment with multivariate state and action spaces 
based on the Gym-based environment 'UR5RandomTargetTask-v0'. 
"""


from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGym
import numpy as np
import gym
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import subprocess





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class UR5JointControl(Environment):
    """
    This environment multivariate space and action spaces by duplicating the
    Gym-based environment 'UR5RandomTargetTask-v0'. 
    """

    C_NAME      = 'UR5JointControl'
    C_LATENCY   = timedelta(0,1,0)
    C_INFINITY  = np.finfo(np.float32).max      
    roscore = subprocess.Popen('roscore')
    rospy.init_node('ur5_lab_training_start',
                anonymous=True, log_level=rospy.WARN)
    task_and_robot_environment_name = 'UR5RandomTargetTask-v0'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=True):
        """
        Parameters:
            p_logging       Boolean switch for logging
        """

        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)

        env              = WrEnvGym(env=StartOpenAI_ROS_Environment(
                                            task_and_robot_environment_name),
                                    p_logging=p_logging,)
        self.num_joint = 6
                 
                 
## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):
        # Setup state space
        self._state_space.add_dim(Dimension(0, 'Px', 'PositionX', '', 'm', 'm', 
                                [-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(1, 'Py', 'PositionY', '', 'm', 'm', 
                                [-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(2, 'Pz', 'PositionZ', '', 'm', 'm', 
                                [-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(3, 'Tx', 'Targetx', '', 'm', 'm', 
                                [-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(4, 'Ty', 'Targety', '', 'm', 'm', 
                                [-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(5, 'Tz', 'Targetz', '', 'm', 'm', 
                                [-math.inf,math.inf]))
            
        # Setup action space
        for idx in range(self.num_joint):
            self._action_space.add_dim(Dimension(idx, 'J%i'%(idx), 'Joint%i'%(idx), '', 'rad', 'rad', [-np.pi,np.pi]))

    
## -------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.state = env.reset()
  

## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_action: Action) -> None:
        self.state, self.reward, self.done, info = env.step(p_action.get_sorted_values())
        

## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self) -> None: 
        self.state = env.get_observation()
        
        close = np.allclose(a=self.state[:3], 
                            b=self.goal_pos[3:], 
                            atol=0.2)
        if close:
            self.done = True
            self.goal_achievement = 1.0
        else:
            self.goal_achievement = 0.0

## -------------------------------------------------------------------------------------------------
    def compute_reward(self) -> Reward:
        reward = self.reward

        return reward

