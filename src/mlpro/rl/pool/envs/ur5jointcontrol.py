## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : urjointcontrol
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-13  0.0.0     WB       Creation
## -- 2021-09-13  1.0.0     WB       Released first version
## -- 2021-09-13  1.0.1     WB       Instantiated without WrEnvGym
## -- 2021-09-23  1.0.2     WB       Increased C_LATENCY
## -- 2021-10-05  1.0.3     SY       Update following new attributes done and broken in State
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2021-10-05)

This module provides an environment with multivariate state and action spaces 
based on the Gym-based environment 'UR5RandomTargetTask-v0'. 
"""


from mlpro.rl.models import *
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
    C_LATENCY   = timedelta(0,5,0)
    C_INFINITY  = np.finfo(np.float32).max      


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=True):
        """
        Parameters:
            p_logging       Boolean switch for logging
        """
        roscore = subprocess.Popen('roscore')
        rospy.init_node('ur5_lab_training_start',
                    anonymous=True, log_level=rospy.WARN)
                    
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)

        self.env = StartOpenAI_ROS_Environment(
                            'UR5RandomTargetTask-v0')
        self.reset()
                 
## -------------------------------------------------------------------------------------------------
    def _obs_to_state(self, observation):
        state = State(self._state_space)
        state.set_values(observation)
        return state
        
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
        for idx in range(6):
            self._action_space.add_dim(Dimension(idx, 'J%i'%(idx), 'Joint%i'%(idx), '', 'rad', 'rad', [-np.pi,np.pi]))

    
## -------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        obs = self.env.reset()
        self._state = self._obs_to_state(obs)

## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_action: Action) -> None:
        obs, self.reward_gym, done, info = self.env.step(p_action.get_sorted_values())
        self._state.set_done(done)
        self._state = self._obs_to_state(obs)

## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self) -> None: 
        obs = self.env.get_observation()
        
        close = np.allclose(a=obs[:3], 
                            b=obs[3:], 
                            atol=0.2)
        if close:
            self._state.set_done(True)
            self.goal_achievement = 1.0
        else:
            self.goal_achievement = 0.0

## -------------------------------------------------------------------------------------------------
    def compute_reward(self) -> Reward:
        reward = Reward(Reward.C_TYPE_OVERALL)
        reward.set_overall_reward(self.reward_gym)
        self.reward = reward
        return self.reward

