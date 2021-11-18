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

        LoadYamlFileParamsTest(rospackage_name="ur5_lab",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="ur5_simple_task_param.yaml")
                    
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)

        # Init OpenAI_ROS ENV
        task_and_robot_environment_name = rospy.get_param(
        '/ur5_lab/task_and_robot_environment_name')
    
        max_step_episode = rospy.get_param(
        '/ur5_lab/max_iterations')

        self.env = StartOpenAI_ROS_Environment(task_and_robot_environment_name, max_step_episode)
        
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
                                p_boundaries=[-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(1, 'Py', 'PositionY', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(2, 'Pz', 'PositionZ', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(3, 'Tx', 'Targetx', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(4, 'Ty', 'Targety', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        self._state_space.add_dim(Dimension(5, 'Tz', 'Targetz', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
            
        # Setup action space
        for idx in range(6):
            self._action_space.add_dim(Dimension(idx, 'J%i'%(idx), 'Joint%i'%(idx), '', 'rad', 'rad', p_boundaries=[-0.1,0.1]))

    
## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=None) -> None:
        random.seed(p_seed)
        obs = self.env.reset()
        self._state = self._obs_to_state(obs)

## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> None:
        obs, self.reward_gym, self.done, info = self.env.step(p_action.get_sorted_values())
        self._state = self._obs_to_state(obs)
        return self._state

## -------------------------------------------------------------------------------------------------
    def compute_done(self, p_state: State) -> bool:
        return self.done

## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        return False

## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self) -> None: 
        obs = self.env.get_observation()
        
        close = np.allclose(a=obs[:3], 
                            b=obs[3:], 
                            atol=0.05)
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

