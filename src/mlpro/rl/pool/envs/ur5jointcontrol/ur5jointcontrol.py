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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.6 (2021-12-21)

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
        rospy.init_node('ur5_lab_training_start', anonymous=True, log_level=rospy.WARN)

        LoadYamlFileParamsTest(rospackage_name="ur5_lab",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="ur5_simple_task_param.yaml")
                    
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)

        # Init OpenAI_ROS ENV
        task_and_robot_environment_name = rospy.get_param('/ur5_lab/task_and_robot_environment_name')
    
        max_step_episode = rospy.get_param('/ur5_lab/max_iterations')

        self.env = StartOpenAI_ROS_Environment(task_and_robot_environment_name, max_step_episode)
        
        self.reset()
                
                
## -------------------------------------------------------------------------------------------------
    def _obs_to_state(self, observation):
        state = State(self._state_space)
        state.set_values(observation)
        return state
        

## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        # Setup state space
        state_space = ESpace()

        state_space.add_dim(Dimension(0, 'Px', 'PositionX', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        state_space.add_dim(Dimension(1, 'Py', 'PositionY', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        state_space.add_dim(Dimension(2, 'Pz', 'PositionZ', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        state_space.add_dim(Dimension(3, 'Tx', 'Targetx', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        state_space.add_dim(Dimension(4, 'Ty', 'Targety', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
        state_space.add_dim(Dimension(5, 'Tz', 'Targetz', '', 'm', 'm', 
                                p_boundaries=[-math.inf,math.inf]))
            
        # Setup action space
        action_space = ESpace()

        for idx in range(6):
            action_space.add_dim(Dimension(idx, 'J%i'%(idx), 'Joint%i'%(idx), '', 'rad', 'rad', p_boundaries=[-0.1,0.1]))

        return state_space, action_space

    
## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        random.seed(p_seed)
        obs = self.env.reset()
        self._state = self._obs_to_state(obs)
        self._state.set_success(True)
        self._state.set_broken(True)


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        obs, self.reward_gym, done, info = self.env.step(p_action.get_sorted_values())
        self._num_cycles += 1
        state = self._obs_to_state(obs)
        close = np.allclose(a=obs[:3], b=obs[3:], atol=0.05)
        
        if not done:
            state.set_broken(False)
            state.set_success(False)
        elif not close and ( self._num_cycles < self.get_cycle_limit() ):
            state.set_broken(True)
            
        return state


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        return self.get_success()


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        return self.get_broken()


## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self) -> None: 
        obs = self.env.get_observation()
        
        close = np.allclose(a=obs[:3], b=obs[3:], atol=0.05)
        if close:
            self._state.set_success(True)


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        reward = Reward(Reward.C_TYPE_OVERALL)
        reward.set_overall_reward(self.reward_gym)
        self.reward = reward
        return self.reward


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        pass

## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        return self.env._max_episode_steps