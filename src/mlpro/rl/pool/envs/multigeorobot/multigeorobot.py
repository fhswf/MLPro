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

        LoadYamlFileParamsTest(rospackage_name="multi_geo_robot_training",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="multi_geo_robot.yaml")  

        # Init OpenAI_ROS ENV
        task_and_robot_environment_name = rospy.get_param(
        '/multi_geo_robot/task_and_robot_environment_name')

        max_step_episode = rospy.get_param(
        '/multi_geo_robot/max_iterations')  

        self.env = StartOpenAI_ROS_Environment(task_and_robot_environment_name, max_step_episode)       
        
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)
        
        self.reset()

## -------------------------------------------------------------------------------------------------
    def setup_spaces(self):
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

        for idx in range(len(self.env.joints)):
            action_space.add_dim(Dimension(idx, 'J%i'%(idx), 'Joint%i'%(idx), '', 'rad', 'rad', p_boundaries=[-0.1,0.1]))

        return state_space, action_space
    
## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        random.seed(p_seed)
        obs = self.env.reset()
        state   = State(self._state_space)
        state.set_values(obs)
        self._set_state(state)

## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:
        obs, self.reward_gym, done, info = self.env.step(p_action.get_sorted_values())
        state = State(self._state_space, p_terminal=done)
        state.set_values(obs)

        return state


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        success = (np.allclose(a=p_state.get_values()[:3], 
                            b=p_state.get_values()[3:], 
                            atol=0.05))

        return success


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        return False


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