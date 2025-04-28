## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.PT
## -- Module  : howto_oa_control_102_pid_control_PT2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-03  0.1.0     ASP      Creation
## -- 2025-01-06  0.2.0     ASP      Implementation
## -- 2025-02-10  0.3.0     ASP      Update simulation parameter
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2025-02-10)

The HowTo is intended to show the behavior of a PT2 control loop with an OA PID-Controller 

You will learn:

1) How to set up a basic control system with a PT2 controlled system and OA PID-Controller

2) How to set up reward function, policy for the OA PID-Controller

3) How to execute a basic control system with an OA PID-Controller and to change the setpoint from outside

4) The behavior of the oa PID-Controller with the given policy and the resulting behavior of the PT2 control system

"""

import numpy as np
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from datetime import timedelta
from mlpro.bf.systems.pool import PT2
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.controlsystems import BasicControlSystem
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.oa.control.controllers import RLPID,wrapper_rl
from stable_baselines3 import PPO
from mlpro_int_sb3.wrappers import WrPolicySB32MLPro
from mlpro.rl.models import *
from mlpro.rl.models_env import Reward
from mlpro.bf.control import ControlledVariable



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1. create a custom reward funtion
class MyReward(FctReward):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging = Log.C_LOG_NOTHING):
        self._reward = Reward(p_value = 0)
        super().__init__(p_logging)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------    
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable= None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        e_band = 0.05
        
        reward = -abs(error_new)- error_new**2 - 3*max(abs(error_new)-e_band,0)**2 + 30*min(abs(error_new),0.03)
        self._reward.set_overall_reward(reward)     
        
        return self._reward
    

# 2 Prepare for test
if __name__ == "__main__":
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 1
    num_dim     = 1
else:
    cycle_limit = 5
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1
    num_dim     = 1


# 2.1 Define init parameters and calculate cycle limit

#init controlled systems parameter
pt2_K = 5
pt2_D = 0.5
pt2_w_0 = 0.5

# calculate cycle limit
cycle_time = 1 / 20 * pt2_w_0
simulation_time = 300 *1 / pt2_w_0
cycle_limit = 6000

# init setpoint
setpoint_value = 10


# 3 Setup closed loop 

# 3.1 controlled system 
my_ctrl_sys = PT2(p_K = pt2_K,
                    p_D = pt2_D,
                    p_omega_0 = pt2_w_0,
                    p_sys_num = 1,
                    p_max_cycle = cycle_limit,
                    p_latency = timedelta( seconds = cycle_time ),
                    p_visualize = visualize,
                    p_logging = logging )

my_ctrl_sys.reset( p_seed = 42 )
y_max = 50
my_ctrl_sys.C_BOUNDARIES = [0,y_max]


# 3.2 OA PID-Controller

# 3.2.1 Define PID-Parameter-Space
p_pid_paramter_space = MSpace() 
dim_kp = Dimension('Kp',p_boundaries = [0.1,50])
dim_Tn = Dimension('Tn',p_unit = 'second',p_boundaries = [0,300])
dim_Tv = Dimension('Tv',p_unit = 'second',p_boundaries = [0,300]) 
p_pid_paramter_space.add_dim(dim_kp)        
p_pid_paramter_space.add_dim(dim_Tn)
p_pid_paramter_space.add_dim(dim_Tv)


#3.2.2 Define PID-Output-Space
p_pid_output_space = MSpace()
p_control_dim = Dimension('u',p_boundaries = [0,500])
p_pid_output_space.add_dim(p_control_dim)


#3.2.3 Init PID-Controller
my_pid_ctrl = PIDController( p_input_space = my_ctrl_sys.get_state_space(),
                    p_output_space = my_ctrl_sys.get_action_space(),
                    p_Kp = 1,
                    p_Tn = 0,
                    p_Tv = 0,
                    p_name = 'PID Controller',
                    p_visualize = visualize,
                    p_logging = logging )  


#3.2.4 Set RL-Policy 
policy_sb3 = PPO( policy="MlpPolicy",learning_rate = 0.001, seed = 42,env = None,_init_setup_model = False)   
  

#3.2.5 Init SB3 to MLPro wrapper
poliy_wrapper = WrPolicySB32MLPro(p_sb3_policy = policy_sb3,
                                p_cycle_limit = cycle_limit,
                                p_observation_space = my_ctrl_sys.get_state_space(),
                                p_action_space = p_pid_paramter_space,p_logging = logging )


#3.2.6 Init PID-Policy
rl_pid_policy = RLPID(p_observation_space = my_ctrl_sys.get_state_space(),
                    p_action_space = p_pid_output_space,
                    p_pid_controller = my_pid_ctrl,
                    p_policy = poliy_wrapper,
                    p_visualize = visualize,
                    p_logging = logging )


#3.2.7 Init OA-PID-Controller
my_ctrl_OA = wrapper_rl.OAControllerRL(p_input_space = my_ctrl_sys.get_state_space(),
                                    p_output_space = p_pid_output_space,
                                    p_rl_policy = rl_pid_policy,
                                    p_rl_fct_reward = MyReward(),
                                    p_visualize = visualize,
                                    p_logging = logging)


# 4. Cascaded control system
mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                    p_controller = my_ctrl_OA,
                                    p_controlled_system = my_ctrl_sys,
                                    p_name = 'Second-Order-System',
                                    p_ctrl_var_integration = False,
                                    p_cycle_limit = cycle_limit,
                                    p_visualize = visualize,
                                    p_logging = logging )



# 5 Set initial setpoint values of the control system
for panel_entry in mycontrolsystem.get_control_panels():
    panel_entry[0].set_setpoint( p_values = np.ones(shape = (num_dim)) * setpoint_value )


# 6 Run some control cycles
if visualize:
    mycontrolsystem.init_plot( p_plot_settings = PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                            p_view_autoselect = True,
                                                            p_step_rate = step_rate,
                                                            p_plot_horizon = 100 ) )
    input('\nPlease arrange all windows and press ENTER to start control processing...')    
    
mycontrolsystem.run()

#switch off adaptivity 
input('Press ENTER to switch off adaptivity and restart PT1 system...')
my_ctrl_OA._adaptivity = False

# set a new cycle limit
mycontrolsystem._cycle_limit = 1000

#reset PT1 System
my_ctrl_sys.reset( p_seed = 42 )

# run control loop again
mycontrolsystem.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')