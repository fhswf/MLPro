## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.PT
## -- Module  : howto_oa_control_103_pid_control_cascade.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-03  0.1.0     ASP      Creation
## -- 2025-01-06  0.2.0     ASP      Implementation
## -- 2025-02-11  0.3.0     ASP      Update simulation parameter 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-02-11)

The HowTo is intended to show the behavior of a cascaded control loop with an oa pid controller as the main controller 

You will learn:

1) How to set up a cascaded control system with a PT1 and PT2 controlled system and OA PID-Controller as main controller and P-Controller as inner controller

2) How to set up reward function, policy for the OA PID-Controller

3) How to execute a cascaded control system and to change the setpoint from outside

4) The behavior of the oa pid controller with the given policy and the resulting behavior of the cascaded control system

"""

import numpy as np
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from datetime import timedelta
from mlpro.bf.systems.pool import PT1,PT2
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.controlsystems import CascadeControlSystem
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.oa.control.controllers import RLPID,wrapper_rl
from stable_baselines3 import DDPG
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
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable= None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        e_band = 0.5
        
        reward = -abs(error_new)- error_new**2 - 10*max(abs(error_new)-e_band,0)**2
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
pt2_K = 1
pt2_D = 1.6165
pt2_w_0 = 0.00577
pt1_T = 1200
pt1_K = 25

# calculate cycle limit
simulation_time = 500 *1 / pt2_w_0
cycle_limit = 16000

# init setpoint
setpoint_value = 40


# 3 Setup inner casscade

# 3.1 controlled system 
my_ctrl_sys_1 = PT1(p_K = pt1_T,
                p_T = pt1_K,
                p_sys_num = 0,
                p_y_start = 0,
                p_latency = timedelta( seconds = 1 ),
                p_visualize = visualize,
                p_logging = logging )

my_ctrl_sys_1.reset( p_seed = 42 )   


# 3.2 P-Controller
my_ctrl_2 = PIDController( p_input_space = my_ctrl_sys_1.get_state_space(),
                    p_output_space = my_ctrl_sys_1.get_action_space(),
                    p_Kp = 0.36,
                    p_Tn = 0,
                    p_Tv = 0,
                    p_integral_off = True,
                    p_derivitave_off = True,
                    p_name = 'PID Controller2',
                    p_visualize = visualize,
                    p_logging = logging )


# 4 Setup outer casscade

# 4.1 controlled system 
my_ctrl_sys_2 = PT2(p_K = pt2_K,
                    p_D = pt2_D,
                    p_omega_0 = pt2_w_0,
                    p_sys_num = 1,
                    p_max_cycle = cycle_limit,
                    p_latency = timedelta( seconds = 4 ),
                    p_visualize = visualize,
                    p_logging = logging )

my_ctrl_sys_2.reset( p_seed = 42 )


# 4.2 OAController (Main Controller)

# 4.2.1 Define PID-Parameter-Space
p_pid_paramter_space = MSpace() 
dim_kp = Dimension('Kp',p_boundaries = [0.1,50])
dim_Tn = Dimension('Tn',p_unit = 'second',p_boundaries = [0,300])
dim_Tv = Dimension('Tv',p_unit = 'second',p_boundaries = [0,300]) 
p_pid_paramter_space.add_dim(dim_kp)        
p_pid_paramter_space.add_dim(dim_Tn)
p_pid_paramter_space.add_dim(dim_Tv)


#4.2.2 Define PID-Output-Space
p_pid_output_space = MSpace()
p_control_dim = Dimension('u',p_boundaries = [0,500])
p_pid_output_space.add_dim(p_control_dim)


#4.2.3 Init PID-Controller
my_ctrl_1 = PIDController( p_input_space = my_ctrl_sys_2.get_state_space(),
                    p_output_space = my_ctrl_sys_2.get_action_space(),
                    p_Kp = 1,
                    p_Tn = 0,
                    p_Tv = 0,
                    p_name = 'PID Controller',
                    p_visualize = visualize,
                    p_logging = logging )  


#4.2.4 Set RL-Policy 
policy_sb3 = DDPG( policy="MlpPolicy",
                learning_rate = 0.005,
                seed = 42,
                env = None,
                _init_setup_model = False,
                learning_starts = 100)    
  

#4.2.5 Init SB3 to MLPro wrapper
poliy_wrapper = WrPolicySB32MLPro(p_sb3_policy = policy_sb3,
                                p_cycle_limit = cycle_limit,
                                p_observation_space = my_ctrl_sys_2.get_state_space(),
                                p_action_space = p_pid_paramter_space,p_logging = logging )


#4.2.6 Init PID-Policy
rl_pid_policy = RLPID(p_observation_space = my_ctrl_sys_2.get_state_space(),
                    p_action_space = p_pid_output_space,
                    p_pid_controller = my_ctrl_1,
                    p_policy = poliy_wrapper,
                    p_visualize = visualize,
                    p_logging = logging )


#4.2.7 Init OA-PID-Controller
my_ctrl_OA = wrapper_rl.OAControllerRL(p_input_space = my_ctrl_sys_2.get_state_space(),
                                    p_output_space = p_pid_output_space,
                                    p_rl_policy = rl_pid_policy,
                                    p_rl_fct_reward = MyReward(),
                                    p_visualize = visualize,
                                    p_logging = logging)


# 5. Cascaded control system
mycontrolsystem = CascadeControlSystem( p_mode = Mode.C_MODE_SIM,
                                        p_controllers = [ my_ctrl_OA, my_ctrl_2],
                                        p_controlled_systems = [my_ctrl_sys_2, my_ctrl_sys_1 ],
                                        p_name = 'Stirring vessel',
                                        p_cycle_limit = cycle_limit,
                                        p_visualize = visualize,
                                        p_logging = logging )


# 6 Set initial setpoint values for all control workflows (=cascades) of the control system
for panel_entry in mycontrolsystem.get_control_panels():
    panel_entry[0].set_setpoint( p_values = np.ones(shape = (num_dim)) * setpoint_value )


# 7 Run some control cycles
if visualize:
    mycontrolsystem.init_plot( p_plot_settings = PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                            p_view_autoselect = True,
                                                            p_step_rate = step_rate,
                                                            p_plot_horizon = 100 ) )
input('\nPlease arrange all windows and press ENTER to start control processing...')    
    
mycontrolsystem.run()

#switch off adaptivity 
input('Press ENTER to switch off adaptivity and restart PT1 system and PT2 system...')
my_ctrl_OA._adaptivity = False

# set a new cycle limit
mycontrolsystem._cycle_limit = 1000

#reset PT1 System
my_ctrl_sys_1.reset( p_seed = 42 )
#reset PT2 System
my_ctrl_sys_1.reset( p_seed = 42 )

# run control loop again
mycontrolsystem.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')