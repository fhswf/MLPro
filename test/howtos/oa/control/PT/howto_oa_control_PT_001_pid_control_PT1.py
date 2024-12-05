## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_oa_control_PT_001_pid_control_PT1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-03  0.1.0     ASP       Creation
## -- 2024-12-06  0.2.0     ASP       Update RLPID
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-12-06)

The HowTo is intended to show the behavior of a first-order system in a closed loop control, without a controller

You will learn:

1) How to set up a PID control system with a PT1-controlled system

2) How to execute a control system and to change the setpoint from outside

3) The Behaviour of the PT1 System. Please vary with the system parameters K and T of the controlled System

"""

import numpy as np
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from datetime import timedelta
from mlpro.bf.systems.pool import PT1
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.controlsystems import BasicControlSystem


from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.oa.control.controllers import RLPID,wrapper_rl
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from mlpro_int_sb3.wrappers import WrPolicySB32MLPro
from mlpro.rl.models import *
from mlpro.rl.models_env import Reward
from mlpro.bf.control import ControlVariable, ControlledVariable


# 1. create a custom reward funtion
class MyReward(FctReward):

    def __init__(self, p_logging=Log.C_LOG_ALL):
        self._reward = Reward(p_value=0)
        self._reward_value =0
        self.error_streak_counter=0
        super().__init__(p_logging)


    
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable= None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        
        
                # Berechnung der Ableitung des Fehlers (Fehleränderungsrate)
        error_derivative = error_new - error_old
        
        # Inverser Reward basierend auf dem neuen Fehler (kleiner Fehler = größerer Reward)
        epsilon = 1e-6  # Kleine Zahl, um Division durch Null zu vermeiden
        reward = 20 / (abs(error_new) + epsilon)
        
        # 1. Bestrafung für große Fehler (Abweichung vom Sollwert)
        penalty_threshold = 3.0  # Schwellenwert für zu große Fehler
        if abs(error_new) > penalty_threshold:
            reward -= (abs(error_new) - penalty_threshold)  # Bestrafung für große Fehler
        
        # 2. Bestrafung für starkes Überschwingen (Wenn der Fehler das Sollwert überschreitet)
        if (error_old > 0 and error_new < 0) or (error_old < 0 and error_new > 0):
            reward -= 5.0  # Bestrafung für Überschwingen (besonders stark bestraft)
        
        # 3. Bestrafung für starke Schwankungen in der Fehleränderung (Oszillationen)
        derivative_threshold = 3  # Schwellenwert für starke Fehleränderungen
        if abs(error_derivative) > derivative_threshold:
            reward -= 2.0 * abs(error_derivative)  # Bestrafung für starke Änderungen
        
        # 4. Exponentielle Belohnung für kleine Fehler (schnelles Erreichen des Sollwerts)
        k = 20  # Tuning-Parameter für exponentielle Bestrafung
        reward += math.exp(-0.3 * (error_new*error_new))*k

        # 5. Zusätzliche Belohnung für konstant kleinen Fehler über Zeit
        # Schwellenwert für "kleinen Fehler"
        small_error_threshold = 1  # Definiere, was als "kleiner Fehler" gilt
        streak_threshold = 10  # Anzahl der aufeinanderfolgenden Schritte für zusätzliche Belohnung

        # Prüfe, ob der Fehler klein ist
        if abs(error_new) < small_error_threshold:
            self.error_streak_counter += 1  # Zähler erhöhen, wenn der Fehler klein ist
        else:
            self.error_streak_counter = 0  # Zähler zurücksetzen, wenn der Fehler größer wird

        # Wenn der Fehler für mehrere Schritte konstant klein war, zusätzliche Belohnung
        if self.error_streak_counter >= streak_threshold:
            reward += 10  # Zusätzliche Belohnung, wenn der Fehler konstant klein bleibt
            # Optional: Zähler zurücksetzen, wenn der maximale Bonus erreicht ist
            # self.error_streak_counter = 0
        reward =min(reward,30)         

        self._reward.set_overall_reward(reward)
        
        return self._reward



# 2 Preparation of demo/unit test mode
T = 5
T_l = 0.1 
cycle_limit = int(3*T/T_l)*10
if __name__ == '__main__':
    # 2.1 Parameters for demo mode
    cycle_limit = cycle_limit
    num_dim     = 1
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 2
  
else:
    # 2.2 Parameters for internal unit test
    cycle_limit = 5
    num_dim     = 4
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1

if __name__ == '__main__':


    setpoint_value =10



# 3 Setup the control system

# 3.1 Controlled system
my_ctrl_sys = PT1(p_K=5,
                p_T=T,
                p_sys_num=0,
                p_y_start=0,#setpoint_value,
                p_latency = timedelta( seconds =T_l),
                p_visualize = visualize,
                p_logging = logging )


# 3.2 Controller
# Define pid paramter space
pid_paramter_space = MSpace()
p_obs = MSpace()
dim_kp = Dimension('Kp',p_boundaries=[0.1,100])
dim_Tn = Dimension('Tn',p_unit='second',p_boundaries=[0,100])
dim_Tv= Dimension('Tv',p_unit='second',p_boundaries=[0,100]) 
pid_paramter_space.add_dim(dim_kp)        
pid_paramter_space.add_dim(dim_Tn)
pid_paramter_space.add_dim(dim_Tv)

# Set a  Policy 
policy_sb3 = PPO(
    policy="MlpPolicy",
    n_steps=1,
    env=None,
    _init_setup_model=False,
    device="cpu",learning_rate=0.003,seed=42)


poliy_wrapper = WrPolicySB32MLPro(p_sb3_policy=policy_sb3,
                                  p_cycle_limit=cycle_limit,
                                  p_observation_space=my_ctrl_sys.get_state_space(),
                                  p_action_space=pid_paramter_space)



# create pid controller
my_pid_ctrl = PIDController( p_input_space = my_ctrl_sys.get_state_space(),
                       p_output_space = my_ctrl_sys.get_action_space(),
                       p_Kp=0.59,
                       p_Tn=1,
                       p_Tv=0.5,
                       p_integral_off=False,
                       p_derivitave_off=False,
                       p_name = 'PID Controller',
                       p_visualize = visualize,
                       p_logging = logging )

#create rl pid policy
rl_pid_policy = RLPID(p_observation_space=my_ctrl_sys.get_state_space(),
                      p_action_space=my_ctrl_sys.get_action_space(),
                      p_pid_controller = my_pid_ctrl,
                      p_policy=poliy_wrapper,
                       p_visualize = visualize,
                       p_logging = logging )

#create OAControllerRL
my_ctrl = wrapper_rl.OAControllerRL(p_input_space=MSpace()
                                    ,p_output_space=MSpace()
                                    ,p_rl_policy=rl_pid_policy
                                    ,p_rl_fct_reward=MyReward()
                                    ,p_visualize = visualize
                                    ,p_logging = logging)


# 3.3 Basic control system
mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                      p_controller = my_ctrl,
                                      p_controlled_system = my_ctrl_sys,
                                      p_name = 'First-Order-System',
                                      p_ctrl_var_integration = False,
                                      p_cycle_limit = cycle_limit,
                                      p_visualize = visualize,
                                      p_logging = logging )



# 4 Set initial setpoint values and reset the controlled system
mycontrolsystem.get_control_panels()[0][0].set_setpoint( p_values = np.ones(shape=(num_dim))*setpoint_value )
my_ctrl_sys.reset( p_seed = 1 )



# 5 Run some control cycles
if ( __name__ == '__main__' ) and visualize:
    mycontrolsystem.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                             p_view_autoselect = True,
                                                             p_step_rate = step_rate,
                                                             p_plot_horizon = 100 ) )
    input('\nPlease arrange all windows and press ENTER to start control processing...')

mycontrolsystem.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')

