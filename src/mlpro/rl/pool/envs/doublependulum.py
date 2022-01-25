## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envs
## -- Module  : doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-25  0.0.0     WB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-01-25)

This module provides an RL environment of double pendulum.
"""


from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
from numpy import sin, cos
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
from collections import deque



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DoublePendulum (Environment):
    """
    This is the main class of the Double Pendulum environment that inherits 
    Environment class from MLPro.
    
    Parameters
    ----------
    p_logging : Log, optional
        logging functionalities. The default is Log.C_LOG_ALL.
    t_step : float, optional
        time for each time step (in seconds). The default is 0.01.
    t_act : int, optional
        action frequency (with respect to the time step). The default is 20.
    max_torque : float, optional
        maximum torque applied to pendulum 1. The default is 20.
    max_speed : float, optional
        maximum speed applied to pendulum 1. The default is 10.
    l1 : float, optional
        length of pendulum 1 in m. The default is 0.5
    l2 : float, optional
        length of pendulum 2 in m. The default is 0.5
    m1 : float, optional
        mass of pendulum 1 in kg. The default is 0.5
    m2 : float, optional
        mass of pendulum 2 in kg. The default is 0.5
    th1 : float, optional 
        initial angle of pendulum 1 in degrees. The default is 0.0
    th1dot : float, optional 
        initial angular velocities of pendulum 1 in degrees per second. The default is 0.0
    th2 : float, optional 
        initial angle of pendulum 2 in degrees. The default is 0.0
    th2dot : float, optional 
        initial angular velocities of pendulum 2 in degrees per second. The default is 0.0
    g : float, optional
        gravitational acceleration. The default is 9.8
    history_length : int, optional
        historical trajectory points to display. The default is 50.
        
    Attributes
    ----------
    C_NAME : str
        name of the environment.
    C_CYCLE_LIMIT : int
        the number of cycle limit.
    C_LATENCY : timedelta()
        latency.
    C_REWARD_TYPE : Reward
        rewarding type.
    """
    C_NAME              = "DoublePendulum"
    C_CYCLE_LIMIT       = 0
    C_LATENCY           = timedelta(0,1,0)    
    C_REWARD_TYPE       = Reward.C_TYPE_OVERALL
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, t_step=0.01, t_act=20, max_torque=20,
                max_speed=10, l1=0.5, l2=0.5, m1=0.5, m2=0.5, th1=0.0, th2=0.0, 
                th1dot=0.0, th2dot=0.0, g=9.8, history_length=50)
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)
        self.t_step = t_step
        self.t_act = t_act
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.th1 = th1
        self.th2 = th2
        self.th1dot = th1dot
        self.th2dot = th2dot
        self.g = g
        self.history_length=50
        
        self.reset()  


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        """
        This method is used to setup action and state spaces of the system.

        Returns
        -------
        state_space : ESpace()
            state space of the system.
        action_space : ESpace()
            action space of the system.

        """
        state_space     = ESpace()
        action_space    = ESpace()

        state_space.add_dim(Dimension(0, 'theta 1', 'th1', 'Angle of Pendulum 1', '', 'degrees', '\textdegrees',[-np.pi, np.pi]))
        state_space.add_dim(Dimension(0, 'omega 1', 'w1', 'Angular Velocity of Pendulum 1', '', 'degrees/second', '\textdegrees/s',[]))
        state_space.add_dim(Dimension(0, 'theta 2', 'th2', 'Angle of pendulum 2', '', 'degrees', '\textdegrees',[-np.pi, np.pi]))
        state_space.add_dim(Dimension(0, 'omega 2', 'w2', 'Angular Velocity of Pendulum 2', '', 'degrees/second', '\textdegrees/s',[]))
        
        action_space.add_dim(Dimension(0, 'E-0 Act', 'R', 'Env-0 Actuator Control', '', '', '', []))

        return state_space, action_space

