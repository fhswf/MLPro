## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.systems.pool
## -- Module  : second_order_system.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-09  0.1.0     ASP       Initial implementation class PT2
## -- 2024-11-10  0.2.0     ASP       class PT2: update methods __init__(), _setup_spaces()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-11-10)

This module provides a simple demo system that represent second order system.
    Further infos : https://www.circuitbread.com/tutorials/second-order-systems-2-3

Simulation approximated with the Runge-Kutta algorithm. 
    Further infos : https://de.wikipedia.org/wiki/Runge-Kutta-Verfahren
"""


import random
import numpy as np
from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.mt import Task
from mlpro.bf.math import Dimension, MSpace, ESpace
from mlpro.bf.systems import State, Action, System




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PT2 (System):
    """
    class second-order-system
    """

    C_NAME          = 'PT2'
    C_BOUNDARIES    = [-1000,1000]
    
    

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_K:float,
                  p_D: float,
                  p_omega_0:float,
                  p_sys_num:int,
                  p_max_cycle:int,
                  p_id=None, 
                  p_name = None, 
                  p_range_max = Task.C_RANGE_NONE, 
                  p_visualize = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        """
        Initialsize second-order-system.

        Parameters
        ----------
        p_K : float
            Gain factor of the system.
        p_D : float
            Damping ration of the system.
        p_omega_0 : float
            Characteristic frequency of the system.
        p_sys_num : float
            Num id of the system.
        p_max_cycle : float
            Number of max cycles.        
        """

        self.K = p_K
        self._D =p_D
        self._omega_0 =p_omega_0
        self._sys_num = p_sys_num
        self._y = np.zeros(p_max_cycle+1)  
        self._dy = np.zeros(p_max_cycle+1)
        self._cycle = 1  

        super().__init__( p_id = p_id, 
                          p_name = p_name,
                          p_range_max = p_range_max, 
                          p_mode = Mode.C_MODE_SIM, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging )       

        self._state_space, self._action_space = self._setup_spaces(p_sys_num=p_sys_num)
        

## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_sys_num: int):

        state_action_space : MSpace = ESpace()    
        state_action_space.add_dim( p_dim = Dimension( p_name_short = 'SYS ' + str(p_sys_num),
                                                           p_base_set = Dimension.C_BASE_SET_R,
                                                           p_boundaries = self.C_BOUNDARIES ) )
        
        return state_action_space, state_action_space
    

  
## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        random.seed( p_seed )
        new_state = State( p_state_space = self.get_state_space(), p_initial = True )
        self._set_state( p_state = new_state )
        self._cycle = 1 


## -------------------------------------------------------------------------------------------------
    def acceleration(self,p_y:float, p_dy:float, p_u:float):

        """
        Initialsize second-order-system.

        Parameters
        ----------
        p_y  : float
            Previous value ofcotrolled variable.
        p_dy : float
            Derivation value of previous cotrolled variable.
        p_u  : float
            Value of control variable    
        """

        #Calculating the second derivative (acceleration)
        return -2 * self._D * self._omega_0 * p_dy - self._omega_0**2 * p_y + self._omega_0**2 * p_u


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step = None):
        
        # get action id
        agent_id  = p_action.get_agent_ids()[0]

        #create a new state 
        new_state = State( p_state_space = self.get_state_space())

        # get control Variable
        u= p_action.get_elem(p_id=agent_id).get_values()[0]

        dt=p_step 

        # Calculation RK4-Koeffizienten 
        k1_y = self._dy[self._cycle-1] * dt
        k1_dy = self.acceleration(self._y[self._cycle-1], self._dy[self._cycle-1], u) * dt

        k2_y = (self._dy[self._cycle-1] + 0.5 * k1_dy) * dt
        k2_dy = self.acceleration(self._y[self._cycle-1] + 0.5 * k1_y, self._dy[self._cycle-1] + 0.5 * k1_dy, u) * dt

        k3_y = (self._dy[self._cycle-1] + 0.5 * k2_dy) * dt
        k3_dy = self.acceleration(self._y[self._cycle-1] + 0.5 * k2_y, self._dy[self._cycle-1] + 0.5 * k2_dy, u) * dt

        k4_y = (self._dy[self._cycle-1] + k3_dy) * dt
        k4_dy = self.acceleration(self._y[self._cycle-1] + k3_y, self._dy[self._cycle-1] + k3_dy, u) * dt

        # update von y und dy
        self._y[self._cycle] = self._y[self._cycle-1] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        self._dy[self._cycle] = self._dy[self._cycle-1] + (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
        
        #set values of new state state 
        new_state.values = [self._y[self._cycle]]
        self._cycle+=1    

        return new_state
