## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.systems.pool
## -- Module  : first_order_system.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-05  0.1.0     ASP      Initial implementation class PT1
## -- 2024-11-05  0.2.0     ASP      class PT1: update methods __init__(), _setup_spaces()
## -- 2024-11-10  0.3.0     ASP      class PT1: update methods __init__(), _setup_spaces()
##                                       - update singature of __init___()
## -- 2024-11-16  0.4.0     ASP      class PT1: update methods _simulate_reaction()
##                                       - changed dt = p_step to dt = p_step.total_seconds() 
## -- 2024-12-03  0.5.0     ASP      class PT1: update methods _reset()
##                                       - add start state of the system
## -- 2024-12-30  0.6.0     ASP      class PT1: Refactoring
##                                      - add C_SAMPLE_FREQ : Specifies how often the system is sampled in a control cycle
##                                      - add self._dt: Sampling time
##                                      - update _simulate_reaction(), _reset()
## -- 2025-01-05  0.7.0     ASP       class PT1: Refactoring
##                                      - changed self.K to self._K
##                                      - changed self.T to self._T
## -- 2025-01-26  0.8.0     ASP       class PT1: Changed parameters and attributes comments
## -- 2025-07-22  0.9.0     DA       Refactoring: __all__ export list, docstring, imports
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.9.0 (2025-07-22)

This module provides a simple demo system that represent a first-order-system

Learn more:

 https://www.circuitbread.com/tutorials/first-order-systems-2-2

"""

import random
from datetime import timedelta

from mlpro.bf import Log, Mode
from mlpro.bf.mt import Task
from mlpro.bf.math import Dimension, MSpace, ESpace
from mlpro.bf.systems import State, Action, System



# Export list for public API
__all__ = [ 'PT1' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PT1 (System):
    """
    Class first-order-system

    Parameters
    ----------
    p_K             : float
        Gain factor of the system
    p_T             : float
        Time constant of the system
    p_sys_num       : int
        Num id of the system
    p_y_start       : float
        Start value of the control variable 
    p_boundaries    : float
        Boundries of the control variable 


    Attributes
    ----------
        _K           : float
            Gain factor of the system.
        _T           : float
            Time constant of the system.
        _sys_num     : int
            Num id of the system
        _y_start     : float
            Start value of the control variable 
        _y_prev      : float
            Old value of the control variable 
        _boundaries  : list
            Boundries of the control variable 
        _dt          : float
            Sampling rate 
        _state_space : MSpace
            State space of system
        _action_space : MSpace
            Action space of system   

    """

    C_NAME          = 'PT1'
    C_PLOT_ACTIVE   = False
    C_LATENCY       = timedelta( seconds = 0.1 )
    C_SAMPLE_FREQ   = 20    

## -------------------------------------------------------------------------------------------------
    def __init__( self,                  
                  p_K:float,
                  p_T: float,
                  p_sys_num:int,
                  p_y_start: float = 0,
                  p_boundaries : list = [-250,250],
                  p_id = None,
                  p_name = C_NAME,
                  p_latency : timedelta = None,
                  p_range_max = Task.C_RANGE_NONE, 
                  p_visualize = False, 
                  p_logging = Log.C_LOG_ALL ): 


        self._K = p_K          
        self._T = p_T
        self._sys_num = p_sys_num
        self._y_start = p_y_start
        self._y_prev = None
        self._boundaries = p_boundaries
        
        super().__init__( p_id = p_id, 
                          p_name = p_name,
                          p_range_max = p_range_max, 
                          p_mode = Mode.C_MODE_SIM, 
                          p_latency = p_latency,
                          p_visualize = False, 
                          p_logging = p_logging )        

        self._dt = self.get_latency().total_seconds() / self.C_SAMPLE_FREQ       
        self._state_space, self._action_space = self._setup_spaces(p_sys_num = p_sys_num)


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_sys_num: int):

        state_action_space : MSpace = ESpace()    
        state_action_space.add_dim( p_dim = Dimension( p_name_short = 'SYS ' + str(p_sys_num),
                                                       p_base_set = Dimension.C_BASE_SET_R,
                                                       p_boundaries = self._boundaries ) )
        
        return state_action_space, state_action_space    

  
## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed = None):

        random.seed( p_seed )
        new_state = State( p_state_space = self.get_state_space(), p_initial = True )    
        new_state.get_feature_data().set_values([self._y_start])    
        self._set_state( p_state = new_state )


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step = None):
        
        # get action id
        agent_id = p_action.get_agent_ids()[0]

        #create a new state 
        new_state = State( p_state_space = self.get_state_space())

        # get control Variable
        u = p_action.get_elem(p_id=agent_id).get_values()[0]

        # first run: assign start value
        if self._y_prev is None:
            self._y_prev = self._y_start
        else:
            # get current state
            self._y_prev = p_state.values[0]


        for step in range(self.C_SAMPLE_FREQ):   

            # rekursions function of first oder system
            y = (self._dt* self._K * u + self._T * self._y_prev) / (self._T + self._dt) 
            self._y_prev = y
        
        # Limit output
        y = max(self._boundaries[0],y)
        y = min(self._boundaries[1],y)


        #set values of new state state 
        new_state.values = [y]    

        return new_state