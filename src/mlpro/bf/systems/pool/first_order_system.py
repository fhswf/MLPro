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
#                                      - update singature of __init___()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2024-11-10)
This module provides a simple demo system that represent a first-order-system
    Further information: https://www.circuitbread.com/tutorials/first-order-systems-2-2
"""

import random
from datetime import timedelta
from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.mt import Task
from mlpro.bf.math import Dimension, MSpace, ESpace
from mlpro.bf.systems import State, Action, System




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PT1 (System):
    """
    Class first-order-system
    """

    C_NAME          = 'PT1'
    C_BOUNDARIES    = [-1000,1000]
    C_PLOT_ACTIVE   = False

    C_LATENCY       = timedelta( seconds = 1 )

    ## -------------------------------------------------------------------------------------------------
    def __init__( self,                  
                  p_K:float,
                  p_T: float,
                  p_sys_num:int,
                  p_id=None,
                  p_name = C_NAME,
                  p_latency : timedelta = None,
                  p_range_max = Task.C_RANGE_NONE, 
                  p_visualize = False, 
                  p_logging=Log.C_LOG_ALL ):
        

        """
        Initialsize first-order-system.

        Parameters
        ----------
        p_K : float
            Gain factor of the system.
        p_T : float
            Time constant of the system.
        p_sys_num : float
            Num id of the system
        """
        
        super().__init__( p_id = p_id, 
                          p_name = p_name,
                          p_range_max = p_range_max, 
                          p_mode = Mode.C_MODE_SIM, 
                          p_latency = p_latency,
                          p_visualize = False, 
                          p_logging = p_logging )
        
        self.K = p_K          
        self.T = p_T
        self._sys_num = p_sys_num
        
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


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step = None):
        
        # get action id
        agent_id  = p_action.get_agent_ids()[0]

        #create a new state 
        new_state = State( p_state_space = self.get_state_space())

        # get control Variable
        u = p_action.get_elem(p_id=agent_id).get_values()[0]

        # get current state
        y_prev = p_state.values[0]

        dt = p_step 
        
        # rekursions function of first oder system
        y = (dt* self.K * u + self.T * y_prev) / (self.T + dt) 
        
        #set values of new state state 
        new_state.values = [y]    

        return new_state