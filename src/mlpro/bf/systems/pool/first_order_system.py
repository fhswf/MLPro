## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.systems.pool
## -- Module  : first_order_system.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-05  0.1.0     ASP       Initial implementation class PT1
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-11-05)

This module provides a simple demo system that represent a first order system
"""


import random
from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.mt import Task
from mlpro.bf.math import Dimension, MSpace, ESpace
from mlpro.bf.systems import State, Action, System




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PT1 (System):
    """
    ...
    """

    C_NAME          = 'PT1'
    C_BOUNDARIES    = [-1000,1000]
    
    

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_K:float,
                  p_T: float,
                  p_sys_num:int,
                  p_id=None, 
                  p_name = None, 
                  p_num_dim:int= 1,
                  p_range_max = Task.C_RANGE_NONE, 
                  p_visualize = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        """
        Initialisiert ein PT1-Glied.

        Parameters
        ----------
        p_K : float
            Gain factor of the system.
        p_T : float
            Time constant of the system.
        """

        self.K = p_K          
        self.T = p_T          
        self._y_prev = 0.0   
        self._previous_tstamp =None
        self._sys_num = p_sys_num

        super().__init__( p_id = p_id, 
                          p_name = p_name,
                          p_range_max = p_range_max, 
                          p_mode = Mode.C_MODE_SIM, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging )       

        self._state_space, self._action_space = self._setup_spaces(p_num_dim=p_num_dim)
        

## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_num_dim: int):

        state_action_space : MSpace = ESpace()
        start = self._sys_num*2
        stop = p_num_dim + self._sys_num*2

        for i in range(start,stop):
            state_action_space.add_dim( p_dim = Dimension( p_name_short = 'var ' + str(i),
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
        u= p_action.get_elem(p_id=agent_id).get_values()[0]

        # get current state
        self._y_prev = p_state.values[0]

        dt =0#p_step    

        #get timestamp
        tstamp=p_action.get_tstamp()

        if self._previous_tstamp is not None:
            dt = tstamp- self._previous_tstamp
            dt = dt.total_seconds()

        # rekursions function of first oder system
        y = (dt* self.K * u + self.T * self._y_prev) / (self.T + dt)  

        #save current timestamp
        self._previous_tstamp= tstamp

        #set values of new state state 
        new_state.values = [y]    

        return new_state