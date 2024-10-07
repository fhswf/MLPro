## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.operators
## -- Module  : cumulator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-07  0.1.0     DA       Creation and initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-10-07)

This module provides an implementation of a cumulator that determins the next control action by
buffering and cumulating it.

"""

import numpy as np

from mlpro.bf.math.basics import Log
from mlpro.bf.mt import Log, Task
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.bf.systems import Action
from mlpro.bf.control import Operator




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Cumulator (Operator):
    """
    Operator that determins the next control action by buffering and cumulating it. The origin action
    provided by a controller is replaced.
    """

    C_NAME      = 'Cumulator'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max=Task.C_RANGE_THREAD, 
                  p_duplicate_data: bool = False, 
                  p_visualize: bool = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        self._action : Action = None

        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        next_action = self.cumulate( p_action = self._get_instance( p_inst = p_inst, p_type = Action, p_remove = True ) )
        p_inst[next_action.id] = (InstTypeNew, next_action )


## -------------------------------------------------------------------------------------------------
    def cumulate(self, p_action : Action) -> Action:
        """
        Returns a new control error as the difference betweeen a specified setpoint and state.

        Parameters
        ----------
        p_action : Action
            Action to be cumulated.

        Returns
        -------
        Action
           Cumulated action.
        """

        if self._action is None:
            self._action = p_action.copy()
        else:
            for controller_id in p_action.get_agent_ids():
                controller_src = p_action.get_elem( p_id = controller_id )
                controller_dst = self._action.get_elem( p_id = controller_id )
                controller_dst.values = np.add( np.array(controller_dst.values), np.array(controller_src.values) )

        return self._action