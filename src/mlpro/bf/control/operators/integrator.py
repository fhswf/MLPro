## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.operators
## -- Module  : integrator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-07  0.1.0     DA       Creation and initial implementation
## -- 2024-10-09  0.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-10-09)

This module provides an implementation of an integrator that determins the next control variable by
buffering and cumulating it.

"""

import numpy as np

from mlpro.bf.math.basics import Log
from mlpro.bf.mt import Log, Task
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.bf.systems import ControlVariable
from mlpro.bf.control import Operator




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Integrator (Operator):
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
        
        self._ctrl_var : ControlVariable = None

        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        ctrl_var = self.integrate( p_ctrl_var = self._get_instance( p_inst = p_inst, p_type = ControlVariable, p_remove = True ) )
        p_inst[ctrl_var.id] = (InstTypeNew, ctrl_var )


## -------------------------------------------------------------------------------------------------
    def integrate(self, p_ctrl_var : ControlVariable) -> ControlVariable:
        """
        Numerically integrates the incoming control variable.

        Parameters
        ----------
        p_ctrl_var : ControlVariable
            Control variable to be cumulated.

        Returns
        -------
        ControlVariable
           Integrated control variable.
        """

        if self._ctrl_var is None:
            self._ctrl_var = p_ctrl_var.copy()
        else:
            for controller_id in p_ctrl_var.get_agent_ids():
                controller_src = p_ctrl_var.get_elem( p_id = controller_id )
                controller_dst = self._ctrl_var.get_elem( p_id = controller_id )
                controller_dst.values = np.add( np.array(controller_dst.values), np.array(controller_src.values) )

        return self._ctrl_var