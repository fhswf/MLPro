## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.operators
## -- Module  : comparator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-06  0.1.0     DA       Creation and initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-10-06)

This module provides an implementation of a comparator that determins the control error based on 
setpoint and controlled variable (system state).

"""

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.math import Element
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.bf.systems import State
from mlpro.bf.control import SetPoint, ControlError, Operator



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Comparator (Operator):
    """
    The comparator computes the control error based on the current setpoint and control system state.
    It consumes (not to say: removes) the current system state and replaces it by a control error.
    """

    C_NAME      = 'Comparator'

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        
        # 1 Get setpoint
        setpoint : SetPoint = self._get_instance( p_inst = p_inst, p_type = SetPoint )


        # 2 Get and remove current system state
        state : State = self._get_instance( p_inst = p_inst, p_type = State, p_remove = True)

        if ( setpoint is not None ) and ( state is not None ):
             control_error = self.get_control_error( p_setpoint = setpoint, p_state = state )
             control_error.id = self.get_so().get_next_inst_id()
             p_inst[control_error.id] = (InstTypeNew, control_error)
        else:
            self.log( Log.C_LOG_TYPE_W, 'Neither found a setpoint nor a state...')


## -------------------------------------------------------------------------------------------------
    def get_control_error(self, p_setpoint: SetPoint, p_state: State) -> ControlError:
        """
        Returns a new control error as the difference betweeen a specified setpoint and state.

        Parameters
        ----------
        p_setpoint : SetPoint
            Setpoint object.
        p_state : State
            State object.

        Returns
        -------
        ControlError
            New control error object.
        """

        error_data   = Element( p_set=p_state.get_feature_data().get_related_set() )
        error_values = np.subtract( np.array(p_state.values), np.array(p_setpoint.values) )
        error_data.set_values( p_values = error_values )
        return ControlError( p_error_data = error_data, p_tstamp = p_state.tstamp )
