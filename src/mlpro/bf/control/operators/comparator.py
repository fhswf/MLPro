## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.operators
## -- Module  : comparator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-06  0.1.0     DA       Creation and initial implementation
## -- 2024-10-08  0.2.0     DA       Validation and various changes
## -- 2024-10-09  0.3.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2024-10-09)

This module provides an implementation of a comparator that determins the control error based on 
setpoint and controlled variable (system state).

"""

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.math import Element
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.bf.control import SetPoint, ControlledVariable, ControlError, Operator



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Comparator (Operator):
    """
    The comparator computes the control error based on the current setpoint and controlled variable.
    It consumes (not to say: removes) the current controlled variable and replaces it by a control error.
    """

    C_NAME      = 'Comparator'

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        
        # 1 Get setpoint
        setpoint : SetPoint = self._get_instance( p_inst = p_inst, p_type = SetPoint )
        if setpoint is None:
            self.log(Log.C_LOG_TYPE_E, 'Setpoint missing!')
            return


        # 2 Get and remove current controlled variable
        ctrlled_var : ControlledVariable = self._get_instance( p_inst = p_inst, p_type = ControlledVariable, p_remove = True)
        if ctrlled_var is None:
            self.log(Log.C_LOG_TYPE_W, 'Controlled variable missing!')
            ctrlled_var = setpoint


        # 3 Compute control error
        control_error = self.get_control_error( p_setpoint = setpoint, p_ctrlled_var = ctrlled_var )
        control_error.id = self.get_so().get_next_inst_id()
        control_error.tstamp = self.get_so().get_tstamp()
        p_inst[control_error.id] = (InstTypeNew, control_error)


## -------------------------------------------------------------------------------------------------
    def get_control_error(self, p_setpoint: SetPoint, p_ctrlled_var: ControlledVariable ) -> ControlError:
        """
        Returns a new control error as the difference betweeen a specified setpoint and controlled
        variable.

        Parameters
        ----------
        p_setpoint : SetPoint
            Setpoint object.
        p_ctrlled_var : ControlledVariable
            Controlled variable object.

        Returns
        -------
        ControlError
            New control error object.
        """

        error_data   = Element( p_set=p_ctrlled_var.get_feature_data().get_related_set() )
        error_values = np.subtract( np.array(p_ctrlled_var.values), np.array(p_setpoint.values) )
        error_data.set_values( p_values = error_values )
        return ControlError( p_error_data = error_data, p_tstamp = p_ctrlled_var.tstamp )
