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
## -- 2024-10-13  0.4.0     DA       Refactoring
## -- 2024-11-10  0.5.0     DA       Refactoring
## -- 2024-11-26  0.6.0     DA       Method Comparator._run(): creation of ControlError only if
## --                                both SetPoint and ControlledVariable are detected
## -- 2024-12-03  0.6.1     DA       Bugfix in method Comparator.get_control_error()
## -- 2025-06-11  0.7.0     DA       Refactoring
## -- 2025-07-18  0.8.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.8.0 (2025-07-18)

This module provides an implementation of a comparator that determins the control error based on 
setpoint and controlled variable (system state).

"""

import numpy as np

from mlpro.bf import Log
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.bf.control import SetPoint, ControlledVariable, ControlError, Operator, get_ctrl_data



# Export list for public API
__all__ = [ 'Comparator' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Comparator (Operator):
    """
    The comparator computes the control error based on the current setpoint and controlled variable.
    It consumes (not to say: removes) the current controlled variable and replaces it by a control error.
    """

    C_NAME      = 'Comparator'

## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict):
        
        # 1 Get setpoint
        setpoint : SetPoint = get_ctrl_data( p_instances = p_instances, p_type = SetPoint, p_remove = True )
        if setpoint is None:
            self.log(Log.C_LOG_TYPE_E, 'Setpoint missing!')
            return


        # 2 Get and remove current controlled variable
        ctrlled_var : ControlledVariable = get_ctrl_data( p_instances = p_instances, p_type = ControlledVariable, p_remove = True )
        if ctrlled_var is None:
            self.log(Log.C_LOG_TYPE_W, 'Controlled variable missing!')
            return


        # 3 Compute control error
        control_error = self.get_control_error( p_setpoint = setpoint, p_ctrlled_var = ctrlled_var )
        p_instances[control_error.id] = (InstTypeNew, control_error)


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

        return ControlError( p_id = self.get_so().get_next_inst_id(),
                             p_value_space = p_ctrlled_var.value_space, 
                             p_values = np.subtract( np.array(p_setpoint.values), np.array(p_ctrlled_var.values) ), 
                             p_tstamp = self.get_so().get_tstamp() )
