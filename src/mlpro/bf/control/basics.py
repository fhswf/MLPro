## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-08-31  0.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-08-31)

This module provides ...
"""

from mlpro.bf.streams.basics import InstDict, Instance, StreamTask, StreamWorkflow
from mlpro.bf.systems import Action, System




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SetPoint (Instance):
    """
    """

    pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CTRLError (Instance):
    """
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Controller (StreamTask):
    """
    Template class for closed-loop controllers.
    """

    C_TYPE          = 'Controller'
    C_NAME          = '????'


## -------------------------------------------------------------------------------------------------
    def set_parameter(self, **p_param):
        pass


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_ctrl_error: CTRLError) -> Action:
        """
        Custom method to compute and return an action based on an incoming control error.

        Parameters
        ----------
        p_ctrl_error : CTRLError
            Control error.


        Returns
        -------

        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiController (Controller, StreamWorkflow):
    """
    """

    C_TYPE          = 'Multi-Controller'
    C_NAME          = ''





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlSystem (Controller, StreamWorkflow):
    """
    """

    C_TYPE          = 'Control System'
    C_NAME          = ''
