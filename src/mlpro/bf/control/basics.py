## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-08-31  0.0.0     DA       Creation 
## -- 2024-09-04  0.1.0     DA       Updates on class design
## -- 2024-09-07  0.2.0     DA       Classes CTRLError, Controller: design updates
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-09-07)

This module provides basic classes around the topic closed-loop control.

"""

from mlpro.bf.various import Log
from mlpro.bf.mt import Task, Workflow
from mlpro.bf.math import Function
from mlpro.bf.streams import InstDict, Instance, StreamTask, StreamWorkflow, StreamShared, StreamScenario
from mlpro.bf.systems import ActionElement, Action, System
from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SetPoint (Instance):
    """
    """

## -------------------------------------------------------------------------------------------------
    def _get_values(self):
        return self.get_feature_data().get_values()


## -------------------------------------------------------------------------------------------------
    def _set_values(self, p_values):
        self.get_feature_data().set_values( p_values = p_values)


## -------------------------------------------------------------------------------------------------
    values = property( fget=_get_values, fset=_set_values )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CTRLError (Instance):
    """
    """

## -------------------------------------------------------------------------------------------------
    def _get_values(self):
        return self.get_feature_data().get_values()


## -------------------------------------------------------------------------------------------------
    def _set_values(self, p_values):
        self.get_feature_data().set_values( p_values = p_values)


## -------------------------------------------------------------------------------------------------
    values = property( fget=_get_values, fset=_set_values )





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
    def _compute_action( self, 
                         p_ctrl_error : CTRLError, 
                         p_action_element : ActionElement,
                         p_ctrl_id : int = 0,
                         p_ae_id : int = 0 ):
        """
        Custom method to compute and an action based on an incoming control error. The result needs
        to be stored in the action element handed over. I/O values can be accessed as follows:

        SISO
        ----
        Get single error value: error_siso = p_ctrl_error.values[p_ctrl_id]
        Set single action value: p_action_element.values[p_ae_id] = action_siso

        MIMO
        ----
        Get multiple error values: error_mimo = p_ctrl_error.values
        Set multiplie action values: p_action_element.values = action_mimo


        Parameters
        ----------
        p_ctrl_error : CTRLError
            Control error.
        p_action_element : ActionElement
            Action element to be filled with resulting action value(s).
        p_ctrl_id : int = 0
            SISO controllers only. Id of the related source value in p_ctrl_error.
        p_ae_id : int = 0 
            SISO controller olny. Id of the related destination value in p_action_element.

        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControllerFct (Controller):
    """
    Wrapper class for controllers based on a mathematical function mapping an error to an action.

    Parameters
    ----------
    p_fct : Function
        Function object mapping a control error to an action

    See class Controller for further parameters.
    """

    C_TYPE          = 'Controller Fct'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_fct : Function,
                  p_name: str = None, 
                  p_range_max=Task.C_RANGE_THREAD, 
                  p_duplicate_data: bool = False, 
                  p_visualize: bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )

        self._fct : Function = p_fct


## -------------------------------------------------------------------------------------------------
    def set_parameter(self, **p_param):
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_ctrl_error: CTRLError) -> Action:

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
class ControlSystem (StreamTask):
    """
    Wrapper class for state-based systems.
    """

    C_TYPE          = 'Control System'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_system : System,
                  p_name: str = None, 
                  p_range_max=Task.C_RANGE_THREAD, 
                  p_duplicate_data: bool = False, 
                  p_visualize: bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )

        self._system : System = p_system





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlShared (StreamShared):

## -------------------------------------------------------------------------------------------------
    def change_setpoint( p_setpoint : SetPoint ):

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlCycle (StreamWorkflow):
    """
    Container class for all tasks of a control cycle.
    """

    C_TYPE          = 'Control Cycle'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = Workflow.C_RANGE_THREAD, 
                  p_class_shared = ControlShared, 
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        super().__init__( p_name=p_name, 
                          p_range_max=p_range_max, 
                          p_class_shared=p_class_shared, 
                          p_visualize=p_visualize,
                          p_logging=p_logging, 
                          **p_kwargs )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlScenario ( StreamScenario ):
    """
    ...
    """

## -------------------------------------------------------------------------------------------------
    def get_control_board(self) -> ControlShared:

        raise NotImplementedError
