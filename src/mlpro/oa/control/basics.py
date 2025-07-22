## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-12  0.0.0     DA       Creation 
## -- 2024-09-16  0.1.0     DA       Initial implementation of class OAController
## -- 2024-09-19  0.2.0     DA       Completion of classes and their parents
## -- 2024-09-27  0.3.0     DA       New method OAController hdl_setpoint_changed
## -- 2024-10-04  0.3.1     DA       Bugfix in OAController.__init__()
## -- 2024-10-09  0.4.0     DA       Refactoring
## -- 2024-12-05  0.5.0     DA       Refactoring and code cleanup
## -- 2024-12-06  0.5.1     DA       Bugfixes in methods OAController.__init__(), ._run()
## -- 2025-06-11  0.6.0     DA       Refactoring
## -- 2025-07-22  0.8.0     DA       Refactoring: __all__ export list, docstring, imports
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.8.0 (2025-07-22)

This module provides basic classes around the topic online-adaptive closed-loop control.

"""


from mlpro.bf import Log
from mlpro.bf.events import Event
from mlpro.bf.mt import Task
from mlpro.bf.math import MSpace
from mlpro.bf.control import Controller, get_ctrl_data, ControlError, ControlVariable
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.bf.ml import Model, Training, TrainingResults



# Export list for public API
__all__ = [ 'OAController',
            'OAMultiController',
            'OAControlledSystem',
            'OAControlPanel',
            'OAControlShared',
            'OAControlWorkflow',
            'OAControlSystem',
            'OAControlTrainingResults',
            'OAControlTraining' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAController (Controller, Model):
    """
    Template class for online-adaptive closed-loop controllers. Please implement methods _compute_action()
    and _adapt() in child classes.

    Parameters
    ----------
    p_ada : bool = True
        Boolean switch for adaptivitiy. Default = True.
    p_name : str = None
        Optional name of the task. Default is None.
    p_range_max : int = Task.C_RANGE_THREAD
        Maximum range of asynchonicity. See class Range. Default is Task.C_RANGE_PROCESS.
    p_visualize : bool = False
        Boolean switch for visualisation. Default = False.
    p_logging = Log.C_LOG_ALL
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE          = 'OA Controller'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_input_space : MSpace,
                  p_output_space : MSpace,
                  p_ada: bool = True,
                  p_id = None,
                  p_name: str = None, 
                  p_range_max = Task.C_RANGE_NONE, 
                  p_visualize: bool = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        Controller.__init__( self,
                             p_input_space = p_input_space,
                             p_output_space = p_output_space,
                             p_id = p_id,
                             p_name = p_name,
                             p_range_max = p_range_max,
                             p_visualize = p_visualize,
                             p_logging = False,
                             **p_kwargs )
        
        Model.__init__( self,
                        p_ada = p_ada,
                        p_name = p_name,
                        p_visualize = p_visualize,
                        p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances: InstDict):
        """
        Computes the next action based on the current control error and adapts on further contextual
        information like setpoint, state, action.
        """

        # 1 Get control error instance
        ctrl_error = get_ctrl_data( p_instances = p_instances, p_type = ControlError, p_remove = True )
        if ctrl_error is None:
            self.log(Log.C_LOG_TYPE_W, 'Control error instance is missing!')
            return

        # 2 Compute the next action
        ctrl_var = self.compute_output( p_ctrl_error = ctrl_error )
        p_instances[ctrl_var.id] = (InstTypeNew, ctrl_var)

        # 3 Adapt
        self.adapt( p_ctrl_error = ctrl_error, p_ctrl_var = ctrl_var )


## -------------------------------------------------------------------------------------------------
    def _adapt( self, p_ctrl_error: ControlError, p_ctrl_var: ControlVariable ) -> bool:
        """
        Specialized custom method for online adaptation in closed-loop control systems.

        Parameters
        ----------
        p_ctrl_error : ControlError
            Control error.
        p_ctrl_var : ControlVariable
            Control variable.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def hdl_setpoint_changed(self, p_event_id:str, p_event_object:Event):
        """
        Handler method to be registered for event ControlPanel.C_EVNET_ID_SETPOINT_CHG. Turns off
        the adaptation for one cycle.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAMultiController: # (MultiController, Model):
    """
    """

    C_TYPE          = 'OA Multi-Controller'
    C_NAME          = ''





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlledSystem: # (ControlledSystem, Model):
    """
    Wrapper class for online-adaptive state-based systems.
    """

    C_TYPE          = 'OA Controlled System'
    C_NAME          = ''





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlPanel: # (ControlPanel):
    """
    Enables external control of a closed-loop control.
    """

    C_TYPE          = 'OA Control Panel'
    C_NAME          = '????'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlShared: # (ControlShared, OAControlPanel):
    """
    ...
    """

    pass
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlWorkflow: # (ControlWorkflow, Model):
    """
    Container class for all tasks of a control workflow.
    """

    C_TYPE          = 'OA Control Workflow'
    C_NAME          = ''





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlSystem: # (ControlSystem, Model):
    """
    ...
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlTrainingResults (TrainingResults):
    """
    ...
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlTraining (Training):
    """
    ...
    """

    pass