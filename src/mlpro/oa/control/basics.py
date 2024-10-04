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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.1 (2024-10-04)

This module provides basic classes around the topic online-adaptive closed-loop control.

"""


from mlpro.bf.systems import State, Action
from mlpro.bf.control import *
from mlpro.bf.ml import Model
from mlpro.bf.streams import InstDict




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
    p_duplicate_data : bool = False
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool = False
        Boolean switch for visualisation. Default = False.
    p_logging = Log.C_LOG_ALL
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE          = 'OA Controller'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_ada: bool = True,
                  p_name: str = None, 
                  p_range_max = Task.C_RANGE_THREAD, 
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        Controller.__init__( self,
                             p_name = p_name,
                             p_range_max = p_range_max,
                             p_duplicate_data = p_duplicate_data,
                             p_visualize = p_visualize,
                             p_logging = False,
                             **p_kwargs )
        
        Model.__init__( self,
                        p_ada = p_ada,
                        p_visualize = p_visualize,
                        p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        """
        Computes the next action based on the current control error and adapts on further contextual
        information like setpoint, state, action.
        """

        # 0 Intro
        setpoint : SetPoint         = None
        state : State               = None
        ctrl_error : ControlError   = None


        # 1 Determine the contextual data for action computation
        for (inst_type, inst) in p_inst.values():
            if isinstance(p_inst,SetPoint):
                setpoint = p_inst
            elif isinstance(p_inst,State):
                state    = p_inst
            elif isinstance(p_inst, ControlError):
                ctrl_error = p_inst

        if setpoint is None:
            raise Error( 'Setpoint instance not found. Please check control cycle.')
        if state is None:
            raise Error( 'State instance not found. Please check control cycle.')
        if ctrl_error is None:
            raise Error( 'Control error instance not found. Please check control cycle.')


        # 2 Compute the next action
        action = self.compute_action( p_ctrl_error = ctrl_error )


        # 3 Adapt
        self.adapt( p_ctrl_error = ctrl_error, 
                    p_state = state, 
                    p_setpoint = setpoint,
                    p_action = action )


## -------------------------------------------------------------------------------------------------
    def _adapt( self, 
                p_setpoint: SetPoint, 
                p_ctrl_error: ControlError, 
                p_state: State, 
                p_action: Action ) -> bool:
        """
        Specialized custom method for online adaptation in closed-loop control scenarios.

        Parameters
        ----------
        p_setpoint : SetPoint
            Setpoint.
        p_ctrl_error : ControlError
            Control error.
        p_state : State
            State of control system.
        p_action : Action
            Current action of the controller.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def hdl_setpoint_changed(self, p_event_id:str, p_event_object:Event):

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAMultiController (MultiController, Model):
    """
    """

    C_TYPE          = 'OA Multi-Controller'
    C_NAME          = ''





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlSystem (ControlSystem, Model):
    """
    Wrapper class for state-based systems.
    """

    C_TYPE          = 'OA Control System'
    C_NAME          = ''





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlPanel (ControlPanel):
    """
    Enables external control of a closed-loop control.
    """

    C_TYPE          = 'OA Control Panel'
    C_NAME          = '????'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlShared (ControlShared, OAControlPanel):
    """
    ...
    """

    pass
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlCycle (ControlCycle, Model):
    """
    Container class for all tasks of a control cycle.
    """

    C_TYPE          = 'OA Control Cycle'
    C_NAME          = ''





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlScenario (ControlScenario, Model):
    """
    ...
    """

    pass
