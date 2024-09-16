## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-12  0.0.0     DA       Creation 
## -- 2024-09-16  0.1.0     DA       Initial implementation of class OAController
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-09-16)

This module provides basic classes around the topic online-adaptive closed-loop control.

"""


from mlpro.bf.systems import State, Action, ActionElement
from mlpro.bf.control import SetPoint, ControlError, Controller, MultiController, ControllerFct
from mlpro.bf.ml import Model
from mlpro.bf.streams import InstDict




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAController (Controller, Model):
    """
    Template class for online-adaptive closed-loop controllers.
    """

    C_TYPE          = 'OA Controller'
    C_NAME          = '????'


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
        p_ctrl_error : ControlError
            Control error.
        p_state : State
            State of control system.
        p_setpoint : SetPoint
            Setpoint.
        """
        
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControllerFct (OAController):
    """
    Wrapper class for controllers based on an online-adaptive function mapping an error to an action.

    Parameters
    ----------
    p_fct : Function
        Function object mapping a control error to an action

    See class Controller for further parameters.
    """

    C_TYPE          = 'OA Controller Fct'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        try:
            self._fct.switch_adaptivity(p_ada)
        except:
            pass





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
class ControlPanel (Log):
    """
    Enables external control of a closed-loop control.
    """

    C_TYPE          = 'Control Panel'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def start(self):
        """
        (Re-)starts a closed-loop control.
        """
        
        self.log(Log.C_LOG_TYPE_S, 'Control process started')
        self._start()


## -------------------------------------------------------------------------------------------------
    def _start(self):
        """
        Custom method to (re-)start a closed-loop control.
        """

        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def stop(self):
        """
        Ends a closed-loop control.
        """

        self.log(Log.C_LOG_TYPE_S, 'Control process stopped')
        self._stop()


## -------------------------------------------------------------------------------------------------
    def _stop(self):
        """
        Custom method to end a closed-loop control.
        """

        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def change_setpoint( self, p_setpoint : SetPoint ):
        """
        Changes the setpoint values of a closed-loop control.

        Parameters
        ----------
        p_setpoint: SetPoint
            New setpoint values.
        """

        self.log(Log.C_LOG_TYPE_S, 'Setpoint values changed to', p_setpoint.values)
        self._change_setpoint( p_setpoint = p_setpoint )


## -------------------------------------------------------------------------------------------------
    def _change_setpoint( self, p_setpoint : SetPoint ):
        """
        Custom method to change setpoint values.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlShared (StreamShared, ControlPanel):
    
## -------------------------------------------------------------------------------------------------
    def _start(self):
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _stop(self):
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _change_setpoint(self, p_setpoint: SetPoint):
        
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
    def __init__( self, 
                  p_mode, 
                  p_cycle_limit=0, 
                  p_visualize:bool=False, 
                  p_logging=Log.C_LOG_ALL ):

        self._control_cycle : ControlCycle = None

        super.__init__( p_mode, 
                        p_cycle_limit=p_cycle_limit, 
                        p_auto_setup=True, 
                        p_visualize=p_visualize, 
                        p_logging=p_logging )
        

## -------------------------------------------------------------------------------------------------
    def get_control_panel(self) -> ControlPanel:
        """
        Returns
        -------
        panel : ControlPanel
            Object that enables the external control of a closed-loop control process.
        """
        return self._control_cycle.get_so()
