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
## -- 2024-09-11  0.3.0     DA       - class CTRLError renamed ControlError
## --                                - new class ControlPanel
## -- 2024-09-27  0.4.0     DA       Class ControlPanel: new parent EventManager
## -- 2024-10-04  0.5.0     DA       Updates on class Controller
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.5.0 (2024-10-04)

This module provides basic classes around the topic closed-loop control.

"""

from mlpro.bf.various import Log, TStampType
from mlpro.bf.mt import Task, Workflow
from mlpro.bf.events import Event, EventManager
from mlpro.bf.math import Element, Function
from mlpro.bf.streams import InstDict, InstType, InstTypeNew, Instance, StreamTask, StreamWorkflow, StreamShared, StreamScenario
from mlpro.bf.systems import ActionElement, Action, State, System
from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SetPoint (Instance):
    """
    Represents a new setpoint in a control loop.

    Parameters
    ----------
    p_setpoint_data : Element
        Container for new setpoint values.
    p_tstamp : TStampType 
        Time stamp.
    **p_kwargs
        Optional further keyword arguments.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_setpoint_data: Element, 
                  p_tstamp: TStampType, 
                  **p_kwargs ):
        
        super().__init__( p_feature_data = p_setpoint_data, 
                          p_label_data = None, 
                          p_tstamp = p_tstamp, 
                          **p_kwargs )
        

## -------------------------------------------------------------------------------------------------
    def _get_values(self):
        return self.get_feature_data().get_values()


## -------------------------------------------------------------------------------------------------
    def _set_values(self, p_values):
        self.get_feature_data().set_values( p_values = p_values)
        self._raise_event( p_event_id = self.C_EVENT_ID_SETPOINT_CHANGED, 
                           p_event_object = Event( p_raising_object=self) )


## -------------------------------------------------------------------------------------------------
    values = property( fget=_get_values, fset=_set_values )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlError (Instance):
    """
    Represents a control error in a control loop.

    Parameters
    ----------
    p_error_data : Element
        Container for new error values.
    p_tstamp : TStampType 
        Time stamp.
    **p_kwargs
        Optional further keyword arguments.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_error_data: Element, 
                  p_tstamp : TStampType, 
                  **p_kwargs ):
        
        super().__init__( p_feature_data = p_error_data, 
                          p_label_data = None, 
                          p_tstamp = p_tstamp, 
                          **p_kwargs )
        

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
        
        # 1 Get control error instance
        for inst_id, (inst_type, inst) in p_inst.items():
            if isinstance(inst, ControlError):
                ctrl_error = inst
                break

        # 2 Compute control action
        action = self.compute_action( p_ctrl_error = ctrl_error )

        # 3 Remove control error and add action
        del p_inst[ctrl_error.id]
        p_inst[action.id] = (InstTypeNew, action)


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_ctrl_error: ControlError) -> Action:
        """
        Custom method to compute and return an action based on an incoming control error.

        Parameters
        ----------
        p_ctrl_error : CTRLError
            Control error.


        Returns
        -------

        """

        # Create new action
        # get action id from so

        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _compute_action( self, 
                         p_ctrl_error : ControlError, 
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
    def compute_action(self, p_ctrl_error: ControlError) -> Action:

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

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_system : System,
                  p_name: str = None, 
                  p_range_max=Task.C_RANGE_THREAD, 
                  p_visualize: bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = False, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )

        self._system : System = p_system


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict ):

        # 0 Intro
        action : Action = None
        state  : State  = None

        # 1 Get action from instance dict
        for (inst_type, inst) in p_inst.values():
            if isinstance(p_inst,Action):
                action = p_inst
            elif isinstance(p_inst,State):
                state


        # 2 Hand over action to wrapped system

        # 3 Add system state t instance dict
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlPanel (EventManager):
    """
    Enables external control of a closed-loop control.
    """

    C_TYPE                  = 'Control Panel'
    C_NAME                  = '????'

    C_EVENT_ID_SETPOINT_CHG = 'Setpoint changed'


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
        self._raise_event( p_event_id = self.C_EVENT_ID_SETPOINT_CHANGED,
                           p_event_object = Event( p_raising_object = self ))


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
    def get_control_panel(self) -> ControlPanel:
        return self.get_so()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlScenario (StreamScenario):
    """
    ...
    """

    C_TYPE      = 'Control Scenario'

 ## -------------------------------------------------------------------------------------------------
    def setup(self):
        self._control_cycle = self._setup( p_mode=self.get_mode(), 
                                           p_visualize=self.get_visualization(),
                                           p_logging=self.get_log_level() )
    

 ## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging) -> ControlCycle:
        """
        Custom method to set up a control cycle. Create a new object of type ControlCycle and add
        all control tasks of your scenario.

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
        p_visualize : bool
            Boolean switch for visualisation.
        p_logging
            Log level (see constants of class Log). Default: Log.C_LOG_ALL. 

        Returns
        -------
        ControlCycle
            Object of type ControlCycle.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _set_mode(self, p_mode):
        self._control_cycle.set_mode


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        self._iterator = iter(self._stream)
        self._iterator.set_random_seed(p_seed=p_seed)


## -------------------------------------------------------------------------------------------------
    def get_control_panel(self) -> ControlPanel:
        """
        Returns
        -------
        panel : ControlPanel
            Object that enables the external control of a closed-loop control process.
        """

        return self._control_cycle.get_control_panel()