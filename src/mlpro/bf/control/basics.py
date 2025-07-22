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
## -- 2024-10-06  0.6.0     DA       New classes ControlTask, Operator
## -- 2024-10-07  0.7.0     DA       - new method ControlShared.get_tstamp()
## --                                - refactoring of class Controller
## -- 2024-10-08  0.8.0     DA       Classes ControlPanel, ControlShared: refactoring
## -- 2024-10-10  0.9.0     DA       - class Controller: bugfix in method compute_output()
## --                                - class ControlWorkflow: method run() redefined
## -- 2024-10-13  0.10.0    DA       Refactoring: changed parent of class Action to Instance
## -- 2024-10-24  0.11.0    DA       Class ControlledSystem: redefinition of method init_plot(),
## --                                update_plot(), remove_plot()
## -- 2024-11-08  0.12.0    DA       Little refactoring of class Operator
## -- 2024-11-09  0.13.0    DA       Various changes and improvements
## -- 2024-11-10  0.14.0    DA       - class ControlWorkflow: master plot disabled
## --                                - new helper functions get_ctrl_data(), replace_ctrl_data()
## -- 2024-11-11  0.15.0    DA       Implementation of custom method ControlWorkflow._on_event()
## -- 2024-11-14  0.16.0    DA       Introduction of time management
## -- 2024-11-15  0.17.0    DA       Various corrections
## -- 2024-11-26  1.0.0     DA       Classes Controller, ControlledSystem: initial idle loop to
## --                                determine the initial system state
## -- 2025-06-02  1.0.1     DA       Implemented method ControlledSystem._reset()
## -- 2025-06-11  1.1.0     DA       Refactoring
## -- 2025-07-18  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-07-18)

This module provides basic classes around the topic closed-loop control.

"""

from typing import Iterable, Tuple, List
from datetime import datetime, timedelta

from mlpro.bf.plot import Figure, PlotSettings
from mlpro.bf.various import Log, TStampType, Timer
from mlpro.bf.mt import Range, Task, Workflow, Shared
from mlpro.bf.ops import Mode
from mlpro.bf.events import Event, EventManager
from mlpro.bf.math import Element, Function, MSpace
from mlpro.bf.streams import InstDict, InstTypeNew, Instance, StreamTask, StreamWorkflow, StreamShared, StreamScenario
from mlpro.bf.systems import Action, System



# Export list for public API
__all__ = [ 'ControlData',
            'get_ctrl_data',
            'replace_ctrl_data',
            'SetPoint',
            'ControlError',
            'ControlVariable',
            'ControlledVariable',
            'ControlTask',
            'Operator',
            'Controller',
            'ControllerFct',
            'ControlledSystem',
            'ControlPanel',
            'ControlShared',
            'ControlPanelEntry',
            'ControlWorkflow',
            'ControlSystem' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlData (Instance):
    """
    Root class for all types of control data.

    Parameters
    ----------
    p_id : int,
        Instance id.
    p_value_space : MSpace
        Metric space of the values.
    p_values : Iterable
        Values.
    p_tstamp : TStampType = None
        Optional time stamp.
    **p_kwargs
        Optional further keyword arguments.
    """

    C_TYPE          = 'Control Data'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id : int,
                  p_value_space : MSpace,
                  p_values : Iterable = None, 
                  p_tstamp: TStampType = None, 
                  **p_kwargs ):
        
        feature_element = Element( p_set = p_value_space )
        feature_element.set_values( p_values = p_values )

        super().__init__( p_feature_data = feature_element, 
                          p_label_data = None, 
                          p_tstamp = p_tstamp, 
                          **p_kwargs )
        
        self.set_id( p_id = p_id )
        
        
## -------------------------------------------------------------------------------------------------
    def _get_value_space(self):
        return self.get_feature_data().get_related_set()
    

## -------------------------------------------------------------------------------------------------
    def _get_values(self):
        return self.get_feature_data().get_values()


## -------------------------------------------------------------------------------------------------
    def _set_values(self, p_values):
        self.get_feature_data().set_values( p_values = p_values)


## -------------------------------------------------------------------------------------------------
    def copy(self):
        duplicate = self.__class__( p_id = self.get_id(),
                                    p_value_space = self.value_space,
                                    p_values = self.values,
                                    p_tstamp=self.get_tstamp(),
                                    p_kwargs=self._get_kwargs() )
        return duplicate


## -------------------------------------------------------------------------------------------------
    value_space = property( fget=_get_value_space )
    values = property( fget=_get_values, fset=_set_values )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
def get_ctrl_data( p_instances: InstDict, p_type: type, p_remove: bool = False) -> ControlData:
    """
    Gets and optionally removes a control data instance of a particular type from the p_instances dictionary.

    Parameters
    ----------
    p_instances : InstDict
        Dictionary of instances.
    p_type: type
        Type of instance to be found.
    p_remove: bool = False
        If true, the found instance is removed.
    """

    ctrl_data_found : ControlData = None
        
    for (inst_type, inst) in p_instances.values():
        if isinstance( inst, p_type):
            ctrl_data_found = inst
            break
        
    if ( p_remove ) and ( ctrl_data_found is not None ):
        del p_instances[ctrl_data_found.id]

    return ctrl_data_found





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
def replace_ctrl_data( p_instances: InstDict, p_ctrl_data: ControlData ):
    """
    Adds the specified control data instance to the p_instances dictionary and removes an already existing
    control data instance of the same type before that.

    Parameters
    ----------
    p_instances : InstDict
        Dictionary of instances.
    p_ctrl_data : ControlData
        Control data instance to be added.
        
    Returns
    -------
    bool
        True, if control data of the same type were actually removed. False otherwise.
    """

    result = ( get_ctrl_data( p_instances = p_instances, p_type = type(p_ctrl_data), p_remove =  True ) != None )
    p_instances[p_ctrl_data.id] = (InstTypeNew, p_ctrl_data)

    return result





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SetPoint (ControlData):
    """
    Setpoint.
    """

    C_NAME      = 'Setpoint' 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlError (ControlData):
    """
    Control error.
    """

    C_NAME      = 'Control Error' 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlVariable (ControlData):
    """
    Output of a controller/input of a controlled system.
    """
    
    C_NAME      = 'Control Variable' 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlledVariable (ControlData):
    """
    Output of a controlled system.
    """
    
    C_NAME      = 'Controlled Variable' 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlTask (StreamTask):
    """
    Base class for all control tasks.
    """

    C_TYPE      = 'Control Task'

## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed = None, **p_kwargs):
        self._reset( p_seed = p_seed, **p_kwargs )
    

## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed = None, **p_kwargs):
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Operator (ControlTask):
    """
    Base class for all operators.

    Parameters
    ----------
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    """

    C_TYPE      = 'Operator'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_range_max=Range.C_RANGE_NONE, 
                  p_visualize = False, 
                  p_logging=Log.C_LOG_ALL ):
        
        super().__init__( p_name = self.C_NAME, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = False, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Controller (ControlTask):
    """
    Template class for closed-loop controllers.

    Parameters
    ----------
    p_input_space : MSpace
        Input (or error) space of the controller.
    p_output_space : MSpace 
        Output (or action) space of the controller.
    p_id = None
        Unique id of the controller
    ...
    """

    C_TYPE          = 'Controller'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_input_space : MSpace,
                  p_output_space : MSpace,
                  p_id = None,
                  p_name: str = None, 
                  p_range_max = Range.C_RANGE_NONE, 
                  p_visualize: bool = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        self._input_space : MSpace     = p_input_space
        self._output_space : MSpace    = p_output_space
        self._last_update : TStampType = None
        self._current_ctrl_var : ControlVariable = None
        self._computation_time : timedelta = None
        
        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = False, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )
        
        if p_id is not None:
            self.id = p_id
        

## -------------------------------------------------------------------------------------------------
    def set_parameter(self, **p_param):
        """
        Custom method to set/change the parameters of a specific controller implementation.

        Parameters
        ----------
        **p_param
            Parameters of the controller.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict):

        # 0 Intro
        so : ControlShared = self.get_so()
        

        # 1 Get control error instance
        ctrl_error = get_ctrl_data( p_instances = p_instances, p_type = ControlError, p_remove = True )
        if ctrl_error is None:
            self.log(Log.C_LOG_TYPE_W, 'Control error instance is missing!')
            return


        # 2 Remove existing control variable from inst dictionary
        get_ctrl_data( p_instances = p_instances, p_type = ControlVariable, p_remove = True )


        # 3 Compute control output
        try:
            compute = ( so.timer.get_time() - self._last_update ) >= ( so.latency - self._computation_time )
        except:
            compute = True

        if compute or ( self._current_ctrl_var is None ):
            self.log(Log.C_LOG_TYPE_I, 'Computation started')
            tstamp_before = so.timer.get_time()
            self._current_ctrl_var = self.compute_output( p_ctrl_error = ctrl_error )
            tstamp_after = so.timer.get_time()
            tdelta = tstamp_after - tstamp_before
            
            # Determine the smallest computation time > 0
            if ( self._computation_time is None ) or ( tdelta < self._computation_time ):
                self._computation_time = tdelta

            self._last_update = tstamp_after
            self.log(Log.C_LOG_TYPE_I, 'Computation finished')
        else:
            self._current_ctrl_var = self._current_ctrl_var.copy()
            self._current_ctrl_var.id = so.get_next_inst_id()
            self._current_ctrl_var.tstamp = so.get_tstamp()
            self.log(Log.C_LOG_TYPE_I, 'Computation skipped. Last action duplicated.')


        # 4 Complete and store new control variable
        p_instances[self._current_ctrl_var.id] = (InstTypeNew, self._current_ctrl_var)


## -------------------------------------------------------------------------------------------------
    def compute_output( self, p_ctrl_error: ControlError ) -> ControlVariable:
        """
        Computes a control variable based on an incoming control error. It creates a new control
        variable and call the custom method _compute_output() to fill it with values.

        Parameters
        ----------
        p_ctrl_error : ControlError
            Control error.

        Returns
        -------
        ControlVariable
            ControlVariable.
        """

        # 1 Create new control variable 
        so = self.get_so()
        ctrl_var    = ControlVariable( p_id = so.get_next_inst_id(),
                                       p_value_space = self._output_space )

        # 2 Call custom method to fill the new action element
        tstamp_before = datetime.now()
        self._compute_output( p_ctrl_error = p_ctrl_error, p_ctrl_var = ctrl_var )
        tstamp_after  = datetime.now()
        tdelta        = tstamp_after - tstamp_before
        so.timer.add_time( p_delta = tdelta )


        # 3 Complete and return the new control variable
        ctrl_var.tstamp = so.get_tstamp()
        return ctrl_var


## -------------------------------------------------------------------------------------------------
    def _compute_output( self, p_ctrl_error : ControlError, p_ctrl_var : ControlVariable ):
        """
        Custom method to compute a control output based on an incoming control error. The result needs
        to be stored in the control variable element handed over.

        Parameters
        ----------
        p_ctrl_error : CTRLError
            Control error.
        p_ctrl_var : ControlVariable
            Control variable to be filled with resulting value(s).
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
                  p_fct: Function,
                  p_name: str = None, 
                  p_range_max = Range.C_RANGE_NONE, 
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
    def _compute_outsput(self, p_ctrl_error: ControlError, p_ctrl_var: ControlVariable):

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlledSystem (ControlTask):
    """
    Wrapper class for state-based systems.
    """

    C_TYPE          = 'Controlled System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_system : System,
                  p_name: str = None, 
                  p_range_max = Range.C_RANGE_NONE, 
                  p_visualize: bool = False, 
                  p_logging = Log.C_LOG_ALL ):
        
        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = False, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging )

        self.system : System = p_system
        self._last_update : TStampType = None
        self._current_action : Action  = None


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None, **p_kwargs):
        return self.system.reset(p_seed, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict ):

        # 0 Intro
        so : ControlShared = self.get_so()


        # 1 Remove an already existing controlled variable
        get_ctrl_data( p_instances = p_instances, p_type = ControlledVariable, p_remove = True )


        # 2 Get and remove control variable
        ctrl_var = get_ctrl_data( p_instances = p_instances, p_type = ControlVariable, p_remove = True )

        if ctrl_var is not None:
            # 3 Process control variable

            # 3.1 Update the current action instance for the wrapped system after the latency time period
            if ( self._last_update is None ) or ( ( so.timer.get_time() - self._last_update ) >= so.latency ):
                self._current_action = Action( p_agent_id = 0,
                                            p_action_space = ctrl_var.get_feature_data().get_related_set(),
                                            p_values = ctrl_var.values,
                                            p_tstamp = ctrl_var.tstamp )
                self._last_update    = so.timer.get_time()
                self.log(Log.C_LOG_TYPE_I, 'Action updated')
                

            # 3.2 Let the wrapped system process the action
            if self.system.process_action( p_action = self._current_action, p_t_step = so.latency_min ):
                so.timer.add_time( p_delta = so.latency_min )

            else:
                self.log(Log.C_LOG_TYPE_E, 'Processing of control variable failed!')

        else:
            self.log(Log.C_LOG_TYPE_W, 'Control variable missing!')


        # 4 Determine the current system state
        state                  = self.system.get_state()
        ctrlled_var            = ControlledVariable( p_id = so.get_next_inst_id(),
                                                     p_value_space = self.system.get_state_space(),
                                                     p_values = state.values,
                                                     p_tstamp = so.get_tstamp() )
        p_instances[ctrlled_var.id] = ( InstTypeNew, ctrlled_var)
        self.log(Log.C_LOG_TYPE_S, 'Controlled variable created')
        

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure = None, p_plot_settings = None):
        super().init_plot(p_figure = p_figure, p_plot_settings = p_plot_settings)
        self.system.init_plot(p_figure = p_figure, p_plot_settings = p_plot_settings)


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_instances : InstDict = None, **p_kwargs):
        super().update_plot(p_instances = p_instances, **p_kwargs)
        self.system.update_plot( **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh : bool = True):
        super().remove_plot(p_refresh = p_refresh)
        self.system.remove_plot(p_refresh = p_refresh)




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlPanel (EventManager):
    """
    Enables external control of a closed-loop control.
    """

    C_TYPE                  = 'Control Panel'
    C_NAME                  = '????'

    C_EVENT_ID_SETPOINT_CHG = 'SETPOINT_CHG'


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
    def set_setpoint( self, p_values: Iterable ):
        """
        Changes the setpoint values of a closed-loop control.

        Parameters
        ----------
        p_values : Iterable
            New setpoint values.
        """

        self.log(Log.C_LOG_TYPE_S, 'Setpoint values changed to', p_values)
        self._set_setpoint( p_values = p_values )
        self._raise_event( p_event_id = self.C_EVENT_ID_SETPOINT_CHG,
                           p_event_object = Event( p_raising_object = self ) )


## -------------------------------------------------------------------------------------------------
    def _set_setpoint( self, p_values: Iterable ):
        """
        Custom method to change setpoint values.

        Parameters
        ----------
        p_values : Iterable
            New setpoint values.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlShared (StreamShared, ControlPanel, Log):
    """
    ...
    """

    C_TID_ADMIN     = 'wf'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_range: int = Range.C_RANGE_PROCESS):

        StreamShared.__init__(self, p_range=p_range)
        Log.__init__(self, p_logging = Log.C_LOG_NOTHING)

        self._next_inst_id                = 0
        self._superior_so : ControlShared = None
        self._top_so : ControlShared      = self
        self._ctrlled_var_space : MSpace  = None
        self._ctrl_var_space : MSpace     = None
        self._timer : Timer               = None
        self._latency : timedelta         = None
        self._latency_min : timedelta     = None


## -------------------------------------------------------------------------------------------------
    def init( self, 
              p_ctrlled_var_space: MSpace, 
              p_ctrl_var_space: MSpace, 
              p_mode: int,
              p_latency: timedelta ):
        """
        Initializes the shared object with contextual information.

        Parameters
        ----------
        p_ctrlled_var_space : MSpace
            Controlled variable space.
        p_ctrl_var_space : MSpace
            Control variable space.
        p_mode : int
            Operation mode (0 = Simulation, 1 = Real operation)
        p_latency : timedelta
             controlled system
        """

        self._ctrlled_var_space = p_ctrlled_var_space
        self._ctrl_var_space    = p_ctrl_var_space

        mode_timer              = 1 - p_mode
        self._timer             = Timer( p_mode = mode_timer )

        if ( self.latency is None ) or ( p_latency < self.latency ):
            self.latency = p_latency     


## -------------------------------------------------------------------------------------------------
    def reset(self, p_instances : InstDict):
        setpoint = get_ctrl_data( p_instances = p_instances, p_type = SetPoint, p_remove = False)
        if setpoint != None:
            replace_ctrl_data( p_instances = self._instances[self.C_TID_ADMIN], p_ctrl_data = setpoint )


## -------------------------------------------------------------------------------------------------
    def get_superior_so(self) -> Shared:
        return self._superior_so
    

## -------------------------------------------------------------------------------------------------
    def set_superior_so(self, p_so : Shared ):
        """
        Sets the superior shared object. This is relevant for cascade control systems, where the
        top level shared object is responsible for system-wide unique instance ids etc.

        Parameters
        ----------
        p_so : Shared
            Superior shared object.
        """

        if ( p_so.latency_min is None ) or ( self.latency_min is None ) or ( self.latency_min < p_so.latency_min ):
            p_so.latency_min = self.latency_min

        self._superior_so = p_so
        self._top_so      = p_so.top_so


## -------------------------------------------------------------------------------------------------
    def get_top_so(self) -> Shared:
        return self._top_so
    

## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        return self._latency


## -------------------------------------------------------------------------------------------------
    def set_latency(self, p_latency : timedelta):
        self._latency = p_latency
        if ( self.latency_min is None ) or ( p_latency < self.latency_min ): 
            self.latency_min = p_latency


## -------------------------------------------------------------------------------------------------
    def get_latency_min(self) -> timedelta:
        if self == self._top_so: 
            return self._latency_min
        else:
            return self._top_so.latency_min


## -------------------------------------------------------------------------------------------------
    def set_latency_min(self, p_latency : timedelta):
        if self == self.top_so: 
            self._latency_min = p_latency
        else: 
            self.top_so.latency_min = p_latency
        

## -------------------------------------------------------------------------------------------------
    def get_next_inst_id(self) -> int:
        """
        Returns the next instance id.

        Returns
        -------
        int
            Next instance id.
        """

        if self.top_so == self:
            self.lock( p_tid = self.C_TID_ADMIN )
            next_id = self._next_inst_id
            self._next_inst_id += 1
            self.unlock()
            return next_id
        
        else:
            return self.top_so.get_next_inst_id()
    

## -------------------------------------------------------------------------------------------------
    def get_tstamp(self) -> TStampType:
        """
        Returns the current process time stamp.
        """

        if self.top_so == self:
            return self._timer.get_time()
        
        else:
            return self.top_so.get_tstamp()
        

## -------------------------------------------------------------------------------------------------
    def get_timer(self) -> Timer:
        if self.top_so == self:
            return self._timer
        else:
            return self.top_so.timer
        
    
## -------------------------------------------------------------------------------------------------
    def _start(self):
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _stop(self):
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _set_setpoint(self, p_values: Iterable):
        
        # 0 Intro
        setpoint : SetPoint = None
        self.lock( p_tid = self.C_TID_ADMIN )


        # 1 Locate the instance dictionary for the first control task
        try:
            inst_admin = self._instances[self.C_TID_ADMIN]
        except:
            inst_admin = {}
            self._instances[self.C_TID_ADMIN] = inst_admin


        # 2 Replace setpoint instance
        get_ctrl_data( p_instances = inst_admin, p_type = SetPoint, p_remove = True)

        setpoint = SetPoint( p_id = self.get_next_inst_id(),
                             p_value_space = self._ctrlled_var_space,
                             p_values = p_values,
                             p_tstamp = self.get_tstamp() )
        
        inst_admin[setpoint.id] = (InstTypeNew, setpoint)


        # 3 Outro
        self.unlock()


## -------------------------------------------------------------------------------------------------
    superior_so = property( fget = get_superior_so, fset = set_superior_so )
    top_so      = property( fget = get_top_so )
    latency     = property( fget = get_latency, fset = set_latency )
    latency_min = property( fget = get_latency_min, fset = set_latency_min )
    timer       = property( fget = get_timer )




ControlPanelEntry = Tuple[ControlPanel, Workflow]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlWorkflow (StreamWorkflow, Mode):
    """
    Container class for all tasks of a control cycle.
    """

    C_TYPE          = 'Control Workflow'
    C_NAME          = ''

    C_PLOT_ACTIVE   = False         # Currently no master plot window here...

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode,
                  p_name: str = None, 
                  p_range_max = Range.C_RANGE_NONE, 
                  p_class_shared = ControlShared, 
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        StreamWorkflow.__init__( self, 
                                 p_name = p_name, 
                                 p_range_max = p_range_max, 
                                 p_class_shared = p_class_shared, 
                                 p_visualize = p_visualize,
                                 p_logging = p_logging, 
                                 **p_kwargs )
        
        Mode.__init__( self, 
                       p_mode = p_mode,
                       p_logging = p_logging )
        
        self.get_so().switch_logging( p_logging = p_logging )

        self._workflows = []


## -------------------------------------------------------------------------------------------------
    def get_control_panels(self) -> List[ControlPanelEntry]:
        result = [ (self.get_so(), self) ]

        for workflow in self._workflows:
            result.extend( workflow.get_control_panels() )

        return result
    

## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):

        if p_mode == self._mode: return

        Mode.set_mode( self, p_mode = p_mode )

        for task in self.tasks:
            try:
                task.set_mode( p_mode = p_mode )
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def assign_so(self, p_so):
        """
        External assignment of shared objects is disabled for control workflows.
        """
        
        self.get_so().superior_so = p_so


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task: Task, p_pred_tasks: list = None):

        if not isinstance( p_task, Workflow ):
            p_task.set_name( 'Workflow ' + self.get_name() + ', ' + p_task.get_name() )

        if isinstance( p_task, ControlWorkflow ):
            self._workflows.append(p_task)
        
        StreamWorkflow.add_task( self, p_task = p_task, p_pred_tasks = p_pred_tasks)
        
        try:
            p_task.set_mode( p_mode = self._mode )
        except:
            pass

        if isinstance( p_task, ControlledSystem ):
            self.get_so().init( p_ctrlled_var_space = p_task.system.get_state_space(),
                                p_ctrl_var_space = p_task.system.get_action_space(),
                                p_mode = self._mode,
                                p_latency = p_task.system.get_latency() )
            

## -------------------------------------------------------------------------------------------------
    def run( self, 
             p_range : int = None, 
             p_wait: bool = False, 
             p_instances : InstDict = None ):
        
        # 1 Transfer the setpoint instances from the predecessor tasks of a superior workflow
        try:
            inst_dict = p_instances.copy()
        except:
            superior_setpoint : SetPoint = None
            inst_dict = None

            superior_so = self.get_so().superior_so

            if superior_so is not None:
                superior_so.lock( p_tid = self.get_tid() )

                for pred_task in self.get_predecessors():
                    superior_setpoint = get_ctrl_data( p_instances = superior_so._instances[pred_task.get_tid()], 
                                                       p_type = SetPoint,
                                                       p_remove = True )
                
                superior_so.unlock()

                if superior_setpoint is not None:
                    self.get_so().set_setpoint( p_values = superior_setpoint.values )

        
        # 2 Execute all tasks
        StreamWorkflow.run( self, p_range = p_range, p_wait = p_wait, p_instances = inst_dict)


## -------------------------------------------------------------------------------------------------
    def _on_finished(self):

        # 1 Add/replace the outcomes of the final task to the instance dict of the initial task
        so = self.get_so()
        so.lock( p_tid = ControlShared.C_TID_ADMIN )
        setpoint = get_ctrl_data( p_instances = so._instances[ControlShared.C_TID_ADMIN], p_type = SetPoint, p_remove = False )

        del so._instances[ControlShared.C_TID_ADMIN]
        new_setpoint = setpoint.copy()
        new_setpoint.id = so.get_next_inst_id()
        new_setpoint.tstamp = so.get_tstamp()
        so._instances[ControlShared.C_TID_ADMIN] = { new_setpoint.id : (InstTypeNew, new_setpoint) }

        for task in self._final_tasks:
            so._instances[ControlShared.C_TID_ADMIN].update(so._instances[task.id])

        so.unlock()

        # 2 Add the outcomes of the final task to the instance dict of a superior shared object
        superior_so = so.superior_so

        if superior_so is not None:
            superior_so.lock( p_tid = self.get_tid() )

            superior_so._instances[self.get_tid()] = {}

            for task in self._final_tasks:
                superior_so._instances[self.get_tid()].update( so._instances[task.id] )

            superior_so.unlock()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlSystem (StreamScenario):
    """
    Template class for custom control systems. Please implement
    """

    C_TYPE      = 'Control System'

## -------------------------------------------------------------------------------------------------
    def setup(self, **p_kwargs):
        self._control_workflow = self._setup( p_mode = self.get_mode(), 
                                              p_visualize = self.get_visualization(),
                                              p_logging = self.get_log_level(),
                                              **p_kwargs )
          

 ## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging, **p_kwargs) -> ControlWorkflow:
        """
        Custom method to set up a control workflow. Create a new object of type ControlWorkflow and
        add all control tasks of your scenario.

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
        p_visualize : bool
            Boolean switch for visualisation.
        p_logging
            Log level (see constants of class Log). Default: Log.C_LOG_ALL. 
        p_kwargs : dict
            Custom keyword parameters.

        Returns
        -------
        ControlCycle
            Object of type ControlWorkflow.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _set_mode(self, p_mode):
        self._control_workflow.set_mode( p_mode = p_mode)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        self._control_workflow.reset( p_seed = p_seed)


## -------------------------------------------------------------------------------------------------
    def get_control_panels(self) -> List[ControlPanelEntry]:
        """
        Returns
        -------
        panel : List[ControlPanelEntry]
            Object that enables the external control of a closed-loop control process.
        """

        return self._control_workflow.get_control_panels()
    

## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):
        
        error = False

        try:
            self._control_workflow.run()
        except KeyError as Argument:
            error = True
            if Argument.args[0] == ControlShared.C_TID_ADMIN:
                self.log(Log.C_LOG_TYPE_E, 'Setpoint missing')
            else:
                self.log(Log.C_LOG_TYPE_E, 'Control instance missing for task', Argument.args[0])

        return False, error, False, False
    

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):
        self._control_workflow.init_plot( p_figure = p_figure,
                                          p_plot_settings = p_plot_settings )
        