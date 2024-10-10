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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.8.0 (2024-10-08)

This module provides basic classes around the topic closed-loop control.

"""

from typing import Iterable
from matplotlib.figure import Figure
from mlpro.bf.plot import PlotSettings
from mlpro.bf.various import Log, TStampType
from mlpro.bf.mt import Figure, PlotSettings, Range, Task, Workflow
from mlpro.bf.ops import Mode
from mlpro.bf.events import Event, EventManager
from mlpro.bf.math import Element, Function, MSpace
from mlpro.bf.streams import InstDict, InstType, InstTypeNew, Instance, StreamTask, StreamWorkflow, StreamShared, StreamScenario
from mlpro.bf.systems import ActionElement, Action, State, System
from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SetPoint (Instance):
    """
    Setpoint.

    Parameters
    ----------
    p_setpoint_data : Element
        Container for new setpoint values.
    p_tstamp : TStampType 
        Time stamp.
    **p_kwargs
        Optional further keyword arguments.
    """

    C_NAME      = 'Setpoint' 

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
    Control error.

    Parameters
    ----------
    p_error_data : Element
        Container for new error values.
    p_tstamp : TStampType 
        Time stamp.
    **p_kwargs
        Optional further keyword arguments.
    """

    C_NAME      = 'Control Error' 

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
class ControlVariable (Action):
    """
    Output of a controller/input of a controlled system.
    """
    
    C_NAME      = 'Control Variable' 






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlledVariable (State):
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
    def _get_instance(self, p_inst: InstDict, p_type: type, p_remove: bool = False) -> Instance:
        """
        Gets and optionally removes an instance of a particular type from the p_inst dictionary.

        Parameters
        ----------
        p_inst: InstDict
            Dictionary of instances.
        p_type: type
            Type of instance to be found.
        p_remove: bool = False
            If true, the found instance is removed.
        """

        inst_found : Instance = None
        
        for (inst_type, inst) in p_inst.values():
            if isinstance( inst, p_type):
                inst_found = inst
                break
        
        if ( p_remove ) and ( inst_found is not None ):
            del p_inst[inst_found.id]

        return inst_found





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Operator (ControlTask):
    """
    Base class for all operators.
    """

    C_TYPE      = 'Operator'





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
                  p_range_max = Task.C_RANGE_NONE, 
                  p_visualize: bool = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        self._input_space : MSpace  = p_input_space
        self._output_space : MSpace = p_output_space
        
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
    def _run(self, p_inst: InstDict):
        
        # 1 Get control error instance
        ctrl_error = self._get_instance( p_inst = p_inst, p_type = ControlError, p_remove = True )
        if ctrl_error is None:
            self.log(Log.C_LOG_TYPE_E, 'Control error instance is missing!')
            return

        # 2 Compute and add control variable
        ctrl_var = self._get_instance( p_inst = p_inst, p_type = ControlVariable )
        ctrl_var = self.compute_output( p_ctrl_error = ctrl_error, p_ctrl_var = ctrl_var )
        p_inst[ctrl_var.id] = (InstTypeNew, ctrl_var)


## -------------------------------------------------------------------------------------------------
    def compute_output( self, 
                        p_ctrl_error: ControlError, 
                        p_ctrl_var: ControlVariable = None ) -> ControlVariable:
        """
        Computes a control variable based on an incoming control error.

        Parameters
        ----------
        p_ctrl_error : ControlError
            Control error object.
        p_ctrl_var : ControlVariable = None
            Optional ControlVariable object to be filled.

        Returns
        -------
        ControlVariable
            New control variable
        """

        # 1 Create new control variable if not provided from outside
        if p_ctrl_var is None:
            new_ctrl_var = ControlVariable( p_agent_id = self.id, 
                                            p_action_space = self._output_space, 
                                            p_tstamp = self.get_so().get_tstamp() )
        else:
            new_ctrl_var = p_ctrl_var
        
        new_ctrl_var.id = self.get_so().get_next_inst_id()


        # 2 Create new control variable element
        new_ctrl_var_elem = ActionElement( p_action_space = self._output_space )


        # 3 Call custom method to fill the new action element
        self._compute_output( p_ctrl_error = p_ctrl_error, 
                              p_ctrl_var_elem = new_ctrl_var_elem )
        

        # 4 Return the new control variable
        return new_ctrl_var


## -------------------------------------------------------------------------------------------------
    def _compute_output( self, 
                         p_ctrl_error : ControlError, 
                         p_ctrl_var_elem : ActionElement ):
        """
        Custom method to compute a control output based on an incoming control error. The result needs
        to be stored in the control variable element handed over.

        Parameters
        ----------
        p_ctrl_error : CTRLError
            Control error.
        p_ctrl_var_elem : ActionElement
            Control variable element to be filled with resulting value(s).
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
                  p_range_max = Task.C_RANGE_NONE, 
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
    def _compute_output(self, p_ctrl_error: ControlError, p_ctrl_var_elem: ActionElement):

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
class ControlledSystem (ControlTask):
    """
    Wrapper class for state-based systems.
    """

    C_TYPE          = 'Controlled System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_system : System,
                  p_name: str = None, 
                  p_range_max=Task.C_RANGE_NONE, 
                  p_visualize: bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_duplicate_data = False, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )

        self.system : System = p_system


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict ):

        # 1 Get and remove control variable and controlled variable from instance dict
        ctrl_var     = self._get_instance( p_inst = p_inst, p_type = ControlVariable, p_remove = True )
        ctrlled_var  = self._get_instance( p_inst = p_inst, p_type = ControlledVariable, p_remove = True )


        # 2 Let the wrapped system process the action
        if self.system.process_action( p_action = ctrl_var ):
            ctrlled_var = self.system.get_state()
            ctrlled_var.id = self.get_so().get_next_inst_id()
            ctrlled_var.tstamp = self.get_so().get_tstamp()
            p_inst[ctrlled_var.id] = ( InstTypeNew, ctrlled_var)
        else:
            self.log(Log.C_LOG_TYPE_E, 'Processing of control variable failed!')





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
        self._next_inst_id = 0


## -------------------------------------------------------------------------------------------------
    def init(self, p_ctrlled_var_space: MSpace, p_ctrl_var_space: MSpace):
        """
        Initializes the shared object with contextual information.

        Parameters
        ----------
        p_ctrlled_var_space : MSpace
            Controlled variable space.
        p_ctrl_var_space : MSpace
            Control variable space.
        """

        self._ctrlled_var_space = p_ctrlled_var_space
        self._ctrl_var_space    = p_ctrl_var_space


## -------------------------------------------------------------------------------------------------
    def reset(self, p_inst: InstDict):
        pass


## -------------------------------------------------------------------------------------------------
    def get_next_inst_id(self) -> int:
        """
        Returns the next instance id.

        Returns
        -------
        int
            Next instance id.
        """

        self.lock( p_tid = self.C_TID_ADMIN )
        next_id = self._next_inst_id
        self._next_inst_id += 1
        self.unlock()
        return next_id
    

## -------------------------------------------------------------------------------------------------
    def get_tstamp(self) -> TStampType:
        
        return None
        #raise NotImplementedError
    
    
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


        # 2 Get or create a setpoint instance
        for (inst_type, inst) in inst_admin.values():
            if isinstance(inst, SetPoint):
                setpoint = inst
                break

        if setpoint is None:
            setpoint_data = Element( p_set = self._ctrlled_var_space )
            setpoint_data.set_values( p_values = p_values )
            setpoint = SetPoint( p_setpoint_data = setpoint_data, 
                                 p_tstamp = self.get_tstamp() )
            setpoint.id = self.get_next_inst_id()
            inst_admin[setpoint.id] = (InstTypeNew, setpoint)
        else:
            setpoint.values = p_values
            setpoint.tstamp = self.get_tstamp()


        # 3 Outro
        self.unlock()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlWorkflow (StreamWorkflow, Mode):
    """
    Container class for all tasks of a control cycle.
    """

    C_TYPE          = 'Control Workflow'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode,
                  p_name: str = None, 
                  p_range_max = Task.C_RANGE_NONE, 
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


## -------------------------------------------------------------------------------------------------
    def get_control_panel(self) -> ControlPanel:
        return self.get_so()
    

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
    def add_task(self, p_task: Task, p_pred_tasks: list = None):
        StreamWorkflow.add_task( self, p_task = p_task, p_pred_tasks = p_pred_tasks)
        
        try:
            p_task.set_mode( p_mode = self._mode )
        except:
            pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlSystem (StreamScenario):
    """
    ...
    """

    C_TYPE      = 'Control System'

## -------------------------------------------------------------------------------------------------
    def setup(self):

        # 1 Setup control workflow
        self._control_workflow = self._setup( p_mode=self.get_mode(), 
                                           p_visualize=self.get_visualization(),
                                           p_logging=self.get_log_level() )
          

 ## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging) -> ControlWorkflow:
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
    def get_control_panel(self) -> ControlPanel:
        """
        Returns
        -------
        panel : ControlPanel
            Object that enables the external control of a closed-loop control process.
        """

        return self._control_workflow.get_control_panel()
    

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
        