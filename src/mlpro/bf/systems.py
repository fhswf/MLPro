## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : systems.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-11-29  1.0.0     DA       Creation 
## -- 2022-11-30  1.0.1     DA       Class System: corrections and deviating default implementations
## --                                for custom methods _compute_success(), _compute_broken()
## -- 2022-12-05  1.1.0     DA       New classes SystemBase, Sensor, Actuator, Controller
## -- 2022-12-09  1.2.0     DA       - Class Controller: new methods get_sensors(), get_actuators()
## --                                - Class System, method add_controller(): all components can now
## --                                  be addressed by their names
## -- 2023-01-14  1.3.0     SY/ML    New class TransferFunction
## -- 2023-01-15  1.3.1     SY       New class UnitConverter
## -- 2023-01-16  1.3.2     SY       Shift UnitConverter to bf.math
## -- 2023-01-18  1.3.3     SY       Debugging on TransferFunction
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.3.3 (2023-01-18)

This module provides models and templates for state based systems.
"""


from time import sleep
from mlpro.bf.various import TStamp, ScientificObject, Label
from mlpro.bf.data import *
from mlpro.bf.plot import Plottable
from mlpro.bf.ops import Mode
from mlpro.bf.math import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class State(Element, TStamp):
    """
    State of a system as an element of a given state space. Additionally, the state can be
    labeled with various properties.

    Parameters
    ----------
    p_state_space : MSpace
        State space of the related system.
    p_initial : bool
        This optional flag signals that the state is the first one after a reset. Default=False.
    p_terminal : bool
        This optional flag labels the state as a terminal state. Default=False.
    p_success : bool
        This optional flag labels the state as an objective state. Default=False.
    p_broken : bool
        This optional flag labels the state as a final error state. Default=False.
    p_timeout : bool
        This optional flag signals that the cycle limit of an episode has been reached. Default=False.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_state_space: MSpace,
                 p_initial: bool = False,
                 p_terminal: bool = False,
                 p_success: bool = False,
                 p_broken: bool = False,
                 p_timeout: bool = False):

        TStamp.__init__(self)
        Element.__init__(self, p_state_space)
        self.set_initial(p_initial)
        self.set_terminal(p_terminal)
        self.set_success(p_success)
        self.set_broken(p_broken)
        self.set_timeout(p_timeout)


## -------------------------------------------------------------------------------------------------
    def get_initial(self) -> bool:
        return self._initial


## -------------------------------------------------------------------------------------------------
    def set_initial(self, p_initial: bool):
        self._initial = p_initial


## -------------------------------------------------------------------------------------------------
    def get_success(self) -> bool:
        return self._success


## -------------------------------------------------------------------------------------------------
    def set_success(self, p_success: bool):
        self._success = p_success


## -------------------------------------------------------------------------------------------------
    def get_broken(self) -> bool:
        return self._broken


## -------------------------------------------------------------------------------------------------
    def set_broken(self, p_broken: bool):
        self._broken = p_broken
        if p_broken:
            self.set_terminal(True)


## -------------------------------------------------------------------------------------------------
    def get_timeout(self) -> bool:
        return self._timeout


## -------------------------------------------------------------------------------------------------
    def set_timeout(self, p_timeout: bool):
        self._timeout = p_timeout
        if p_timeout:
            self.set_terminal(True)


## -------------------------------------------------------------------------------------------------
    def get_terminal(self) -> bool:
        return self._terminal


## -------------------------------------------------------------------------------------------------
    def set_terminal(self, p_terminal: bool):
        self._terminal = p_terminal





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ActionElement (Element):
    """
    Single entry of an action. See class Action for further details.

    Parameters
    ----------
    p_action_space : Set
        Related action space.
    p_weight : float
        Weight of action element. Default = 1.0.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_action_space : Set, 
                  p_weight : float = 1.0):

        super().__init__(p_action_space)
        self.set_weight(p_weight)


## -------------------------------------------------------------------------------------------------
    def get_weight(self):
        return self.weight


## -------------------------------------------------------------------------------------------------
    def set_weight(self, p_weight):
        self.weight = p_weight





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Action(ElementList, TStamp):
    """
    Objects of this class represent actions of (multi-)agents. Every element
    of the internal list is related to an agent, and its partial subsection.
    Action values for the first agent can be added while object instantiation.
    Action values of further agents can be added by using method self.add_elem().

    Parameters
    ----------
    p_agent_id        
        Unique id of (first) agent to be added
    p_action_space : Set   
        Action space of (first) agent to be added
    p_values : np.ndarray          
        Action values of (first) agent to be added
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_agent_id = 0, 
                  p_action_space : Set = None, 
                  p_values: np.ndarray = None ):

        ElementList.__init__(self)
        TStamp.__init__(self)

        if (p_action_space is not None) and (p_values is not None):
            e = ActionElement(p_action_space)
            e.set_values(p_values)
            self.add_elem(p_agent_id, e)


## -------------------------------------------------------------------------------------------------
    def get_agent_ids(self):
        return self.get_elem_ids()


## -------------------------------------------------------------------------------------------------
    def get_sorted_values(self) -> np.ndarray:
        # 1 Determine overall dimensionality of action vector
        num_dim = 0
        action_ids = []

        for elem in self._elem_list:
            num_dim = num_dim + elem.get_related_set().get_num_dim()
            action_ids.extend(elem.get_related_set().get_dim_ids())

        # 2 Transfer action values
        action = np.zeros(num_dim)

        for elem in self._elem_list:
            for elem_action_id in elem.get_related_set().get_dim_ids():
                i = action_ids.index(elem_action_id)
                action[i] = elem.get_value(elem_action_id)

        # 3 Return sorted result array
        return action





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FctSTrans (Log):
    """
    Template class for state transition functions.

    Parameters
    ----------
    p_logging 
        Log level (see class Log for more details). Default = Log.C_LOG_ALL.
    """

    C_TYPE = 'Fct STrans'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL ):
        Log.__init__( self, p_logging=p_logging ) 


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """
        Simulates a state transition based on a state and action. Custom method _simulate_reaction()
        is called.

        Parameters
        ----------
        p_state : State
            System state.
        p_action : Action
            Action to be processed.

        Returns
        -------
        new_state : State
            Result state after state transition.
        """

        self.log(Log.C_LOG_TYPE_I, 'Start simulating a state transition...')
        return self._simulate_reaction( p_state = p_state, p_action = p_action )


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """
        Custom method for a simulated state transition. See method simulate_reaction() for further
        details.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FctSuccess (Log):
    """
    Template class for functions that determine whether or not a state is a success state. 

    Parameters
    ----------
    p_logging 
        Log level (see class Log for more details). Default = Log.C_LOG_ALL.
    """

    C_TYPE = 'Fct Success'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL ):
        Log.__init__( self, p_logging=p_logging ) 


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """
        Assesses the given state regarding success criteria. Custom method _compute_success() is called.

        Parameters
        ----------
        p_state : State
            System state.

        Returns
        -------
        success : bool
            True, if given state is a success state. False otherwise.
        """

        self.log(Log.C_LOG_TYPE_I, 'Assessment for success...')
        return self._compute_success( p_state = p_state )


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        """
        Custom method for assessment for success. See method compute_success() for further details.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FctBroken (Log):
    """
    Template class for functions that determine whether or not a state is a broken state. 

    Parameters
    ----------
    p_logging 
        Log level (see class Log for more details). Default = Log.C_LOG_ALL.
    """

    C_TYPE = 'Fct Broken'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL ):
        Log.__init__( self, p_logging=p_logging ) 


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """
        Assesses the given state regarding breakdown criteria. Custom method _compute_success() is called.

        Parameters
        ----------
        p_state : State
            System state.

        Returns
        -------
        broken : bool
            True, if given state is a breakdown state. False otherwise.
        """

        self.log(Log.C_LOG_TYPE_I, 'Assessment for breakdown...')
        return self._compute_broken( p_state = p_state )


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        """
        Custom method for assessment for breakdown. See method compute_broken() for further details.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SystemBase (FctSTrans, FctSuccess, FctBroken, Mode, Plottable, ScientificObject):
    """
    Base class for state based systems.

    Parameters
    ----------
    p_mode 
        Mode of the system. Possible values are Mode.C_MODE_SIM(default) or Mode.C_MODE_REAL.
    p_latency : timedelta
        Optional latency of the system. If not provided, the internal value of constant C_LATENCY 
        is used by default.
    p_fct_strans : FctSTrans
        Optional external function for state transition. 
    p_fct_success : FctSuccess
        Optional external function for state evaluation 'success'.
    p_fct_broken : FctBroken
        Optional external function for state evaluation 'broken'.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging 
        Log level (see class Log for more details). Default = Log.C_LOG_ALL.

    Attributes
    ----------
    _latency : timedelta
        Latency of the system.
    _state : State
        Current state of system.
    _prev_state : State
        Previous state of system.
    _last_action : Action
        Last action.
    _fct_strans : FctSTrans
        Internal state transition function.
    _fct_success : FctSuccess
        Internal function for state evaluation 'success'.
    _fct_broken : FctBroken
        Internal function for state evaluation 'broken'.
    """

    C_TYPE          = 'System Base'

    C_LATENCY       = timedelta(0, 1, 0)  # Default latency 1s

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_mode = Mode.C_MODE_SIM,
                  p_latency : timedelta = None,
                  p_fct_strans : FctSTrans = None,
                  p_fct_success : FctSuccess = None,
                  p_fct_broken : FctBroken = None,
                  p_mujoco_file = None,
                  p_frame_skip : int = 1,
                  p_state_mapping = None,
                  p_action_mapping = None,
                  p_use_radian : bool = True,
                  p_camera_conf : tuple = (None, None, None),
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL ):

        self._fct_strans            = p_fct_strans
        self._fct_success           = p_fct_success
        self._fct_broken            = p_fct_broken
        self._mujoco_handler        = None
        self._state_space : MSpace  = None
        self._action_space : MSpace = None
        self._state                 = None
        self._prev_state            = None
        self._last_action           = None

        FctSTrans.__init__(self, p_logging=p_logging)
        FctSuccess.__init__(self, p_logging=p_logging)
        FctBroken.__init__(self, p_logging=p_logging)
        Mode.__init__(self, p_mode=p_mode, p_logging=p_logging)
        Plottable.__init__(self, p_visualize=p_visualize)

        self._state_space, self._action_space = self.setup_spaces()

        if p_mujoco_file is not None:
            from mlpro.wrappers.mujoco import MujocoHandler

            self._mujoco_handler = MujocoHandler(
                                        p_mujoco_file=p_mujoco_file, 
                                        p_frame_skip=p_frame_skip,
                                        p_system_state_space=self.get_state_space(),
                                        p_system_action_space=self.get_action_space(),
                                        p_state_mapping=p_state_mapping,
                                        p_action_mapping=p_action_mapping,
                                        p_use_radian=p_use_radian, 
                                        p_camera_conf=p_camera_conf,
                                        p_visualize=p_visualize,
                                        p_logging=p_logging)
            
            self.set_latency(timedelta(0,0.05,0))
        else:
            self.set_latency(p_latency)


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        """
        Static template method to set up and return state and action space of environment.
        
        Returns
        -------
        state_space : MSpace
            State space object
        action_space : MSpace
            Action space object

        """

        return None, None


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        Log.switch_logging(self, p_logging)
        if self._fct_strans is not None:
            self._fct_strans.switch_logging(p_logging)
        if self._fct_success is not None:
            self._fct_success.switch_logging(p_logging)
        if self._fct_broken is not None:
            self._fct_broken.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        """
        Returns latency of the system.
        """

        return self._latency


## -------------------------------------------------------------------------------------------------
    def set_latency(self, p_latency: timedelta = None) -> None:
        """
        Sets latency of the system. If p_latency is None latency will be reset to internal value of 
        attribute C_LATENCY.

        Parameters
        ----------
        p_latency : timedelta
            New latency value 
        """

        if p_latency is None:
            self._latency = self.C_LATENCY
        else:
            self._latency = p_latency


## -------------------------------------------------------------------------------------------------
    def get_state_space(self) -> MSpace:
        return self._state_space


## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return self._action_space
        
        
## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.

        Parameters
        ----------
        p_seed : int
            Seed parameter for an internal random generator
        """

        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=None) -> None:
        """
        Resets the system to an initial state by calling the related custom method _reset().

        Parameters
        ----------
        p_seed : int
            Seed parameter for an internal random generator
        """

        self.log(self.C_LOG_TYPE_I, 'Reset')
        self._num_cycles = 0
        self._reset(p_seed)

        # Put Mujoco here
        if self._mujoco_handler is not None:
            current_state = self.get_state()
            if callable(getattr(self, '_obs_to_mujoco', None)):
                current_state = self._obs_to_mujoco(current_state)

            ob = self._mujoco_handler._reset_simulation(current_state)
            
            self._state = State(self.get_state_space())
            self._state.set_values(ob)

        if self._state is not None:
            self._state.set_initial(True)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        Custom method to reset the system to an initial/defined state. Use method _set_status() to
        set the state.

        Parameters
        ----------
        p_seed : int
            Seed parameter for an internal random generator
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_state(self) -> State:
        """
        Returns current state of the system.
        """

        return self._state


## -------------------------------------------------------------------------------------------------
    def _set_state(self, p_state: State):
        """
        Explicitly sets the current state of the system. Internal use only.
        """

        self._state = p_state


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action: Action) -> bool:
        """
        Processes a state transition based on the current state and a given action. The state
        transition itself is implemented in child classes in the custom method _process_action().

        Parameters
        ----------
        p_action : Action
            Action to be processed

        Returns
        -------
        success : bool
            True, if action processing was successfull. False otherwise.
        """

        self.log(self.C_LOG_TYPE_I, 'Start processing action')

        state = self.get_state()
        result = self._process_action(p_action)
        self._prev_state = state
        self._last_action = p_action

        if result:
            self.log(self.C_LOG_TYPE_I, 'Action processing finished successfully')
            return True
        else:
            self.log(self.C_LOG_TYPE_E, 'Action processing failed')
            return False


## -------------------------------------------------------------------------------------------------
    def _process_action(self, p_action: Action) -> bool:
        """
        Internal custom method for state transition with default implementation. To be redefined in 
        a child class on demand. See method process_action() for further details.
        """

        # 0 Intro
        for agent in p_action.get_elem_ids():
            self.log(self.C_LOG_TYPE_I, 'Actions of agent', agent, '=', p_action.get_elem(agent).get_values())

        # 1 State transition
        if self._mode == self.C_MODE_SIM:
            # 1.1 Simulated state transition
            self._set_state(self.simulate_reaction(self.get_state(), p_action))

        elif self._mode == self.C_MODE_REAL:
            # 1.2 Real state transition

            # 1.2.1 Export action to executing system
            if not self._export_action(p_action):
                self.log(self.C_LOG_TYPE_E, 'Action export failed!')
                return False

            # 1.2.2 Wait for the defined latency
            latency = self.get_latency().total_seconds()
            self.log(Log.C_LOG_TYPE_I, 'Waiting the system latency time of', str(latency), 'seconds...')
            sleep(latency)

            # 1.2.3 Import state from executing system
            if not self._import_state():
                self.log(self.C_LOG_TYPE_E, 'State import failed!')
                return False

        # 2 State evaluation
        state = self.get_state()
        state.set_success(self.compute_success(state))
        state.set_broken(self.compute_broken(state))

        # 3 Outro
        return True


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State = None, p_action: Action = None) -> State:
        """
        Simulates a state transition based on a state and an action. The simulation step itself is
        carried out either by an internal custom implementation in method _simulate_reaction() or
        by an embedded external function.

        Parameters
        ----------
        p_state : State
            Current state.
        p_action : Action
            Action.

        Returns
        -------
        State
            Subsequent state after transition
        """

        if self._fct_strans is not None:
            return self._fct_strans.simulate_reaction(p_state, p_action)
        elif self._mujoco_handler is not None:
            self._mujoco_handler._step_simulation(p_action)

            # Delay because of the simulation
            sleep(self.get_latency().total_seconds())
            ob = self._mujoco_handler._get_obs()

            current_state = self.state_from_mujoco(ob)

            return current_state
        else:
            return self._simulate_reaction(p_state, p_action)


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """
        Custom method for a simulated state transition. Implement this method if no external state
        transition function is used. See method simulate_reaction() for further
        details.
        """
        
        raise NotImplementedError('External FctSTrans object not provided. Please implement inner state transition here.')


## -------------------------------------------------------------------------------------------------    
    def state_from_mujoco(self, p_mujoco_state):
        """
        State conversion method from converting MuJoCo state to MLPro state.
        """
        
        mujoco_state = self._state_from_mujoco(p_mujoco_state)
        mlpro_state = State(self.get_state_space())
        mlpro_state.set_values(mujoco_state)
        return mlpro_state


## -------------------------------------------------------------------------------------------------
    def _state_from_mujoco(self, p_mujoco_state):
        """
        Custom method for to do transition between MuJoCo state and MLPro state. Implement this method
        if the MLPro state has different dimension from MuJoCo state.

        Parameters
        ----------
        p_mujoco_state : Numpy
            MuJoCo state.

        Returns
        -------
        Numpy
            Modified MuJoCo state
        """

        return p_mujoco_state


## -------------------------------------------------------------------------------------------------
    def _export_action(self, p_action: Action) -> bool:
        """
        Mode C_MODE_REAL only: exports given action to be processed externally (for instance by a 
        real hardware). Please redefine. 

        Parameters
        ----------
        p_action : Action
            Action to be exported

        Returns
        -------
        success : bool
            True, if action export was successful. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _import_state(self) -> bool:
        """
        Mode C_MODE_REAL only: imports state from an external system (for instance a real hardware). 
        Please redefine. Please use method _set_state() for internal update.

        Returns
        -------
        success : bool
            True, if state import was successful. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """
        Assesses the given state whether it is a 'success' state. Assessment is carried out either by
        a custom implementation in method _compute_success() or by an embedded external function.

        Parameters
        ----------
        p_state : State
            State to be assessed.

        Returns
        -------
        success : bool
            True, if the given state is a 'success' state. False otherwise.
        """

        if self._fct_success is not None:
            return self._fct_success.compute_success(p_state)
        else:
            return FctSuccess.compute_success(self, p_state)


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        """
        Custom method for assessment for success. Implement this method if no external function is 
        used. See method compute_success() for further details.
        """

        return False


## -------------------------------------------------------------------------------------------------
    def get_success(self) -> bool:
        if self._state is None: return False
        return self._state.get_success()


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """
        Assesses the given state whether it is a 'broken' state. Assessment is carried out either by
        a custom implementation in method _compute_broken() or by an embedded external function.

        Parameters
        ----------
        p_state : State
            State to be assessed.

        Returns
        -------
        broken : bool
            True, if the given state is a 'broken' state. False otherwise.
        """

        if self._fct_broken is not None:
            return self._fct_broken.compute_broken(p_state)
        else:
            return FctBroken.compute_broken(self, p_state)


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        """
        Custom method for assessment for breakdown. Implement this method if no external function is 
        used. See method compute_broken() for further details.
        """

        return False


## -------------------------------------------------------------------------------------------------
    def get_broken(self) -> bool:
        if self._state is None: return False
        return self._state.get_broken()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Sensor (Dimension):
    """
    Template for a sensor.
    """
    
    C_TYPE      = 'Sensor'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Actuator (Dimension):
    """
    Template for an actuator.
    """
    
    C_TYPE      = 'Actuator'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Controller (EventManager):
    """
    Template for a controller that enables access to sensors and actuators.

    Parameters
    ----------
    p_id
        Unique id of the controller.
    p_name : str
        Optional name of the controller.
    p_logging 
        Log level (see class Log for more details). Default = Log.C_LOG_ALL.
    p_kwargs : dict
        Further keyword arguments specific to the controller.

    Attributes
    ----------
    C_EVENT_COMM_ERROR
        Event that is raised on a communication error
    """

    C_TYPE              = 'Controller'
    C_EVENT_COMM_ERROR  = 'COMM_ERROR'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id, p_name:str = '', p_logging : bool = Log.C_LOG_ALL, **p_kwargs):
        self._id        = p_id
        self._name      = p_name
        self._kwargs    = p_kwargs.copy()

        if self._name != '':
            self.C_NAME = self.C_NAME + ' ' + self._name

        self._sensors   = Set()
        self._actuators = Set()

        EventManager.__init__(self, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def reset(self) -> bool:
        """
        Resets the controller by calling custom method _reset().

        Returns
        -------
        result : bool
            True, if successful. False otherwise. Additionally event C_EVENT_COMM_ERROR is raised.
        """

        self.log(Log.C_LOG_TYPE_I, 'Reset started')
        if not self._reset():
            self.log(Log.C_LOG_TYPE_E, 'Reset failed')
            self._raise_event(self.C_EVENT_COMM_ERROR, Event(p_raising_object=self))
            return False
        else:
            self.log(Log.C_LOG_TYPE_I, 'Reset finished successfully')
            return True
        

## -------------------------------------------------------------------------------------------------
    def _reset(self) -> bool:
        """
        Custom reset method.

        Returns
        -------
        result : bool
            True, if successful. False otherwise. 
        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def add_sensor(self, p_sensor : Sensor):
        """
        Adds a sensor to the controller.

        Parameters
        ----------
        p_sensor : Sensor
            Sensor object to be added.
        """

        self._sensors.add_dim(p_dim=p_sensor)

    
## -------------------------------------------------------------------------------------------------
    def get_sensors(self) -> Set:
        """
        Returns the internal set of sensors.

        Returns
        -------
        sensors : Set
            Set of sensors.
        """

        return self._sensors


## -------------------------------------------------------------------------------------------------
    def get_sensor(self, p_id) -> Sensor:
        """
        Returns a sensor.
        """

        return self._sensors.get_dim(p_id=p_id)


## -------------------------------------------------------------------------------------------------
    def get_sensor_value(self, p_id):
        """
        Deternines the value of a sensor by calling custom method _get_sensor_value().

        Parameters
        ----------
        p_id
            Id of the sensor.

        Returns
        -------
        value
            Current value of the sensor or None on a communication error. In that case, event 
            C_EVENT_COMM_ERROR is raised additionally.
        """

        sensor_name = self._sensors.get_dim(p_id).get_name()
        self.log(Log.C_LOG_TYPE_I, 'Getting value of sensor "' + sensor_name + '"...')
        sensor_value = self._get_sensor_value(p_id)

        if sensor_value is None:
            self.log(Log.C_LOG_TYPE_E, 'Value of sensor "' + sensor_name + '" could not be determined')
            self._raise_event( p_event_id=self.C_EVENT_COMM_ERROR, p_event_object=Event(p_raising_object=self, p_sensor_id=p_id) )
            return None
        else:
            self.log(Log.C_LOG_TYPE_I, 'Value of sensor "' + sensor_name + '" = ' + str(sensor_value))
            return sensor_value


## -------------------------------------------------------------------------------------------------
    def _get_sensor_value(self, p_id):
        """
        Custom method to get a sensor value. See method get_sensor_value() for further details.

        Parameters
        ----------
        p_id
            Id of the sensor.

        Returns
        -------
        value
            Current value of the sensor or None on a communication error.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def add_actuator(self, p_actuator : Actuator):
        """
        Adds an actuator to the controller.

        Parameters
        ----------
        p_actuator : Actuator
            Actuator object to be added.
        """

        self._actuators.add_dim(p_dim=p_actuator)


## -------------------------------------------------------------------------------------------------
    def get_actuators(self) -> Set:
        """
        Returns the internal set of actuators.

        Returns
        -------
        actuators : Set
            Set of actuators.
        """

        return self._actuators


## -------------------------------------------------------------------------------------------------
    def get_actuator(self, p_id) -> Actuator:
        """
        Returns an actuator.
        """

        return self._actuators.get_dim(p_id=p_id)


## -------------------------------------------------------------------------------------------------
    def set_actuator_value(self, p_id, p_value) -> bool:
        """
        Sets the value of an actuator by calling custom method _set_actuatur_value().

        Parameters
        ----------
        p_id
            Id of the actuator.
        p_value
            New actuator value.

        Returns
        -------
        successful : bool
            True, if successful. False otherwise. In that case, event C_EVENT_COMM_ERROR is raised
            additionally.
        """

        actuator_name = self._actuators.get_dim(p_id).get_name()
        self.log(Log.C_LOG_TYPE_I, 'Setting new value of actuator "' + actuator_name + '"...')
        successful = self._set_actuator_value(p_id=p_id, p_value=p_value)

        if not successful:
            self.log(Log.C_LOG_TYPE_E, 'Value of actuator "' + actuator_name + '" could not be set')
            self._raise_event(p_event_id=self.C_EVENT_COMM_ERROR, p_event_object=Event(p_raising_object=self, p_actuator_id=p_id))
            return False
        else:
            self.log(Log.C_LOG_TYPE_I, 'Value of actuator "' + actuator_name + '" set to ' + str(p_value))
            return True


## -------------------------------------------------------------------------------------------------
    def _set_actuator_value(self, p_id, p_value) -> bool:
        """
        Custom method to set an actuator value. See method set_sensor_value() for further details.

        Parameters
        ----------
        p_id
            Id of the actuator.
        p_value
            New actuator value.

        Returns
        -------
        successful : bool
            True, if successful. False otherwise.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class System (SystemBase):
    """
    Template for more specific state based systems focussing on hardware control. In addition
    to class SystemBase a controller management is added to access external hardware in mode
    C_MODE_REAL.

    See class SystemBase for further detais.
    """

    C_TYPE      = 'System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode = Mode.C_MODE_SIM, 
                  p_latency : timedelta = None, 
                  p_fct_strans : FctSTrans = None, 
                  p_fct_success : FctSuccess = None, 
                  p_fct_broken : FctBroken = None, 
                  p_mujoco_file = None,
                  p_frame_skip : int = 1,
                  p_state_mapping = None,
                  p_action_mapping = None,
                  p_use_radian : bool = True,
                  p_camera_conf : tuple = (None, None, None),
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):

        SystemBase.__init__( self,
                             p_mode=p_mode, 
                             p_latency=p_latency, 
                             p_fct_strans=p_fct_strans, 
                             p_fct_success=p_fct_success, 
                             p_fct_broken=p_fct_broken, 
                             p_mujoco_file=p_mujoco_file,
                             p_frame_skip=p_frame_skip,
                             p_state_mapping=p_state_mapping,
                             p_action_mapping=p_action_mapping,
                             p_use_radian=p_use_radian,
                             p_camera_conf=p_camera_conf,
                             p_visualize=p_visualize, 
                             p_logging=p_logging )

        self._controllers       = []
        self._mapping_actions   = {}
        self._mapping_states    = {}

    
## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=None) -> None:
        """
        Resets the system by calling method SystemBase.reset() and the related custom method _reset().
        Furthermore, in real operation mode all assigned controllers are reset as well.
        """

        SystemBase.reset(self, p_seed)

        if self.get_mode() == Mode.C_MODE_REAL:
            for con in self._controllers: con.reset()
            self._import_state()


## -------------------------------------------------------------------------------------------------
    def add_controller(self, p_controller : Controller, p_mapping : list) -> bool:
        """
        Adds a controller and a related mapping of states and actions to sensors and actuators.

        Parameters
        ----------
        p_controller : Controller
            Controller object to be added.
        p_mapping : list
            A list of mapping tuples following the syntax ( [Type = 'S' or 'A'], [Name of state/action] [Name of sensor/actuator] )

        Returns
        -------
        successful : bool
            True, if controller and related mapping was added successfully. False otherwise.
        """

        # 0 Preparation
        states      = self._state_space
        actions     = self._action_space
        sensors     = p_controller.get_sensors()
        actuators   = p_controller.get_actuators()
        mapping_int = []
        successful  = True
        self.log(Log.C_LOG_TYPE_I, 'Adding controller "' + p_controller.get_name() + '"...')


        # 1 Check/conversion of mapping entries
        for entry in p_mapping:
            if entry[0] == 'S':
                # 1.1 Check state and sensor
                try:
                    state_id = states.get_dim_by_name(entry[1]).get_id()
                except:
                    self.log(Log.C_LOG_TYPE_E, 'Invalid state component "' + entry[1] + '"')
                    successful = False

                try:
                    sensor_id = sensors.get_dim_by_name(entry[2]).get_id()
                except:
                    self.log(Log.C_LOG_TYPE_E, 'Invalid sensor "' + entry[2] + '"')
                    successful = False
                
                if successful:
                    mapping_int.append( ( entry[0], entry[1], entry[2], state_id, sensor_id ) )
                    
            elif entry[0] == 'A':
                # 1.2 Check action and actuator
                try:
                    action_id = actions.get_dim_by_name(entry[1]).get_id()
                except:
                    self.log(Log.C_LOG_TYPE_E, 'Invalid action component "' + entry[1] + '"')
                    successful = False

                try:
                    actuator_id = actuators.get_dim_by_name(entry[2]).get_id()
                except:
                    self.log(Log.C_LOG_TYPE_E, 'Invalid sensor "' + entry[2] + '"')
                    successful = False
                
                if successful:
                    mapping_int.append( ( entry[0], entry[1], entry[2], action_id, actuator_id ) )

            else:
                raise ParamError('Type "' + entry[0] + '" not valid!')

        if not successful:
            self.log(Log.C_LOG_TYPE_E, 'Controller "' + p_controller.get_name() + '" could not be added')
            return False


        # 2 Takeover of mapping entries and controller
        for entry in mapping_int:
            if entry[0] == 'S':
                self._mapping_states[entry[3]] = ( p_controller, entry[4] )
                self.log(Log.C_LOG_TYPE_I, 'State component "' + entry[1] + '" assigned to sensor "' + entry[2] +'"')
            else:
                self._mapping_actions[entry[3]] = ( p_controller, entry[4] )
                self.log(Log.C_LOG_TYPE_I, 'Action component "' + entry[1] + '" assigned to actuator "' + entry[2] +'"')

        self._controllers.append(p_controller)
        self.log(Log.C_LOG_TYPE_I, 'Controller "' + p_controller.get_name() + '" added successfully')
        return True


## -------------------------------------------------------------------------------------------------
    def _import_state(self) -> bool:

        # 1 Initialization
        new_state  = State( p_state_space=self._state_space )
        successful = True
        self.log(Log.C_LOG_TYPE_I, 'Start importing state...')

        # 2 Import of all related sensor values 
        for state_dim in self._state_space.get_dims():
            state_dim_id   = state_dim.get_id()
            state_dim_name = state_dim.get_name()

            try:
                mapping = self._mapping_states[state_dim_id]
            except:
                self.log(Log.C_LOG_TYPE_E, 'State component "' + state_dim_name + '" not assigned to a controller/sensor')
                successful = False
            else:
                sensor_value = mapping[0].get_sensor_value( p_id = mapping[1] )

                if sensor_value is None: 
                    successful = False
                else:
                    new_state.set_value(p_dim_id=state_dim_id, p_value = sensor_value )

        if not successful: 
            return False

        # 3 Assessment of new state
        new_state.set_success( self.compute_success(new_state) )
        new_state.set_broken( self.compute_broken(new_state) )

        # 4 Set new state
        self._set_state(p_state=new_state)
        return True


## -------------------------------------------------------------------------------------------------
    def _export_action(self, p_action: Action) -> bool:

        # 1 Initialization
        successful = True
        self.log(Log.C_LOG_TYPE_I, 'Start exporting action...')

        # Export auf all related actuator values
        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(p_id=agent_id)

            for action_dim_id in action_elem.get_dim_ids():
                try:
                    mapping = self._mapping_actions[action_dim_id]
                except:
                    action_dim_name = action_elem.get_related_set().get_dim(action_dim_id).get_name()
                    self.log(Log.C_LOG_TYPE_E, 'Action component "' + action_dim_name + '" not assigned to a controller/sensor')
                    successful = False
                else:
                    actuator_value = action_elem.get_value(p_dim_id=action_dim_id)
                    successful = successful and mapping[0].set_actuator_value(p_id=mapping[1], p_value=actuator_value)

        return successful





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class TransferFunction(ScientificObject, Log, Label):
    """
    This class serves as a base class of transfer functions, which provides the main attributes of
    a transfer function. By default, there are several ready-to-use transfer function types
    available. If none of them suits to your transfer function, then you can also select a 'custom'
    type of transfer function and design your own function. Another possibility is to use a function
    approximation functionality provided by MLPro (coming soon).
    
    Parameters
    ----------
    p_name : str
        name of the transfer function.
    p_id : int
        unique id of the transfer function. Default: None.
    p_type : int
        type of the transfer function. Default: None.
    p_unit_in : str
        unit of the transfer function's input. Default: None.
    p_unit_out : str
        unit of the transfer function's output. Default: None.
    p_dt : float
        delta time. Default: 0.01.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    p_args : dict
        extra parameter for each specific transfer function.
        
    Attributes
    ----------
    C_TYPE : str
        type of the base class. Default: 'TransferFunction'.
    C_NAME : str
        name of the transfer function. Default: ''.
    C_TRF_FUNC_LINEAR : int
        linear function. Default: 0.
    C_TRF_FUNC_CUSTOM : int
        custom transfer function. Default: 1.
    C_TRF_FUNC_APPROX : int
        function approximation. Default: 2.
    """

    C_TYPE              = 'TransferFunction'
    C_NAME              = ''
    C_TRF_FUNC_LINEAR   = 0
    C_TRF_FUNC_CUSTOM   = 1
    C_TRF_FUNC_APPROX   = 2


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_id:int=None,
                 p_type:int=None,
                 p_unit_in:str=None,
                 p_unit_out:str=None,
                 p_dt:float=0.01,
                 p_logging=Log.C_LOG_ALL,
                 **p_args) -> None:

        self.C_NAME = p_name
        self.set_type(p_type)
        self.dt = p_dt
        self._unit_in = p_unit_in
        self._unit_out = p_unit_out

        Log.__init__(self, p_logging=p_logging)
        Label.__init__(self, p_name, p_id)
        
        if self.get_type() is not None:
            self.set_function_parameters(p_args)
        else:
            raise NotImplementedError('Please define p_type!')


## -------------------------------------------------------------------------------------------------
    def get_units(self):
        """
        This method provides a functionality to get the SI units of the input and output data.

        Returns
        -------
        self._unit_in : str
            the SI unit of the input data.
        self._unit_out : str
            the SI unit of the output data.
        """
        return self._unit_in, self._unit_out


## -------------------------------------------------------------------------------------------------
    def set_type(self, p_type:int):
        """
        This method provides a functionality to set the type of the transfer function.

        Parameters
        ----------
        p_type : int
            the type of the transfer function.
        """
        self._type = p_type


## -------------------------------------------------------------------------------------------------
    def get_type(self) -> int:
        """
        This method provides a functionality to get the type of the transfer function.

        Returns
        -------
        int
            the type of the transfer function.
        """
        return self._type


## -------------------------------------------------------------------------------------------------
    def __call__(self, p_input, p_range=None):
        """
        This method provides a functionality to call the transfer function by giving an input value.

        Parameters
        ----------
        p_input :
            input value.
        p_range :
            range of the calculation. None means 0. Default: None.

        Returns
        -------
        output :
            output value.
        """
        if self.get_type() == self.C_TRF_FUNC_LINEAR:
            output = self.linear(p_input, p_range)
        
        elif self.get_type() == self.C_TRF_FUNC_CUSTOM:
            output = self.custom_function(p_input, p_range)
        
        elif self.get_type() == self.C_TRF_FUNC_APPROX:
            output = self.function_approximation(p_input, p_range)
        
        return output


## -------------------------------------------------------------------------------------------------
    def set_function_parameters(self, p_args:dict) -> bool:
        """
        This method provides a functionality to set the parameters of the transfer function.

        Parameters
        ----------
        p_args : dict
            set of parameters of the transfer function.

        Returns
        -------
        bool
            true means no parameters are missing.
        """
        if self.get_type() == self.C_TRF_FUNC_LINEAR:
            try:
                self.m = p_args['m']
            except:
                raise NotImplementedError('Parameter m for linear function is missing.')
            try:
                self.b = p_args['b']
            except:
                raise NotImplementedError('Parameter b for linear function is missing.')
        
        elif self.get_type() == self.C_TRF_FUNC_CUSTOM:
            for key, val in p_args.items():
                exec(key + '=val')
        
        elif self.get_type() == self.C_TRF_FUNC_APPROX:
            raise NotImplementedError('Function approximation is not yet available.')
                
        return True


## -------------------------------------------------------------------------------------------------
    def linear(self, p_input:float, p_range=None) -> float:
        """
        This method provides a functionality for linear transfer function.
        
        Formula --> y = mx+b
        y = output
        m = slope
        x = input
        b = y-intercept

        Parameters
        ----------
        p_input : float
            input value.
        p_range :
            range of the calculation. None means 0. Default: None.

        Returns
        -------
        float
            output value.
        """
        if p_range is None:
            return self.m * p_input + self.b
        else:
            points = int(p_range/self.dt)
            output = 0
            for x in range(points+1):
                current_input = p_input + x * self.dt
                output += self.m * current_input + self.b
            return output


## -------------------------------------------------------------------------------------------------
    def custom_function(self, p_input, p_range=None):
        """
        This function represents the template to create a custom function and must be redefined.

        Parameters
        ----------
        p_input :
            input value.
        p_range :
            range of the calculation. None means 0. Default: None.

        Returns
        -------
        float
            output value.
        """  
        if p_range is None:
            raise NotImplementedError('This custom function is missing.')
        else:
            raise NotImplementedError('This custom function is missing.')
        

## -------------------------------------------------------------------------------------------------
    def plot(self, p_x_init:float, p_x_end:float):
        """
        This methods provides functionality to plot the defined function within a range.

        Parameters
        ----------
        p_x_init : float
            The initial value of the input (x-axis).
        p_x_end : float
            The end value of the input (x-axis).
        """
        x_value = []
        output = []
        p_range = p_x_end-p_x_init
        points = int(p_range/self.dt)

        for x in range(points+1):
            current_input = p_x_init + x * self.dt
            x_value.append(current_input)
            output.append(self(current_input, p_range=None))
        
        fig, ax = plt.subplots()
        ax.plot(x_value, output, linewidth=2.0)
        plt.show()


## -------------------------------------------------------------------------------------------------
    def function_approximation(self, p_input, p_range=None):
        """
        The function approximation is not yet ready (coming soon).

        Parameters
        ----------
        p_input : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.
        """
        raise NotImplementedError('Function approximation is not yet available in this version.')
