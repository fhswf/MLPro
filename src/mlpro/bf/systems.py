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
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.1 (2022-11-30)

This module provides models and templates for state based systems.
"""


from time import sleep
from mlpro.bf.various import TStamp, ScientificObject
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
        Log level (see class Log for more details).
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
        Log level (see class Log for more details).
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
        Log level (see class Log for more details).
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
class System (FctSTrans, FctSuccess, FctBroken, Mode, Plottable, ScientificObject):
    """
    Template class for state based systems.

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
        Log level (see class Log for more details).

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

    C_TYPE          = 'System'

    C_LATENCY       = timedelta(0, 1, 0)  # Default latency 1s

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_mode = Mode.C_MODE_SIM,
                  p_latency : timedelta = None,
                  p_fct_strans : FctSTrans = None,
                  p_fct_success : FctSuccess = None,
                  p_fct_broken : FctBroken = None,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL ):

        self._fct_strans   = p_fct_strans
        self._fct_success  = p_fct_success
        self._fct_broken   = p_fct_broken
        self._state_space  = None
        self._action_space = None
        self._state        = None
        self._prev_state   = None
        self._last_action  = None

        FctSTrans.__init__(self, p_logging=Log.C_LOG_NOTHING)
        FctSuccess.__init__(self, p_logging=Log.C_LOG_NOTHING)
        FctBroken.__init__(self, p_logging=Log.C_LOG_NOTHING)
        Mode.__init__(self, p_mode=p_mode, p_logging=p_logging)
        Plottable.__init__(self, p_visualize=p_visualize)
        self.set_latency(p_latency)

        self._state_space, self._action_space = self.setup_spaces()


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
        if self._state is not None:
            self._state.set_initial(True)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        Custom method to reset the system to an initial/defined state. 

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
            sleep(self.get_latency().total_seconds())

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
        Please redefine. Please use method set_state() for internal update.

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