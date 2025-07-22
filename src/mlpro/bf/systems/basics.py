## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.systems
## -- Module  : basics.py
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
## -- 2023-01-24  1.3.4     SY       Quality Assurance on TransferFunction
## -- 2023-01-27  1.4.0     MRD      Integrate MuJoCo as an optional state transition
## -- 2023-02-04  1.5.0     DA       United classes SystemBase, System to new class System
## -- 2023-02-13  1.5.1     MRD      Simplify State Space and Action Space generation
## -- 2023-02-20  1.6.0     DA       Class System: new parent class LoadSave to enable persistence
## -- 2023-02-23  1.6.1     MRD      Add the posibility to customize the action between MLPro and MuJoCo
## -- 2023-03-04  1.7.0     DA       Class System: redefinition of methods load(), _save(), init_plot(),
## --                                update_plot()
## -- 2023-03-07  1.7.1     SY       Remove TransferFunction from bf.systems
## -- 2023-03-07  1.7.2     DA       Bugfix in method System._save()
## -- 2023-03-08  1.7.3     MRD      Auto rename System, set latency from MuJoCo xml file
## -- 2023-03-27  1.8.0     DA       Class System: refactoring of persistence
## -- 2023-04-04  1.9.0     LSB      Class State inherits form Instance
## -- 2023-04-04  1.9.1     LSB      Class State: New method Copy()
## -- 2023-04-05  1.9.2     LSB      Refactor: Copy method of State, copying all the attributes
## -- 2023-04-11  1.9.3     MRD      Add custom reset functionality for MuJoCo
## -- 2023-04-19  1.10.1    LSB      Mew DemoScenario class for system demonstration
## -- 2023-04-20  1.10.2    LSB      Refactoring State-Instance inheritence
## -- 2023-05-03  1.11.0    LSB      Enhancing System Class for task and workflow architecture 
## -- 2023-05-03  1.11.1    LSB      Bug Fix: Visualization for DemoScenario
## -- 2023-05-05  1.12.0    LSB      New Class SystemShared
## -- 2023-05-13  1.13.0    LSB      New parameter p_t_step in simulate reaction method
## -- 2023-05-31  1.13.1    LSB      Updated the copy method of state, for copying the ID
## -- 2023-05-31  1.13.2    LSB      Refactored the t_step handling, to avoid unncessary execution of try block
## -- 2023-05-31  1.13.3    LSB      Removing obsolete env attribute from function
## -- 2023-06-06  1.14.0    LSB      New functions to fetch the functions of a system
## -- 2023-05-01  2.0.0     LSB      New class MultiSystem
## -- 2024-05-14  2.0.1     SY       Migration from MLPro to MLPro-Int-MuJoCo
## -- 2024-05-24  2.1.0     DA       Class State: removed parent class TStamp
## -- 2024-09-07  2.2.0     DA       - Class ActionElement: new property values
## --                                - Renamed Class Controller to SAGateway
## --                                - Renamed method System.add_controller to add_sagateway
## -- 2024-09-09  2.3.0     DA       Class Action: parent TSTamp replaced by Instance
## -- 2024-09-11  2.4.0     DA       - code review and documentation
## --                                - new method State.get_kwargs()
## -- 2024-10-06  2.5.0     DA       New property attribute State.value
## -- 2024-12-11  2.5.0     DA       New method DemoScenario.init_plot()
## -- 2025-07-18  2.6.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.6.0 (2025-07-18)

This module provides models and templates for state based systems.
"""


from time import sleep
from datetime import timedelta
import random

import numpy as np

from mlpro.bf.various import Log, TStampType, ScientificObject, Persistent
from mlpro.bf.ops import Mode, ScenarioBase
from mlpro.bf.exceptions import *
from mlpro.bf.events import Event, EventManager
from mlpro.bf.mt import *
from mlpro.bf.data import *
from mlpro.bf.plot import Figure, Plottable, PlotSettings
from mlpro.bf.math import *
from mlpro.bf.streams import Instance



# Export list for public API
__all__ = [ 'State',
            'Action',
            'ActionElement',
            'FctSTrans',
            'FctSuccess',
            'FctBroken',
            'Sensor',
            'Actuator',
            'SAGateway',
            'SystemShared',
            'System',
            'MultiSystem',
            'DemoScenario' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class State(Instance, Element):
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
    p_kwargs : dict
        Further optional named parameters.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_state_space: MSpace,
                 p_initial: bool = False,
                 p_terminal: bool = False,
                 p_success: bool = False,
                 p_broken: bool = False,
                 p_timeout: bool = False,
                 **p_kwargs):

        Element.__init__(self, p_state_space)
        Instance.__init__(self, p_feature_data=self, **p_kwargs)
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
    def get_feature_data(self) -> Element:
        return self
    

## -------------------------------------------------------------------------------------------------
    def set_feature_data(self, p_feature_data: Element):
        self.set_values(p_feature_data.get_values())


## -------------------------------------------------------------------------------------------------
    def get_kwargs(self):
        return self._get_kwargs()
    

## -------------------------------------------------------------------------------------------------
    def copy(self):
        """
        Returns a copy of the state element

        Returns
        -------
        copied_state: State
            The copy of original state object.
        """

        broken = self.get_broken()
        success = self.get_success()
        initial = self.get_initial()
        terminal = self.get_terminal()
        timeout = self.get_timeout()
        state_space = self.get_related_set()
        copied_state = self.__class__(p_state_space=state_space,
                                      p_broken= broken,
                                      p_success=success,
                                      p_initial=initial,
                                      p_terminal=terminal,
                                      p_timeout=timeout)
        copied_state.set_values(self.get_values())
        copied_state.set_tstamp(self.get_tstamp())
        try:
            copied_state.set_id(self.get_id())
        except:
            pass
        return copied_state


## -------------------------------------------------------------------------------------------------
    values = property( fget=Element.get_values, fset=Element.set_values)




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ActionElement (Element):
    """
    Single entry of an action. See class Action for further details.

    Parameters
    ----------
    p_action_space : Set
        Related action space.
    p_weight : float = 1.0
        Weight of action element. Default = 1.0.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_action_space : Set, 
                  p_weight : float = 1.0 ):

        Element.__init__(self, p_action_space)
        self.set_weight(p_weight)


## -------------------------------------------------------------------------------------------------
    def get_weight(self):
        return self.weight


## -------------------------------------------------------------------------------------------------
    def set_weight(self, p_weight : float):
        self.weight = p_weight


## -------------------------------------------------------------------------------------------------
    values = property( fget=Element.get_values, fset=Element.set_values)


    


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Action(Instance, ElementList):
    """
    Objects of this class represent actions of (multi-)agents. Every element of the internal list is
    related to an agent, and its partial subsection. Action values for the first agent can be added 
    while object instantiation. Action values of further agents can be added by using method self.add_elem().

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
                  p_values: np.ndarray = None,
                  p_tstamp : TStampType = None ):

        ElementList.__init__(self)
        action_elem = None

        if ( p_action_space is not None ) and ( p_values is not None ):
            action_elem = ActionElement(p_action_space)
            action_elem.set_values(p_values)
            self.add_elem(p_agent_id, action_elem)

        Instance.__init__( self, p_feature_data=action_elem, p_tstamp = p_tstamp )


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
    def simulate_reaction(self, p_state: State, p_action: Action, p_t_step : timedelta = None) -> State:
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
        # Check if the p_t_step is to be ignored
        if p_t_step is not None:
            try:
                return self._simulate_reaction( p_state = p_state, p_action = p_action, p_t_step = p_t_step )
            except TypeError:
                return self._simulate_reaction(p_state=p_state, p_action=p_action)
        else:
            return self._simulate_reaction(p_state=p_state, p_action=p_action)


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
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
class SAGateway (EventManager):
    """
    Template for a gateway implementation that enables access to sensors and actuators.

    Parameters
    ----------
    p_id
        Unique id of the gateway.
    p_name : str
        Optional name of the gateway.
    p_logging 
        Log level (see class Log for more details). Default = Log.C_LOG_ALL.
    p_kwargs : dict
        Further keyword arguments specific to the gateway.

    Attributes
    ----------
    C_EVENT_COMM_ERROR
        Event that is raised on a communication error
    """

    C_TYPE              = 'SAGateway'
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
        Resets the gateway by calling custom method _reset().

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
        Adds a sensor to the gateway.

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
        Adds an actuator to the gateway.

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
class SystemShared(Shared):
    """
    A specialised shared object for managing IPC between MultiSystems.

    :TODO: Entry Systems to be handled yet, that get action from outside.

    Parameters
    ----------
    p_range
        The multiprocessing range for the specific process. Default is None.


    Attributes
    ----------
    _spaces
        Spaces of all the systems registered in the Shared Object.

    _states
        States of all the systems registered in the Shared Object.

    _actions
        Corresponding Actions for all the systems.

    _action_dimensions
        All the dimensions present in the Shared Object.

    _mappings
        Mapping configurations for system to action mapping.

    """

    C_NAME = 'System Shared'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_range: int = Range.C_RANGE_NONE):


        Shared.__init__(self,
                        p_range=p_range)

        self._spaces: dict = {}
        self._states: dict = {}
        self._actions: dict = {}
        # self._action_dimensions: set = set()

        # Mappings in the form 'dim : [(output_sys, out_dim), ]'
        self._mappings = {}


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed: int = None):
        """
        Resets the shared object.

        Parameters
        ----------
        p_seed : int
            Seed for reproducibility.
        """

        #  TODO: How do you reset systems in a multiprocess, through the shared object or the workflow itself?
        # TODO: Maybe try to raise an event in the shared object for resetting the system.
        #  But events are not yet supported in the multiprocessing.

        self._states.clear()
        self._actions.clear()

        for system in self._spaces.keys():
            self._states[system] = State(p_state_space=self._spaces[system][0])
            self._actions[system] = np.zeros(len(self._spaces[system][1].get_dims()))


## -------------------------------------------------------------------------------------------------
    def update_state(self, p_sys_id, p_state: State) -> bool:
        """
        Updates the states in the Shared Object.

        Parameters
        ----------
            p_state : State
                The id of the system, for which action is to be fetched.

        Returns
        -------
            bool
        """
        # 1. Add the system state to the shared object for further access
        self._states[p_sys_id] = p_state.copy()

        # 2. Also forward the state values to corresponding mapped dimensions
        self._map_values(p_state=self._states[p_sys_id])


## -------------------------------------------------------------------------------------------------
    def _map_values(self, p_state: State = None, p_action:Action = None):
        """
        Updates the action values based on a new state, in a MultiSystem Context.

        Parameters
        ----------
        p_sys_id
            Id of the system from which the state is received.

        p_state : State
            The State of the system which affects the action.

        """

        # 1. Check is the state is to be mapped?
        if p_state is not None:

            # 1.1 Extract the values of state for each dimension
            for id in p_state.get_dim_ids():
                value = p_state.get_value(id)

                # 1.2 Extract mappings for each of the dimension
                for output_sys, output_dim_type, output_dim in self._map(p_input_dim=id):
                    # Update the values if the receiver dimension is a State
                    if output_dim_type == 'S':
                        self._states[output_sys].set_value(p_dim_id=output_dim, p_value=value)
                    # Update the values if the receiver dimension is an Action
                    if output_dim_type == 'A':
                        action_space = self._spaces[output_sys][1]
                        self._actions[output_sys][action_space.get_dim_ids().index(output_dim)] = value

        # TODO: Check how to get the action dimensions from the action object

        # 2. Check if action is to be mapped?
        if p_action is not None:
            elem_ids = p_action.get_elem_ids()
            action_dims = []
            action_values = []

            # 1.1 Extract the ids and values of action for each element
            for elem_id in elem_ids:
                action_dims.extend(p_action.get_elem(elem_id).get_related_set().get_dim_ids())
                action_values.extend(p_action.get_elem(elem_id).get_values())

            # 1.2 Iterate over the dimensions
            for i,id in enumerate(action_dims):
                value = action_values[i]

                # 1.3 Extract mappings for each dimension
                for output_sys, output_dim_type, output_dim in self._map(p_input_dim=id):

                    # Update the values if receiver dim is a State
                    if output_dim_type == 'S':
                        self._states[output_sys].set_value(p_dim_id=output_dim, p_value=value)
                    # Update the values if the receiver is an Action
                    if output_dim_type == 'A':
                        action_space = self._spaces[output_sys][1]
                        self._actions[output_sys][action_space.get_dim_ids().index(output_dim)] = value


## -------------------------------------------------------------------------------------------------
    def get_actions(self):

        return self._actions


## -------------------------------------------------------------------------------------------------
    def get_action(self, p_sys_id) -> Action:
        """
        Fetches the corresponding action for a particular system.

        Parameters
        ----------
        p_sys_id
            The id of the system, for which action is to be fetched.

        Returns
        -------
        action : Action
            The corresponding action for the system.
        """

        action = Action(p_action_space=self._spaces[p_sys_id][1], p_values=self._actions[p_sys_id])
        return action


## -------------------------------------------------------------------------------------------------
    def get_states(self):
        """
        Fetch the states of all the internal systems

        Returns
        -------
        states: dict
            Returns the state of each of the system registered on the shared object.
        """

        return self._states


## -------------------------------------------------------------------------------------------------
    def get_state(self, p_sys_id) -> State:
        """
        Fetches the state of a particular system from the Shared Object.

        Parameters
        ----------
        p_sys_id
            The id of the system, of which state is to be fetched.

        Returns
        -------
        state : State
            The corresponding state of the system.
        """

        state = self._states[p_sys_id].copy()

        return state


## -------------------------------------------------------------------------------------------------
    def _map(self, p_input_dim = None):
        """
        Maps a dimension to output dimension with info about output sys, output dim type and output dim.

        Parameters
        ----------
        p_sys_id
            Id of the system for which action is to be mapped into the mappings.

        p_state : State
            The State to be mapped into the 'states' dictionary.

        Returns
        -------
        mapping : (output_sys, output_dim_type, output_dim)
            A tuple of tuples of System id and dimension id mappings from State to Action respectively.
        """


        mappings = self._mappings[p_input_dim]

        return mappings


## -------------------------------------------------------------------------------------------------
    def register_system(self,
                        p_sys_id = None,
                        p_state_space : MSpace = None,
                        p_action_space : MSpace = None,
                        p_mappings = None):
        """
        Registers the system in the Shared Object and sets up the dimension to dimension mapping.

        Parameters
        ----------
        p_system : System
            The system to be registered.

        p_mappings :
            Mappings corresponding the system in the form:
            ( (ip dim_type, op dim_type), (input_sys_id, input_dim_id) , (op_sys_id, op_dimension_id) )


        Returns
        -------

        """
        try:

            system_id = p_sys_id
            state_space = p_state_space
            action_space = p_action_space

            # :TODO: Also check if the system is already registered and raise error
            # Register the state and action spaces of the system
            self._spaces[system_id] = (state_space.copy(p_new_dim_ids=False), action_space.copy(p_new_dim_ids=False))
            # Create an initial dummy state of the system in the shared object
            self._states[system_id] = State(self._spaces[system_id][0])

            # Not needed
            # self._action_dimensions.update(*action_space.get_dim_ids())

            # Create dummy zero actions for the system in Shared object
            self._actions[system_id] = np.zeros(len(action_space.get_dim_ids()))

            # If no mappings to be taken care of, Return
            if p_mappings is None:
                return

            # Setup mappings in the shared object
            for (in_dim_type, output_dim_type), (input_sys_id, input_dim), (output_sys_id, output_dim) in p_mappings:

                if input_dim not in self._mappings.keys():
                    self._mappings[input_dim] = []

                self._mappings[input_dim].append((output_sys_id, output_dim_type, output_dim))

            return

        except:

            raise Error("Registration of the system failed. Possible reason maybe false provision of mappings")

        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class System (FctSTrans, FctSuccess, FctBroken, Task, Mode, Plottable, Persistent, ScientificObject):
    """
    Base class for state based systems.

    Parameters
    ----------
    p_id
        Optional external id
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_THREAD.
    p_autorun : int
        On value C_AUTORUN_RUN method run() is called imediately during instantiation.
        On vaule C_AUTORUN_LOOP method run_loop() is called.
        Value C_AUTORUN_NONE (default) causes an object instantiation without starting further
        actions.    
    p_class_shared = None
        Optional class for a shared object (class Shared or a child class of Shared)
    p_mode = Mode.C_MODE_SIM
        Mode of the system. Possible values are Mode.C_MODE_SIM(default) or Mode.C_MODE_REAL.
    p_latency : timedelta = None
        Optional latency of the system. If not provided, the internal value of constant C_LATENCY 
        is used by default.
    p_t_step : timedelta = None
        ...
    p_fct_strans : FctSTrans
        Optional external function for state transition. 
    p_fct_success : FctSuccess
        Optional external function for state evaluation 'success'.
    p_fct_broken : FctBroken
        Optional external function for state evaluation 'broken'.
    p_mujoco_file
        Path to XML file for MuJoCo model.
    p_frame_skip : int
        MuJoCo only: frame to be skipped every step. Default = 1.
    p_state_mapping = None
        MuJoCo only: state mapping if the MLPro state and MuJoCo state have different naming.
    p_action_mapping = None
        MuJoCo only: action mapping if the MLPro action and MuJoCo action have different naming.
    p_use_radian : bool
        MuJoCo only: use radian if the action and the state based on radian unit. Default = True.
    p_camera_conf : tuple
        MuJoCo only: default camera configuration on MuJoCo Simulation (xyz position, elevation, distance).
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

    C_TYPE          = 'System'

    C_LATENCY       = timedelta(0, 1, 0)  # Default latency 1s

    C_PLOT_ACTIVE   = True

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id = None,
                  p_name : str =None,
                  p_range_max : int = Async.C_RANGE_NONE,
                  p_autorun = Task.C_AUTORUN_NONE,
                  p_class_shared = None,
                  p_mode = Mode.C_MODE_SIM,
                  p_latency : timedelta = None,
                  p_t_step : timedelta = None,
                  p_fct_strans : FctSTrans = None,
                  p_fct_success : FctSuccess = None,
                  p_fct_broken : FctBroken = None,
                  p_mujoco_file = None,
                  p_frame_skip : int = 1,
                  p_state_mapping = None,
                  p_action_mapping = None,
                  p_camera_conf : tuple = (None, None, None),
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        self._fct_strans            = p_fct_strans
        self._fct_success           = p_fct_success
        self._fct_broken            = p_fct_broken
        self._mujoco_handler        = None
        self._state_space : MSpace  = None
        self._action_space : MSpace = None
        self._state                 = None
        self._prev_state            = None
        self._last_action           = None
        self._gateways              = []
        self._mapping_actions       = {}
        self._mapping_states        = {}
        self._t_step                = p_t_step

        if p_mujoco_file is not None:
            try:
                from mlpro_int_mujoco.wrappers import MujocoHandler
            except:
                raise ImplementationError('MLPro-Int-MuJoCo package is missing! Please refer to https://mlpro-int-mujoco.readthedocs.io/en/latest/')

            if p_name is not None:
                self.C_NAME = p_name
            else:
                self.C_NAME = p_mujoco_file.split("/")[-1][:p_mujoco_file.split("/")[-1].find(".")]

            self._mujoco_file    = p_mujoco_file
            self._frame_skip     = p_frame_skip
            self._state_mapping  = p_state_mapping
            self._action_mapping = p_action_mapping
            self._camera_conf    = p_camera_conf

            self._mujoco_handler = MujocoHandler(
                                        p_mujoco_file=self._mujoco_file,
                                        p_frame_skip=self._frame_skip,
                                        p_state_mapping=self._state_mapping,
                                        p_action_mapping=self._action_mapping,
                                        p_camera_conf=self._camera_conf,
                                        p_visualize=p_visualize,
                                        p_logging=p_logging)

            self._state_space, self._action_space = self._mujoco_handler.setup_spaces()
            # Get Latency
            mujoco_latency = self._mujoco_handler.get_latency()

            if mujoco_latency is not None:
                self.set_latency(timedelta(0,mujoco_latency,0))
            else:
                if p_latency is not None:
                    self.set_latency(p_latency)
                else:
                    raise ImplementationError('Please provide p_latency or set the timestep on the MuJoCo xml file!')
        else:
            self._mujoco_file = None
            self._state_space, self._action_space = self.setup_spaces()
            self.set_latency(p_latency)

        FctSTrans.__init__(self, p_logging=p_logging)
        FctSuccess.__init__(self, p_logging=p_logging)
        FctBroken.__init__(self, p_logging=p_logging)
        Mode.__init__(self, p_mode=p_mode, p_logging=p_logging)
        Plottable.__init__(self, p_visualize=p_visualize)
        Persistent.__init__(self, p_id=p_id, p_logging=p_logging)


        Task.__init__(self,
                        p_id=p_id,
                        p_name =p_name,
                        p_range_max = p_range_max,
                        p_autorun = p_autorun,
                        p_class_shared=p_class_shared,
                        p_visualize=p_visualize,
                        p_logging=p_logging,
                        **p_kwargs)

        self._registered_on_so = False


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
    def _reduce_state(self, p_state:dict, p_path:str, p_os_sep:str, p_filename_stub:str):
        """
        An embedded MuJoCo system can not be pickled and needs to be removed from the pickle stream.
        """

        p_state['_mujoco_handler'] = None


## -------------------------------------------------------------------------------------------------
    def _complete_state(self, p_path:str, p_os_sep:str, p_filename_stub:str):

        if self._mujoco_file is None: return

        try:
            from mlpro_int_mujoco.wrappers import MujocoHandler
        except:
            raise ImplementationError('MLPro-Int-MuJoCo package is missing! Please refer to https://mlpro-int-mujoco.readthedocs.io/en/latest/')


        self._mujoco_handler = MujocoHandler( p_mujoco_file=self._mujoco_file,
                                              p_frame_skip=self._frame_skip,
                                              p_state_mapping=self._state_mapping,
                                              p_action_mapping=self._action_mapping,
                                              p_camera_conf=self._camera_conf,
                                              p_visualize=self.get_visualization(),
                                              p_logging=self.get_log_level() )

        self._mujoco_handler._system_state_space  = self.get_state_space()
        self._mujoco_handler._system_action_space = self.get_action_space()


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
    def get_fct_strans(self):
        """
        Returns the state transition function of the system, if exists, otherwise, the system itself.

        Returns
        -------
        fct_strans: FctSTrans
            State transition function of the system, if exists. Otherwise, system itself.

        """
        if self._fct_strans is not None:
            return self._fct_strans
        else:
            return self


## -------------------------------------------------------------------------------------------------
    def get_fct_broken(self):
        """
        Returns the broken computation function of the system, if exists, otherwise, the system itself.

        Returns
        -------
        fct_broken: FctBroken
            Broken computation function of the system, if exists. Otherwise, system itself.

        """
        if self._fct_broken is not None:
            return self._fct_broken
        else:
            return self


## -------------------------------------------------------------------------------------------------
    def get_fct_success(self):
        """
        Returns the Success computation function of the system, if exists, otherwise, the system itself.

        Returns
        -------
        fct_success: FctSuccess
            Success computation function of the system, if exists. Otherwise, system itself.

        """
        if self._fct_success is not None:
            return self._fct_success
        else:
            return self


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
        Resets the system to an initial state. If MuJoCo is not used, the custom method _reset() is
        called.

        Parameters
        ----------
        p_seed : int
            Seed parameter for an internal random generator
        """

        self.log(self.C_LOG_TYPE_I, 'Reset')
        self._num_cycles = 0

        # Put Mujoco here
        if self._mujoco_handler is not None:
            try:
                self._reset(p_seed)
            except NotImplementedError:
                ob = self._mujoco_handler._reset_simulation()
                self._state = State(self.get_state_space())
                self._state.set_values(ob)

        else:
            self._reset(p_seed)


        if self._state is not None:
            self._state.set_initial(True)

        if self.get_mode() == Mode.C_MODE_REAL:
            for con in self._gateways: con.reset()
            self._import_state()


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
    def add_gateway(self, p_gateway : SAGateway, p_mapping : list) -> bool:
        """
        Adds a sensor/actuator-gateway and a related mapping of states and actions to sensors and actuators.

        Parameters
        ----------
        p_gateway : SAGateway
            gateway object to be added.
        p_mapping : list
            A list of mapping tuples following the syntax ( [Type = 'S' or 'A'], [Name of state/action] [Name of sensor/actuator] )

        Returns
        -------
        successful : bool
            True, if gateway and related mapping was added successfully. False otherwise.
        """

        # 0 Preparation
        states      = self._state_space
        actions     = self._action_space
        sensors     = p_gateway.get_sensors()
        actuators   = p_gateway.get_actuators()
        mapping_int = []
        successful  = True
        self.log(Log.C_LOG_TYPE_I, 'Adding gateway "' + p_gateway.get_name() + '"...')


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
            self.log(Log.C_LOG_TYPE_E, 'SA-Gateway "' + p_gateway.get_name() + '" could not be added')
            return False


        # 2 Takeover of mapping entries and gateway
        for entry in mapping_int:
            if entry[0] == 'S':
                self._mapping_states[entry[3]] = ( p_gateway, entry[4] )
                self.log(Log.C_LOG_TYPE_I, 'State component "' + entry[1] + '" assigned to sensor "' + entry[2] +'"')
            else:
                self._mapping_actions[entry[3]] = ( p_gateway, entry[4] )
                self.log(Log.C_LOG_TYPE_I, 'Action component "' + entry[1] + '" assigned to actuator "' + entry[2] +'"')

        self._gateways.append(p_gateway)
        self.log(Log.C_LOG_TYPE_I, 'SA-Gateway "' + p_gateway.get_name() + '" added successfully')
        return True


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
    def get_so(self) -> SystemShared:

        return Task.get_so(self)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_action:Action, p_t_step : timedelta = None):
        """
        Run method that runs the system as a task. It runs the process_action() method of the system with
        action as a parameter.

        Parameters
        ----------
        p_t_step : timedelta
            Time for which the system must be simulated.

        """

        action = self.get_so().get_action(p_sys_id = self.get_id())
        self.process_action(p_action=action, p_t_step = p_t_step)
        self.get_so().update_state(p_sys_id = self.get_id(), p_state = self.get_state())


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a state transition based on the current state and a given action. The state
        transition itself is implemented in child classes in the custom method _process_action().

        Parameters
        ----------
        p_action : Action
            Action to be processed

        p_t_step : timedelta
            The timestep for which the system is to be simulated

        Returns
        -------
        success : bool
            True, if action processing was successfull. False otherwise.
        """

        self.log(self.C_LOG_TYPE_I, 'Start processing action')

        state = self.get_state()

        if p_t_step is None:
            t_step = self._t_step
        else:
            t_step = p_t_step

        # Check if the t_step shall be ignored, when None
        if t_step is not None:
            try:
                result = self._process_action(p_action, p_t_step = t_step)
            except TypeError:
                result = self._process_action(p_action)
        else:
            result = self._process_action(p_action)

        self._prev_state  = state
        self._last_action = p_action

        if result:
            self.log(self.C_LOG_TYPE_I, 'Action processing finished successfully')
            return True
        else:
            self.log(self.C_LOG_TYPE_E, 'Action processing failed')
            return False


## -------------------------------------------------------------------------------------------------
    def _process_action(self, p_action: Action, p_t_step : timedelta = None) -> bool:
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
            # Check if the p_t_step shall be ignored
            if p_t_step is not None:
                try:
                    self._set_state(self.simulate_reaction(self.get_state(), p_action, p_t_step = p_t_step ))
                except TypeError:
                    self._set_state(self.simulate_reaction(self.get_state(), p_action))
            else:
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
    def simulate_reaction(self, p_state: State = None, p_action: Action = None, p_t_step:timedelta = None) -> State:
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
        # 1. Check if there is an external simulation function provided to the System
        if self._fct_strans is not None:

            # 1.1 Check if there is a valid timestep, or if it shall be ignored?
            if p_t_step is not None:
                try:
                    return self._fct_strans.simulate_reaction(p_state, p_action, p_t_step)
                except TypeError:
                    return self._fct_strans.simulate_reaction(p_state, p_action)
            else:
                return self._fct_strans.simulate_reaction(p_state, p_action)

        # 2. Check if there's is Mujoco Handler
        elif self._mujoco_handler is not None:
            # Check if there is changing in action
            action = self.action_to_mujoco(p_action)
            self._mujoco_handler._step_simulation(action)

            # Delay because of the simulation
            sleep(self.get_latency().total_seconds())
            ob = self._mujoco_handler._get_obs()

            current_state = self.state_from_mujoco(ob)

            return current_state

        # 3. Or else execute the user defined reaction simulation
        else:
            # 3.1 Check if the p_t_step shall be ignored
            if p_t_step is not None:
                try:
                    return self._simulate_reaction(p_state, p_action, p_t_step)
                except TypeError:
                    return self._simulate_reaction(p_state, p_action)
            else:
                return self._simulate_reaction(p_state, p_action)


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step:timedelta = None) -> State:
        """
        Custom method for a simulated state transition. Implement this method if no external state
        transition function is used. See method simulate_reaction() for further
        details.
        """

        raise NotImplementedError('External FctSTrans object not provided. Please implement inner state transition here.')


## -------------------------------------------------------------------------------------------------
    def action_to_mujoco(self, p_mlpro_action):
        """
        Action conversion method from converting MLPro action to MuJoCo action.
        """
        action = p_mlpro_action.get_sorted_values()
        return self._action_to_mujoco(action)


## -------------------------------------------------------------------------------------------------
    def _action_to_mujoco(self, p_mlpro_action):
        """
        Custom method for to do transition between MuJoCo state and MLPro state. Implement this method
        if the MLPro state has different dimension from MuJoCo state.

        Parameters
        ----------
        p_mujoco_state : Numpy
            MLPro action.

        Returns
        -------
        Numpy
            Modified MLPro action
        """

        return p_mlpro_action


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
                self.log(Log.C_LOG_TYPE_E, 'State component "' + state_dim_name + '" not assigned to a gateway/sensor')
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
                    self.log(Log.C_LOG_TYPE_E, 'Action component "' + action_dim_name + '" not assigned to a gateway/sensor')
                    successful = False
                else:
                    actuator_value = action_elem.get_value(p_dim_id=action_dim_id)
                    successful = successful and mapping[0].set_actuator_value(p_id=mapping[1], p_value=actuator_value)

        return successful


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
    def init_plot( self,
                   p_figure:Figure = None,
                   p_plot_settings : PlotSettings = None,
                   **p_kwargs):

        if self._mujoco_handler is not None: return
        super().init_plot(p_figure=p_figure, p_plot_settings=p_plot_settings, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        if self._mujoco_handler is not None: return
        super().update_plot(**p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiSystem(Workflow, System):

    """
    A complex system of systems.
    
    Parameters
    ----------
    p_name
    p_id
    p_range_max
    p_autorun
    p_class_shared
    p_mode
    p_latency
    p_t_step
    p_fct_strans
    p_fct_success
    p_fct_broken
    p_mujoco_file
    p_frame_skip
    p_state_mapping
    p_action_mapping
    p_camera_conf
    p_visualize
    p_logging
    p_kwargs
    """

    C_TYPE = 'Multi-System'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_name: str = None, 
                 p_id = None,
                 p_range_max=Async.C_RANGE_NONE,
                 p_autorun = Task.C_AUTORUN_NONE,
                 p_class_shared = SystemShared,
                 p_mode=Mode.C_MODE_SIM, 
                 p_latency: timedelta = None,
                 p_t_step:timedelta = None, 
                 p_fct_strans: FctSTrans = None, 
                 p_fct_success: FctSuccess = None, 
                 p_fct_broken: FctBroken = None, 
                 p_mujoco_file=None, 
                 p_frame_skip: int = 1, 
                 p_state_mapping=None, 
                 p_action_mapping=None, 
                 p_camera_conf: tuple = (None, None, None), 
                 p_visualize: bool = False, 
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        System.__init__( self,
                         p_name=p_name,
                         p_id = p_id,
                         p_range_max = p_range_max,
                         p_autorun=p_autorun,
                         p_class_shared = p_class_shared,
                         p_mode = p_mode, 
                         p_latency=p_latency, 
                         p_t_step = p_t_step,
                         p_fct_strans=p_fct_strans, 
                         p_fct_success=p_fct_success, 
                         p_fct_broken=p_fct_broken,
                         p_mujoco_file=p_mujoco_file,
                         p_frame_skip=p_frame_skip, 
                         p_state_mapping=p_state_mapping,
                         p_action_mapping=p_action_mapping, 
                         p_camera_conf=p_camera_conf, 
                         p_visualize=p_visualize, 
                         p_logging=p_logging )

        Workflow.__init__(self, 
                          p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_class_shared = p_class_shared,
                          p_visualize = p_visualize, 
                          p_logging = p_logging,
                          **p_kwargs)
        
        self._subsystems = []
        self._subsystem_ids = []
        self._t_step = p_t_step


## -------------------------------------------------------------------------------------------------
    def add_system(self, p_system : System, p_mappings):
        """
        Adds sub system to the MultiSystem.

        Parameters
        ----------
        p_system : System
            The system to be added.

        p_mappings : list
            The mappings corresponding the system in the form :
            [ ((ip dim_type, op dim_type), (input_sys_id, input_dim_id), (op_sys_id, op_dimension_id)), ... ]

        Returns
        -------

        """
        # Register the systems in native list.
        self._subsystems.append(p_system)

        # Register the systems ids in a native list
        self._subsystem_ids.append(p_system.get_id())

        # Register the system on the shared object (SystemShared).
        p_system._registered_on_so = self.get_so().register_system(p_sys_id = p_system.get_id(),
                                                                      p_state_space=p_system.get_state_space(),
                                                                      p_action_space=p_system.get_action_space(),
                                                                      p_mappings = p_mappings)

        # Add the system as a task in the workflow
        self.add_task(p_system)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        Resets the MultiSystem and all the sub-systems inside.

        Parameters
        ----------
        p_seed: int
            Seed for the purpose of reproducibility

        """
        # Reset the shared object (SystemShared).
        self.get_so().reset(p_seed=p_seed)

        # Reset all the subsystems
        for system in self._subsystems:
            system.reset(p_seed = p_seed)


## -------------------------------------------------------------------------------------------------
    def get_subsystem_ids(self):
        return self._subsystem_ids


## -------------------------------------------------------------------------------------------------
    def get_subsystems(self):
        return self._subsystems


## -------------------------------------------------------------------------------------------------
    def get_subsystem(self, p_system_id) -> System:
        """
        Returns a sub system from the MultiSystem.

        Parameters
        ----------
        p_system_id
            Id of the system to be returned

        Returns
        -------
        sub_system: System
            The system to be returned by ID.

        """
        return self._subsystems[self._subsystem_ids.index(p_system_id)]


## -------------------------------------------------------------------------------------------------
    def get_states(self):
        """
        Returns a list of the states of all the Sub-Systems in the MultiSystem.

        Returns
        -------
        states : dict
            States of all the subsystems.
        """

        so = self.get_so()

        return so.get_states()


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State = None, p_action: Action = None, p_t_step : timedelta = None) -> State:
        """
        Simulates the multisystem, based on the action and time step.

        Parameters
        ----------
        p_state: State
            State of the system.

        p_action: Action.
            Action provided externally for the simulation of the system.

        Returns
        -------
        current_state: State
            The new state of the system after simulation.

        """

        # 1. Register the MultiSystem in the SO, as it is not yet registered, unlike subsystems are
        # registered in the add system call.

        if not self._registered_on_so:
            self._registered_on_so = self.get_so().register_system(p_sys_id=self.get_id(),
                                                                   p_state_space=self.get_state_space(),
                                                                   p_action_space = self.get_action_space())

        # Calculate the greatest possible timestep
        # if self._t_step is None:
        #     ts_list = []
        #     for id in self.get_subsystem_ids():
        #         sys_ts = self.get_subsystem(id)._t_step
        #         if sys_ts is not None:
        #             ts_list.append(self.get_subsystem(id)._t_step)




        # Recommend using Time() instead of using timedelta

        # 2. Get SO
        so = self.get_so()

        # 3. Forward the input action to the corresponding systems
        so._map_values(p_action=p_action)

        # 4. Run the workflow
        self.run(p_action = self.get_so().get_actions(), p_t_step = self._t_step)

        # 5. Return the new state at current timestep
        return so.get_state(p_sys_id=self.get_id())


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """
        Returns true if the system is broken
        """
        # TODO: Shall return true if any of the system returns true?

        return False


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """
        Returns true if the system has reached success.
        
        Parameters
        ----------
        p_state

        Returns
        -------

        # TODO: Shall return true if any of the system returns true?

        """
        return False





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DemoScenario(ScenarioBase):

    """
        Demo Scenario Class to demonstrate systems.
        
        Parameters
        ----------
        p_system : System
            Mandatory parameter, takes the system for the scenario.
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
        p_action_pattern : str
            The action pattern to be used for demonstration. Default is C_ACTION_RANDOM
        p_action_random : list
            The action to be executed in constant mode. Mandatory when the mdoe is constant.
        p_id
            Optional external id
        p_cycle_limit : int
            Maximum number of cycles. Default = 0 (no limit).
        p_auto_setup : bool
            If True custom method setup() is called after initialization.
        p_visualize : bool
            Boolean switch for visualisation. Default = True.
        p_logging
            Log level (see constants of class Log). Default: Log.C_LOG_ALL.  
    """    

    C_NAME              = 'Demo System Scenario'
    C_ACTION_RANDOM     = 'random'
    C_ACTION_CONSTANT   = 'constant'

## -------------------------------------------------------------------------------------------------
    def  __init__(self,
                 p_system : System,
                 p_mode, 
                 p_action_pattern : str = 'random',
                 p_action : list = None,
                 p_id=None, 
                 p_cycle_limit=0, 
                 p_auto_setup: bool = True, 
                 p_visualize: bool = True, 
                 p_logging=Log.C_LOG_ALL):
        

        self._system = p_system
        self._action_pattern = p_action_pattern
        self._action = p_action

        ScenarioBase.__init__(self,
                              p_mode = p_mode, 
                              p_id = p_id, 
                              p_cycle_limit = p_cycle_limit, 
                              p_auto_setup = p_auto_setup, 
                              p_visualize = p_visualize, 
                              p_logging = p_logging)

        self._action_length = len(self._system.get_action_space().get_dims())

        if (self._action_pattern == DemoScenario.C_ACTION_CONSTANT): 

            if self._action is None:
                raise ParamError("Please provide a value for action, when running in constant action mode.")
            
            if not self._action_length == len(self._action):
                raise ParamError("Please provide the action as a list of length equal to the number"+
                                 " of dimenstions in the action space of the system.")
        
        self.reset()


## -------------------------------------------------------------------------------------------------
    def setup(self):

        """Set's up the system spaces."""

        self._system.setup_spaces()

## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        
        """
        Returns the latency of the system.
        """
        
        return self._system.get_latency()


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        """
        Resets the Scenario and the system. Sets up the action and state spaces of the system.

        Parameters
        ----------
        p_seed
            Seed for the purpose of reproducibility.
        """

        self._system.reset(p_seed = p_seed)
        self._system.init_plot()


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        """
        Runs the custom scenario cycle, through the run method of the scenario base. 
        Checks and returns the brokent state, false otherwise.
        """
        
        self.log(Log.C_LOG_TYPE_I, "Generating new action")
        
        action = self._get_next_action()

        self._system.process_action(p_action=action)

        broken = self._system.compute_broken(p_state=self._system.get_state()) 

        return False, broken, False, False


## -------------------------------------------------------------------------------------------------
    def _get_next_action(self):

        """
        Generates new action based on the pattern provided by the user.
        """
        
        action_space = self._system.get_action_space()

        if self._action_pattern == self.C_ACTION_CONSTANT:
            return Action(p_action_space = action_space, p_values = self._action)

        action = []

        for dim in action_space.get_dims():
            if dim.get_base_set() in (Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_R):
                action.append(random.uniform(*dim.get_boundaries()))

            elif dim.get_base_set() == Dimension.C_BASE_SET_Z:
                action.append(random.randint(*dim.get_boundaries()))

        return Action(p_action_space=action_space, p_values=action)
    

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure = None, p_plot_settings = None, p_window_title = None):
        self._system.init_plot( p_figure = p_figure, p_plot_settings= p_plot_settings, p_window_title = p_window_title )
 
 
## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
    #    super().update_plot(**p_kwargs)
       self._system.update_plot(**p_kwargs)