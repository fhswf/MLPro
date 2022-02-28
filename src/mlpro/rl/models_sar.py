## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.rl
## -- Module  : models_sar.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-18  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-06-15  1.0.1     SY       Bugfixing in class ActionElement
## -- 2021-08-28  1.0.2     DA       Bugfixes and minor improvements
## -- 2021-09-11  1.0.3     MRD      Change Header information to match our new library name
## -- 2021-09-19  1.0.4     MRD      Change SARBuffer Class and Inherits SARBufferElement with base
## --                                class Buffer
## -- 2021-09-25  1.0.5     MRD      Remove Previous state into the buffer. Add Next state to the buffer
## --                                Remove clearing buffer on every reset. The clearing buffer should
## --                                be controlled from the policy
## -- 2021-10-05  1.0.6     DA       Class State: new attributes done, broken and related methods 
## -- 2021-10-05  1.0.7     SY       Bugfixes and minor improvements
## -- 2021-12-12  1.0.8     DA       Reward type C_TYPE_EVERY_ACTION disabled
## -- 2021-12-19  1.1.0     DA       Class State: replaced term 'done' by 'success'
## -- 2021-12-21  1.1.1     DA       Class State: 
## --                                - new attributes _initial, _terminal, _timeout
## --                                - all attributes can be set as parameters of the constructor
## -- 2022-02-11  1.1.2     DA       Class Reward: dealing with arrays as overall reward
## -- 2022-02-28  1.1.3     SY       Refactoring due to auto generated ID in class Dimension
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.3 (2022-02-28)

This module provides model classes for state, action and reward data and their buffering.
"""

from mlpro.bf.data import *
from mlpro.bf.ml import *
from mlpro.bf.plot import *


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class State(Element, TStamp):
    """
    State of an environment as an element of a given state space. Additionally, the state can be
    labeled with various properties.

    Parameters
    ----------
    p_state_space : MSpace
        State space of the related environment.
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
class ActionElement(Element):

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_action_space: Set, p_weight=1.0) -> None:
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
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_agent_id=0, p_action_space=None, p_values: np.ndarray = None):
        """
        Parameters:
            p_agent_id        Unique id of (first) agent to be added
            p_action_space    Action space of (first) agent to be added
            p_values          Action values of (first) agent to be added
        """

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
class Reward(TStamp):
    """
    Objects of this class represent rewards of environments. The internal structure
    depends on the reward type. Three types are supported as listed below.
    """

    C_TYPE_OVERALL = 0  # Reward is a scalar (default)
    C_TYPE_EVERY_AGENT = 1  # Reward is a scalar for every agent
    C_TYPE_EVERY_ACTION = 2  # Reward is a scalar for every agent and action

    C_VALID_TYPES = [C_TYPE_OVERALL, C_TYPE_EVERY_AGENT, C_TYPE_EVERY_ACTION]

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_type=C_TYPE_OVERALL, p_value=0):
        """
        Parameters:
            p_type          Reward type (default: C_TYPE_OVERALL)
            p_value         Overall reward value (reward type C_TYPE_OVERALL only)
        """

        if p_type not in self.C_VALID_TYPES:
            raise ParamError('Reward type ' + str(p_type) + ' not supported.')

        TStamp.__init__(self)
        self.type = p_type
        self.agent_ids = []
        self.rewards = []
        if self.type == self.C_TYPE_OVERALL:
            self.set_overall_reward(p_value)

    ## -------------------------------------------------------------------------------------------------
    def get_type(self):
        return self.type

    ## -------------------------------------------------------------------------------------------------
    def is_rewarded(self, p_agent_id) -> bool:
        try:
            i = self.agent_ids.index(p_agent_id)
            return True
        except ValueError:
            return False

    ## -------------------------------------------------------------------------------------------------
    def set_overall_reward(self, p_reward) -> bool:
        if self.type != self.C_TYPE_OVERALL:
            return False

        try:
            self.overall_reward = p_reward[0]
        except:
            self.overall_reward = p_reward

        return True

    ## -------------------------------------------------------------------------------------------------
    def get_overall_reward(self):
        try:
            return self.overall_reward[0]
        except:
            return self.overall_reward

    ## -------------------------------------------------------------------------------------------------
    def add_agent_reward(self, p_agent_id, p_reward) -> bool:
        if self.type != self.C_TYPE_EVERY_AGENT:
            return False
        self.agent_ids.append(p_agent_id)
        self.rewards.append(p_reward)
        return True

    ## -------------------------------------------------------------------------------------------------
    def get_agent_reward(self, p_agent_id):
        if self.type == self.C_TYPE_OVERALL:
            return self.overall_reward

        try:
            i = self.agent_ids.index(p_agent_id)
        except ValueError:
            return None

        return self.rewards[i]

    ## -------------------------------------------------------------------------------------------------
    def add_action_reward(self, p_agent_id, p_action_id, p_reward) -> bool:
        if self.type != self.C_TYPE_EVERY_ACTION:
            return False

        try:
            i = self.agent_ids.index(p_agent_id)
            r = self.rewards[i]
            r[0].append(p_action_id)
            r[1].append(p_reward)
        except ValueError:
            self.agent_ids.append(p_agent_id)
            self.rewards.append([[p_action_id], [p_reward]])

        return True

    ## -------------------------------------------------------------------------------------------------
    def get_action_reward(self, p_agent_id, p_action_id):
        if self.type != self.C_TYPE_EVERY_ACTION:
            return None

        try:
            i_agent = self.agent_ids.index(p_agent_id)
        except ValueError:
            return None

        try:
            r = self.rewards[i_agent]
            i_action = r[0].index(p_action_id)
            return r[1][i_action]
        except:
            return None


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SARSElement(BufferElement):
    """
    Element of a SARSBuffer.
    """

    def __init__(self, p_state: State, p_action: Action, p_reward: Reward, p_state_new: State):
        """
        Parameters:
            p_state         State of an environment
            p_action        Action of an agent
            p_reward        Reward of an environment
            p_state_new     State of the environment as reaction to the action
        """

        super().__init__({"state": p_state, "action": p_action, "reward": p_reward, "state_new": p_state_new})


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SARSBuffer(Buffer):
    """
    State-Action-Reward-State-Buffer in dictionary.
    """

    pass
