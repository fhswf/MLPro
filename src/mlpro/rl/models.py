## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.models
## -- Module  : rl
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-18  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-06-13  1.1.1     DA       New method Agent.reset_episode() and call in class Training;
## --                                Some bugfixes in class MultiAgent
## -- 2021-06-15  1.1.2     SY       Bugfixing in class ActionElement
## -- 2021-06-25  1.2.0     DA       New class RLDataStoring (based on new basic class DataStoring);
## --                                Extension of classes Scenario and Training by data logging; 
## --                                New method Environment.get_reward_type();
## --                                New methods Agent.get_name(), Agent.set_name()
## --                                New method Training.save_data()
## -- 2021-07-01  1.2.1     DA       Bugfixes in classes MultiAgent and Training
## -- 2021-07-06  1.2.2     SY       Update method Training.save_data()
## -- 2021-08-26  1.3.0     DA       New classes: EnvBase, EnvModel, SARBuffer, SARBufferelement, 
## --                                Policy, HPTuningRL
## --                                Enhancements on Agent class;
## --                                Class WrapperGym renamed to WrEnvGym;
## --                                Incompatible changes on classes Agent, MultiAgent, Scenario
## -- 2021-08-27  1.3.1     SY       Move classes WrEnvPZoo, WrPolicyRAY to a new module wrapper
## -- 2021-08-28  1.3.2     DA       Bugfixes and minor improvements
## -- 2021-09-11  1.3.2     MRD      Change Header information to match our new library name
## -- 2021-09-18  1.3.3     MRD      Implement new SARBuffer class and SARBufferelement. The buffer
## --                                is now working in dictionary. Now the SARBuffer is inside
## --                                the Policy and EnvModel instead of the Agent.
## --                                Added "Done" as default input for Agent.adapt()
## -- 2021-09-19  1.3.3     MRD      Change SARBuffer Class and Inherits SARBufferElement with base
## --                                class Buffer
## -- 2021-09-22  1.3.4     WB       Added PrioritizedBuffer Class and PrioritizedBufferElement,
## --                                including the required SegmentTree data structure
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.4 (2021-09-22)

This module provides model classes for reinforcement learning tasks.
"""


import numpy as np
from typing import List, Callable
from time import sleep
import random
import operator
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.ml import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class State(Element, TStamp):
    """
    Objects of this class represent states of a given metric state space.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_state_space:MSpace):
        TStamp.__init__(self)
        Element.__init__(self, p_state_space)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ActionElement(Element):

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_action_space:Set, p_weight=1.0) -> None:
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
    of the internal list is related to an agent and it's partial subaction.
    Action values for the first agent can be added while object instantiation.
    Action values of further agents can be added by using method self.add_elem().
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_agent_id=0, p_action_space=None, p_values:np.ndarray=None):
        """
        Parameters:
            p_agent_id        Unique id of (first) agent to be added
            p_action_space    Action space of (first) agent to be added
            p_values          Action values of (first) agent to be added
        """

        ElementList.__init__(self)
        TStamp.__init__(self)

        if ( p_action_space is not None ) and ( p_values is not None):
            e = ActionElement(p_action_space)
            e.set_values(p_values)
            self.add_elem(p_agent_id, e)


## -------------------------------------------------------------------------------------------------
    def get_agent_ids(self):
        return self.get_elem_ids()


## -------------------------------------------------------------------------------------------------
    def get_sorted_values(self) -> np.ndarray:
        # 1 Determine overall dimensionality of action vector
        num_dim     = 0
        action_ids  = []

        for elem in self._elem_list:
            num_dim = num_dim + elem.get_related_set().get_num_dim()
            action_ids.extend(elem.get_related_set().get_dim_ids())

        action_ids.sort()

        # 2 Transfer action values
        action = np.zeros(num_dim)

        for elem in self._elem_list:
            for elem_action_id in elem.get_related_set().get_dim_ids():
                i         = action_ids.index(elem_action_id)
                action[i] = elem.get_value(elem_action_id)

        # 3 Return sorted result array
        return action





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Reward(TStamp):
    """
    Objects of this class represent rewards of environments. The internal structure
    depends of the reward type. Three types are supported as listed below.
    """

    C_TYPE_OVERALL        = 0    # Reward is a scalar (default)
    C_TYPE_EVERY_AGENT    = 1    # Reward is a scalar for every agent
    C_TYPE_EVERY_ACTION   = 2    # Reward is a scalar for every agent and action

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_type=C_TYPE_OVERALL, p_value=0):
        """
        Parameters:
            p_type          Reward type (default: C_TYPE_OVERALL)
            p_value         Overall reward value (reward type C_TYPE_OVERALL only)
        """

        TStamp.__init__(self)
        self.type           = p_type
        self.agent_ids      = []
        self.rewards        = []
        if self.type == self.C_TYPE_OVERALL: self.set_overall_reward(p_value)


## -------------------------------------------------------------------------------------------------
    def get_type(self):
        return self.type        


## -------------------------------------------------------------------------------------------------
    def is_rewarded(self, p_agent_id) -> bool:
        try:
            i = self.agent_ids.index(p_agent_id)
            return True
        except:
            return False


## -------------------------------------------------------------------------------------------------
    def set_overall_reward(self, p_reward) -> bool:
        if self.type != self.C_TYPE_OVERALL: return False
        self.overall_reward = p_reward    
        return True


## -------------------------------------------------------------------------------------------------
    def get_overall_reward(self):
        return self.overall_reward


## -------------------------------------------------------------------------------------------------
    def add_agent_reward(self, p_agent_id, p_reward) -> bool:
        if self.type != self.C_TYPE_EVERY_AGENT: return False
        self.agent_ids.append(p_agent_id) 
        self.rewards.append(p_reward)   
        return True


## -------------------------------------------------------------------------------------------------
    def get_agent_reward(self, p_agent_id):
        if self.type == self.C_TYPE_OVERALL: return self.overall_reward

        try:
            i = self.agent_ids.index(p_agent_id)    
        except:
            return None   

        return self.rewards[i]


## -------------------------------------------------------------------------------------------------
    def add_action_reward(self, p_agent_id, p_action_id, p_reward) -> bool:
        if self.type != self.C_TYPE_EVERY_ACTION: return False

        try:
            i = self.agent_ids.index(p_agent_id)
            r = self.rewards[i]
            r[0].append(p_action_id)
            r[1].append(p_reward)
        except:
            self.agent_ids.append(p_agent_id)
            self.rewards.append([ [p_action_id],[p_reward] ])

        return True


## -------------------------------------------------------------------------------------------------
    def get_action_reward(self, p_agent_id, p_action_id):
      if self.type != self.C_TYPE_EVERY_ACTION: return None
      
      try:
          i_agent = self.agent_ids.index(p_agent_id)
      except:
          return None

      try:
          r = self.rewards[i_agent]
          i_action = r[0].index(p_action_id)
          return r[1][i_action]
      except:
          return None





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvBase(Log, Plottable):
    """
    Base class for all environment classes. It defines the interface and elementry properties for
    an environment in the context of reinforcement learning.
    """

    C_TYPE          = 'Environment Base'
    C_NAME          = '????'

    C_LATENCY       = timedelta(0,1,0)              # Default latency 1s

    C_REWARD_TYPE   = Reward.C_TYPE_OVERALL         # Default reward type for reinforcement learning

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_latency:timedelta=None, p_logging=True):
        """
        Parameters:
            p_latency           Optional: latency of environment. If not provided
                                internal value C_LATENCY will be used by default
            p_logging           Boolean switch for logging
        """

        Log.__init__(self, p_logging=p_logging)
        self._state_space      = ESpace()
        self._action_space     = ESpace()
        self._state            = None
        self._last_action      = None
        self.goal_achievement  = 0.0
        self.broken            = False
        self.done              = False
        self.set_latency(p_latency)


## -------------------------------------------------------------------------------------------------
    def get_state_space(self):
        return self._state_space


## -------------------------------------------------------------------------------------------------
    def get_action_space(self):
        return self._action_space


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        """
        Returns latency of environment.
        """

        return self.latency


## -------------------------------------------------------------------------------------------------
    def set_latency(self, p_latency:timedelta=None) -> None:
        """
        Sets latency of environment. If p_latency is None latency will be reset
        to internal value of attribute C_LATENCY.

        Parameters:
          p_latency       New latency 
        """

        if p_latency is None:
            self.latency = self.C_LATENCY
        else:
            self.latency = p_latency


## -------------------------------------------------------------------------------------------------
    def get_reward_type(self):
        return self.C_REWARD_TYPE


## -------------------------------------------------------------------------------------------------
    def get_state(self) -> State:
        """
        Returns current state of environment.
        """

        return self.state


## -------------------------------------------------------------------------------------------------
    def _set_state(self, p_state:State) -> None:
        """
        Explicitely sets the current state of the environment. Internal use only.
        """

        self.state = p_state


## -------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """
        Resets environment to initial state. Please redefine.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action:Action) -> bool:
        """
        Processes given action and updates the state of the environment.

        Parameters:
            p_action      Action to be processed

        Returns:
            True, if action processing was successfull. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def compute_reward(self) -> Reward:
        """
        Computes a reward. Please redefine.

        Returns:
          Reward object
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Environment(EnvBase):
    """
    This class represents the central environment model to be reused/inherited in own rl projects.
    """

    C_TYPE          = 'Environment'
 
    C_MODE_SIM      = 0
    C_MODE_REAL     = 1

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mode=C_MODE_SIM, p_latency:timedelta=None, p_logging=True):
        """
        Parameters:
            p_mode              Mode of environment (simulation/real)
            p_latency           Optional: latency of environment. If not provided
                                internal value C_LATENCY will be used by default
            p_logging           Boolean switch for logging
        """

        super().__init__(p_latency=p_latency, p_logging=p_logging)
        self._setup_spaces()
        self.set_mode(p_mode)


## -------------------------------------------------------------------------------------------------
    def get_mode(self):
        return self._mode


## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):
        if p_mode == self.C_MODE_SIM:
            self.log(self.C_LOG_TYPE_I, 'Switched to mode SIMULATION')
        elif p_mode == self.C_MODE_REAL:
            self.log(self.C_LOG_TYPE_I, 'Switched to mode REAL')
        else:
            self.log(self.C_LOG_TYPE_E, 'Wrong mode', p_mode)
            self.set_mode(self.C_MODE_SIM)
            return

        self._mode = p_mode


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action:Action) -> bool:
        """
        Processes given action and updates the state of the environment.

        Parameters:
            p_action      Action to be processed

        Returns:
            True, if action processing was successfull. False otherwise.
        """

        # 0 Some initial stuff
        self.last_action = p_action
        self.log(self.C_LOG_TYPE_I, 'Start processing action')
        for agent in p_action.get_elem_ids():
            self.log(self.C_LOG_TYPE_I, 'Actions of agent', agent, '=', p_action.get_elem(agent).get_values())

        # 1 State transition
        if self._mode == self.C_MODE_SIM:
            # 1.1 Simulated state transition
            self._simulate_reaction(p_action)

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
        self._evaluate_state()

        self.log(self.C_LOG_TYPE_I, 'Action processing finished successfully')
        return True


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):
        """
        Implement this method to enrich the state and action space with specific 
        dimensions. 
        """

        # 1 Setup state space
        # self.state_space.add_dim(Dimension(0, 'Pos', 'Position', '', 'm', 'm', [-50,50]))
        # self.state_space.add_dim(Dimension(1, 'Vel', 'Velocity', '', 'm/sec', '\frac{m}{sec}', [-50,50]))

        # 2 Setup action space
        # self.action_space.add_dim(Dimension(0, 'Rot', 'Rotation', '', '1/sec', '\frac{1}{sec}', [-50,50]))

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_action:Action) -> None:
        """
        Mode C_MODE_SIM only: simulates a state transition of the environment 
        based on a new action. Method to be redefined. Please use method 
        set_state() for internal update.

        Parameters:
            p_action      Action to be processed
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _export_action(self, p_action:Action) -> bool:
        """
        Mode C_MODE_REAL only: exports given action to be processed externally 
        (for instance by a real hardware). Please redefine. 

        Parameters:
            p_action      Action to be exported

        Returns:
            True, if action export was successful. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _import_state(self) -> bool:
        """
        Mode C_MODE_REAL only: imports state from an external system (for instance a real hardware). 
        Please redefine. Please use method set_state() for internal update.

        Returns:
          True, if state import was successful. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self) -> None:
        """
        Updates the goal achievement value in [0,1] and the flags done and broken based on the 
        current state. Please redefine.
        """

        self.goal_achievement = 0.0
        self.done             = False
        self.broken           = False

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SARBufferElement(BufferElement):
    """
    Element of a State-Action-Reward-Buffer.
    """
    pass


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SARBuffer(Buffer):
    """
    State-Action-Reward-Buffer in dictionary.
    """
    pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PrioritizedBufferElement(SARBufferElement):
    """
    Element of a State-Action-Reward-Buffer.
    """
    pass


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PrioritizedBuffer(SARBuffer):
    """
    Prioritized Sampling State-Action-Reward-Buffer in dictionary.
    Reference: 
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    """
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_size=1, alpha: float=0.3, beta: float=1):
    
        """
        Parameters:
            p_size (int, optional): Buffer size. Defaults to 1.
            alpha (float, optional): Prioritization level. Defaults to 0.3
            beta (float, optional): Prioritization Control. Defaults to 1. Should be increased gradualy to 1 by the end of training.
        """
        assert alpha >= 0
        assert beta >= 0
        super().__init__(p_size=p_size)
        
        self.alpha = alpha
        self.beta = beta
        
        tree_capacity = 1
        while tree_capacity < self._size:
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0
    
    
## -------------------------------------------------------------------------------------------------    
    def add_element(self, p_elem:PrioritizedBufferElement)
        """
        Add element to the buffer.

        Parameters:
            p_elem (BufferElement): Element of Buffer
        """
        super().add_element(p_elem)
        idx = len(self._data_buffer)-1
        self.sum_tree[idx] = self.max_priority**self.alpha
        self.min_tree[idx] = self.max_priority**self.alpha
    
    
## -------------------------------------------------------------------------------------------------
    def _gen_sample_ind(self, p_num:int) -> list:
        """
        Generate random indices from the buffer.

        Parameters:
            p_num (int): Number of sample

        Returns:
            List of incides
        """
        buffer_length = len(self._data_buffer)
        p_sum = self.sum_tree.sum(0, buffer_length-1)
        p_list_idx = []
        segment = p_sum / buffer_length
        for i in range(p_num):
            a = segment*i
            b = segment*(i+1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            p_list_idx.append(idx)
        return p_list_idx
        

## -------------------------------------------------------------------------------------------------
    def _extract_rows(self, p_list_idx:list):
        """
        Extract the element in the buffer based on a
        list of indices.

        Parameters:
            p_list_idx (list): List of indices

        Returns:
            Samples in dictionary
        """
        super()._extract_rows(p_list_idx)
        weights = []
        buffer_length = len(self._data_buffer)
        
        p_min = self.min_tree.min()/self.sum_tree.sum()
        max_weight = (p_min*buffer_length)**(-self.beta)
        p_sample = self.sum_tree[p_list_idx]/self.sum_tree.sum()
        weights = ((p_sample*buffer_length)**(-self.beta))/max_weight
        
        rows['weights'] = weights
        rows['p_list_idx'] = p_list_idx
        
        return rows


## -------------------------------------------------------------------------------------------------
    def get_latest(self):
        """
        Returns latest buffered element. 
        """
        try:
            return self._extract_rows([len(self._data_buffer)-1])
        except:
            return None
        
        
## -------------------------------------------------------------------------------------------------
    def get_all(self):
        """
        Return all buffered elements.
        """
        p_list_idx = [i for i in range(len(self._data_buffer))]
        return self._extract_rows(p_list_idx)


## -------------------------------------------------------------------------------------------------
    def update_priorities(self, p_list_idx:list, priorities:np.ndarray)
        """
        Updates the priority tree.
        Needs to be called during each training step, utilising the element-wise calculated loss.
        """
        assert len(p_list_idx) == len(priorities)
        assert np.min(priorities) > 0 
        assert min(p_list_idx) >= 0
        assert max(p_list_idx) <= len(self._data_buffer)
        
        self.sum_tree[p_list_idx] = priorities**self.alpha
        self.min_tree[p_list_idx] = priorities**self.alpha
        
        self.max_priority = max(self.max_priority, np.max(priorities))


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SegmentTree:
    """ 
    Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    Attributes:
        capacity (int)
        tree (list)
        operation (function)
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, capacity: int, operation: Callable, init_value: float):
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation
        
        
## -------------------------------------------------------------------------------------------------
    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )


## -------------------------------------------------------------------------------------------------
    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying 'self.operation'."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)


## -------------------------------------------------------------------------------------------------
    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2


## -------------------------------------------------------------------------------------------------
    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SumSegmentTree(SegmentTree):
    """ 
    Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, capacity: int):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )


## -------------------------------------------------------------------------------------------------
    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)


## -------------------------------------------------------------------------------------------------
    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MinSegmentTree(SegmentTree):
    """ 
    Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )


## -------------------------------------------------------------------------------------------------
    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvModel(EnvBase, Adaptive):
    """
    Template class for a Real world model to be used for model based agents.
    """

    C_TYPE      = 'EnvModel'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_buffer_size=1, p_buffer_cls=SARBuffer, p_ada=True, p_logging=True):
        EnvBase.__init__(self, p_logging=p_logging)
        Adaptive.__init__(self, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Policy(Adaptive, Plottable):
    """
    This class represents the policy of a single-agent. It is adaptive and can be trained with
    State-Action-Reward (SAR) data that will be expected as a SAR buffer object. 
    The three main learning paradigms of machine learning to train a policy are supported:

    a) Training by Supervised Learning
    The entire SAR data set inside the SAR buffer shall be adapted.

    b) Training by Reinforcement Learning
    The latest SAR data record inside the SAR buffer shall be adapted.

    c) Training by Unsupervised Learning
    All state data inside the SAR buffer shall be adapted.

    Furthermore a policy class can compute actions from states.

    Hyperparameters of the policy should be stored in the internal object self._hp_list, so that
    they can be tuned from outside. Optionally a policy-specific callback method can be called on 
    changes. For more information see class HyperParameterList.
    """

    C_TYPE          = 'Policy'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_state_space:MSpace, p_action_space:MSpace, p_buffer_size=1, p_buffer_cls=SARBuffer, p_ada=True, p_logging=True):
        """
         Parameters:
            p_state_space       State space object
            p_action_space      Action space object
            p_buffer_size       Size of the buffer
            p_buffer_cls        Buffer class to be used
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        super().__init__(p_buffer=p_buffer_cls(p_buffer_size), p_ada=p_ada, p_logging=p_logging)
        self._state_space   = p_state_space
        self._action_space  = p_action_space
        self.set_id(0)


## -------------------------------------------------------------------------------------------------
    def get_state_space(self) -> MSpace:
        return self._state_space


## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return self._action_space


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id):
        self._id = p_id


## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Adapts the policy based on State-Action-Reward (SAR) data that will be expected as a SAR
        buffer object. Please call super-method at the beginning of your own implementation and
        adapt only if it returns True.

        Parameters:
            p_arg[0]            SAR Buffer object
        """

        if not super().adapt(*p_args): return False
        return True


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state:State) -> Action:
        """
        Specific action computation method to be redefined. 

        Parameters:
            p_state       State of environment

        Returns:
            Action object
        """

        raise NotImplementedError

## -------------------------------------------------------------------------------------------------
    def add_buffer(self, p_buffer_element: SARBufferElement):
        """
        Intended to save the data to the buffer. By default it save the SARS data.
        
        """
        buffer_element = self._add_additional_buffer(p_buffer_element)
        self._buffer.add_element(buffer_element)

## -------------------------------------------------------------------------------------------------
    def _add_additional_buffer(self, p_buffer_element:SARBufferElement):
        """
        Intended to add additional buffer element to the buffer other than SARS data. Redefine this method
        to add additional buffer element
        """
        
        return p_buffer_element


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        """
        Intended to clear internal temporary attributes, buffers, ... Can be used while training
        to prepare the next episode.
        """

        pass






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Agent(Policy):
    """
    This class represents a single agent model.
    """

    C_TYPE          = 'Agent'
    C_NAME          = 'Standard'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_policy:Policy, p_envmodel:EnvModel=None, p_name='', p_id=0, p_ada=True, p_logging=True):
        """
        Parameters:
            p_policy            Policy object
            p_name              Optional name of agent
            p_id                Unique agent id (especially important for multi-agent scenarios)
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        if p_name != '': 
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)

        self._state             = None
        self._previous_state    = None
        self._previous_action   = None
        self._policy            = p_policy
        self._action_space      = self._policy.get_action_space()
        self._state_space       = self._policy.get_state_space()
        self._envmodel          = p_envmodel

        self._set_id(p_id)
        self.switch_logging(p_logging)
        self.switch_adaptivity(p_ada)
        self.clear_buffer()


## -------------------------------------------------------------------------------------------------
    def _set_id(self, p_id): 
        super().set_id(p_id)
        self._policy.set_id(p_id)


## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id):
        """
        The unique agent id will be defined while instantiation and can't be changed any more.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        return self._name


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name):
        self._name   = p_name
        self.C_NAME  = p_name


## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Default adaption implementation of a single agent.

        Parameters:
            p_args[0]       Reward object (see class Reward)
            p_args[1]       Done Flag

        Returns:
            True, if something has beed adapted
        """

        # 1 Check: Adaption possible?
        if self._adaptivity == False:
            self.log(self.C_LOG_TYPE_I, 'Adaption disabled')
            return False

        if self._previous_state is None:
            self.log(self.C_LOG_TYPE_I, 'Adaption: previous state None -> adaptivity skipped')
            return False

        reward = p_args[0]
        done = p_args[1]
        self.log(self.C_LOG_TYPE_I, 'Adaption: previous state =', self._previous_state.get_values(), '; reward = ', p_args[0].get_agent_reward(self._id))

        # 2 Add data to SAR buffer
        self._policy.add_buffer(SARBufferElement(dict(previous_state=self._previous_state, action=self._previous_action, state=self._state, reward=reward, done=done)))

        # 3 Adapt policy
        return self._policy.adapt()


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        """
        Default implementation of a single agent.
        """

        self.log(self.C_LOG_TYPE_I, 'Action computation: state received = ', p_state.get_values())
        self._previous_state  = self._state
        self._state           = p_state
        self._previous_action = self._policy.compute_action(p_state)
        return self._previous_action


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._policy.clear_buffer()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiAgent(Agent):
    """
    This class implements a reinforcement learning multi-agent model.
    """

    C_TYPE          = 'Multi-Agent'
    C_NAME          = ''
    C_SUFFIX        = '.cfg'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name='', p_ada=True, p_logging=True):
        """
        Parameters:
            p_name              Name of agent
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        self._agents = []
        self.set_name(p_name)
        self.switch_logging(p_logging=p_logging)
        self.switch_adaptivity(p_ada=p_ada)
        

## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging=True) -> None: 
        super().switch_logging(p_logging=p_logging)

        for agent_entry in self._agents:
            agent_entry[0].switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def get_filename(self) -> str:
        return self.C_TYPE + ' ' + self.C_NAME + self.C_SUFFIX


## -------------------------------------------------------------------------------------------------
    def load(self, p_path, p_filename=None) -> bool:
        # load all subagents
        for i, agent_entry in enumerate(self._agents):
            agent       = agent_entry[0]
            agent_name  = agent.C_TYPE + ' ' + agent.C_NAME + '(' + str(i) + ')'

            if agent.load(p_path, agent_name + agent.C_SUFFIX):
                self.log(Log.C_LOG_TYPE_I, agent_name + ' loaded')
            else:
                self.log(Log.C_LOG_TYPE_E, agent_name + ' not loaded')
                return False

        return True


## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename=None) -> bool:
        # save all subagents
        for i, agent_entry in enumerate(self._agents):
            agent       = agent_entry[0]
            agent_name  = agent.C_TYPE + ' ' + agent.C_NAME + '(' + str(i) + ')'

            if agent.save(p_path, agent_name + agent.C_SUFFIX):
                self.log(Log.C_LOG_TYPE_I, agent_name + ' saved')
            else:
                self.log(Log.C_LOG_TYPE_E, agent_name + ' not saved')
                return False

        return True


## -------------------------------------------------------------------------------------------------
    def add_agent(self, p_agent:Agent, p_weight=1.0) -> None:
        """
        Adds agent object to internal list of agents. 

        Parameters:
            p_agent           Agent object
            p_weight          Optional weight for the agent

        Returns:
            Nothing
        """

        p_agent.switch_adaptivity(self._adaptivity)
        self._agents.append([p_agent, p_weight])
        p_agent.set_name(str(p_agent.get_id()) + ' ' + p_agent.get_name())
        self.log(Log.C_LOG_TYPE_I, p_agent.C_TYPE + ' ' + p_agent.get_name() + ' added.')


## -------------------------------------------------------------------------------------------------
    def get_agents(self):
        return self._agents


## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        if not self._adaptivity:
            self.log(self.C_LOG_TYPE_I, 'Adaption disabled')
            return False

        reward = p_args[0]
        done = p_args[1]

        self.log(self.C_LOG_TYPE_I, 'Start of adaption for all agents...')      

        adapted = False
        for agent_entry in self._agents:
            agent = agent_entry[0]
            if ( reward.get_type() != Reward.C_TYPE_OVERALL ) and not reward.is_rewarded(agent.get_id()): continue
            self.log(self.C_LOG_TYPE_I, 'Start adaption for agent', agent.get_id())
            adapted = adapted or agent.adapt(reward, done)

        self.log(self.C_LOG_TYPE_I, 'End of adaption for all agents...')        

        return adapted


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state:State) -> Action:
        self.log(self.C_LOG_TYPE_I, 'Start of action computation for all agents...')      

        action = Action()

        for agent, weight in self._agents:
            state_agent     = State(agent.get_state_space())
            state_ids       = agent.get_state_space().get_dim_ids()

            for state_id in state_ids:
                state_agent.set_value(state_id, p_state.get_value(state_id))

            action_agent    = agent.compute_action(state_agent)

            action_element  = action_agent.get_elem(agent.get_id())
            action_element.set_weight(weight)
            action.add_elem(agent.get_id(), action_element)

        self.log(self.C_LOG_TYPE_I, 'End of action computation for all agents...')  
        return action      


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        for agent_entry in self._agents:
            agent_entry[0].clear_buffer()


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        """
        Doesn't support embedded plot of underlying agent hierarchy.
        """

        self.log(self.C_LOG_TYPE_I, 'Init vizualization for all agents...')      

        for agent_entry in self._agents: agent_entry[0].init_plot(None)


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        self.log(self.C_LOG_TYPE_I, 'Start vizualization for all agents...')      

        for agent_entry in self._agents: agent_entry[0].update_plot()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLDataStoring(DataStoring):
    """
    Derivate of basic class DataStoring that is specialized to store episodical training data in the
    context of reinforcement learning.
    """

    # Frame ID renamed
    C_VAR0              = 'Episode ID'

    # Variables for training header data storage
    C_VAR_NUM_CYLCLES   = 'Number of cycles'
    C_VAR_ENV_DONE      = 'Goal reached'
    C_VAR_ENV_BROKEN    = 'Env broken'

    # Variables for episodical detail data storage
    C_VAR_CYCLE         = 'Cycle'
    C_VAR_DAY           = 'Day'
    C_VAR_SEC           = 'Second'
    C_VAR_MICROSEC      = 'Microsecond'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_space:Set=None):
        """
        Parameters:
            p_space         Space object that provides dimensional information for raw data. If None
                            a training header data object will be instantiated.
        """
        
        self.space = p_space

        if self.space is None:
            # Initialization as a training header data storage
            self.variables  = [ self.C_VAR_NUM_CYLCLES, self.C_VAR_ENV_DONE, self.C_VAR_ENV_BROKEN ]

        else:
            # Initalization as an episodical detail data storage
            self.variables  = [ self.C_VAR_CYCLE, self.C_VAR_DAY, self.C_VAR_SEC, self.C_VAR_MICROSEC ]
            self.var_space  = []
    
            for dim_id in self.space.get_dim_ids():
                dim = self.space.get_dim(dim_id)
                self.var_space.append(dim.get_name_short())

            self.variables.extend(self.var_space)

        super().__init__(self.variables)


## -------------------------------------------------------------------------------------------------
    def get_variables(self):
        return self.variables


## -------------------------------------------------------------------------------------------------
    def get_space(self):
        return self.space


## -------------------------------------------------------------------------------------------------
    def add_episode(self, p_episode_id):
        self.add_frame(p_episode_id)
        self.current_episode = p_episode_id


## -------------------------------------------------------------------------------------------------
    def memorize_row(self, p_cycle_id, p_tstamp:timedelta, p_data):
        """
        Memorizes an episodical data row.

        Parameters: 
            p_cycle_id          Cycle id
            p_tstamp            Time stamp
            p_data              Data that meet the dimensionality of the related space
        """

        self.memorize(self.C_VAR_CYCLE, self.current_episode, p_cycle_id)
        self.memorize(self.C_VAR_DAY, self.current_episode, p_tstamp.days)
        self.memorize(self.C_VAR_SEC, self.current_episode, p_tstamp.seconds)
        self.memorize(self.C_VAR_MICROSEC, self.current_episode, p_tstamp.microseconds)

        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_episode, p_data[i])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Scenario(Log, LoadSave):
    """
    Template class for an rl sceario consisting of an environment and an agent. Please
    implement method setup() to setup env and agent structure.
    """

    C_TYPE              = 'RL-Scenario'
    C_NAME              = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mode=Environment.C_MODE_SIM, p_ada=True, p_cycle_len:timedelta=None, p_cycle_limit=0, p_visualize=True, p_logging=True):
        """
        Parameters:
            p_mode              Operation mode of environment (see Environment.C_MODE_*)
            p_ada               Boolean switch for adaptivity of agent
            p_cycle_len         Fixed cycle duration (optional)
            p_cycle_limit       Maximum number of cycles (0=no limit)
            p_visualize         Boolean switch for env/agent visualisation
            p_logging           Boolean switch for logging functionality
        """

        # 0 Intro
        self._env           = None
        self._agent         = None
        self._cycle_len     = p_cycle_len
        self._cycle_limit   = p_cycle_limit
        self._visualize     = p_visualize
        Log.__init__(self, p_logging=p_logging)

        # 1 Setup entire scenario
        self._setup(p_mode, p_ada, p_logging)

        # 2 Init timer
        if self._env.get_mode() == Environment.C_MODE_SIM:
            t_mode = Timer.C_MODE_VIRTUAL
        else:
            t_mode = Timer.C_MODE_REAL

        if self._cycle_len is not None:
            t_lap_duration = p_cycle_len
        else:
            t_lap_duration = self._env.get_latency()

        self._timer  = Timer(t_mode, t_lap_duration, self._cycle_limit)


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_logging:bool):
        """
        Here's the place to explicitely setup the entire rl scenario. Please bind your env to
        self._env and your agent to self._agent. 

        Parameters:
            p_mode              Operation mode of environment (see Environment.C_MODE_*)
            p_ada               Boolean switch for adaptivity of agent
            p_logging           Boolean switch for logging functionality
       """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        Environment and timer will be resetted. Agent's internal buffer data will be cleared but
        it's policy will not be touched.
        """

        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Scenario reset...')
        self._env.reset()
        self._agent.clear_buffer()

        if self._visualize:
            self._env.init_plot()
            self._agent.init_plot()

        self._timer.reset()
        self._env.get_state().set_tstamp(self._timer.get_time())



## -------------------------------------------------------------------------------------------------
    def get_env(self):
        return self._env


## -------------------------------------------------------------------------------------------------
    def get_agent(self):
        return self._agent
      

## -------------------------------------------------------------------------------------------------
    def run_cycle(self, p_cycle_id, p_ds_states:RLDataStoring=None, p_ds_actions:RLDataStoring=None, p_ds_rewards:RLDataStoring=None):
        """
        Processes a single control cycle with optional data logging.

        Parameters:
            p_cycle_id          Cycle id
            p_ds_states         Optional external data storing object that collects environment state data
            p_ds_actions        Optional external data storing object that collects agent action data
            p_ds_rewards        Optional external data storing object that collects environment reeward data
        """

        # 0 Cycle intro
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Start of cycle', str(p_cycle_id))


        # 1 Environment: get and log current state
        state   = self._env.get_state()
        if p_ds_states is not None:
            p_ds_states.memorize_row(p_cycle_id, self._timer.get_time(), state.get_values())


        # 2 Agent: compute and log next action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent computes action...')
        action  = self._agent.compute_action(state)
        ts      = self._timer.get_time()
        action.set_tstamp(ts)
        if p_ds_actions is not None:
            p_ds_actions.memorize_row(p_cycle_id, ts, action.get_sorted_values())


        # 3 Environment: process agent's action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Env processes action...')
        self._env.process_action(action)
        self._timer.add_time(self._env.get_latency())     # in virtual mode only...
        self._env.get_state().set_tstamp(self._timer.get_time())


        # 4 Environment: compute and log reward
        reward  = self._env.compute_reward()
        ts      = self._timer.get_time()
        reward.set_tstamp(ts)
        if p_ds_rewards is not None:
            if ( reward.get_type() == Reward.C_TYPE_OVERALL ) or ( reward.get_type() == Reward.C_TYPE_EVERY_AGENT ):
                reward_values = np.zeros(p_ds_rewards.get_space().get_num_dim())

                for i, agent_id in enumerate(p_ds_rewards.get_space().get_dim_ids()): 
                    reward_values[i] = reward.get_agent_reward(agent_id)
                
                p_ds_rewards.memorize_row(p_cycle_id, ts, reward_values)


        # 5 Agent: adapt policy
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent adapts policy...')
        self._agent.adapt(reward, self._env.done)


        # 6 Optional visualization
        if self._visualize:
            self._env.update_plot()
            self._agent.update_plot()


        # 7 Wait for next cycle (virtual mode only)
        if ( self._timer.finish_lap() == False ) and ( self.cycle_len is not None ):
            self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Process timed out !!!')


        # 8 Cycle outro
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': End of cycle', str(p_cycle_id), '\n')


## -------------------------------------------------------------------------------------------------
    def run(self, p_exit_when_broken=True, p_exit_when_done=True, p_ds_states:RLDataStoring=None, p_ds_actions:RLDataStoring=None, p_ds_rewards:RLDataStoring=None):
        """
        Processes control cycles in a loop. Termination depends on parameters.

        Parameters:
            p_exit_when_broken      If True, loop terminates when environment has boken
            p_exit_when_done        If True, loop terminates when goal of environment was achieved
            p_ds_states             Optional external data storing object that collects environment state data
            p_ds_actions            Optional external data storing object that collects agent action data
            p_ds_rewards            Optional external data storing object that collects environment reeward data

        Returns:
            done                    True if environment reached it's goal
            broken                  True if environment has broken
            num_cycles              Number of cycles
        """
        
        # 1 Preparation
        done = False
        self.reset()


        # 2 Start run
        self.log(self.C_LOG_TYPE_I, 'Run started')
        cycle_id  = 1

        while True:
            # 2.1 Process one cycle
            self.run_cycle(cycle_id, p_ds_states=p_ds_states, p_ds_actions=p_ds_actions, p_ds_rewards=p_ds_rewards)

            # 2.2 Check and handle environment's health
            if self._env.broken: 
                self.log(self.C_LOG_TYPE_E, 'Environment broken!')
                if p_exit_when_broken: 
                    break
                else:
                    self.log(self.C_LOG_TYPE_I, 'Reset environment')
                    self._env.reset()

            # 2.3 Check and handle environment's done state
            if self._env.done != done:
                done = self._env.done
                if done == True:
                    self.log(self.C_LOG_TYPE_I, 'Environment goal achieved')
                else:
                    self.log(self.C_LOG_TYPE_W, 'Environment goal missed')

            if p_exit_when_done and done: break

            # 2.4 Next cycle id
            if self._cycle_limit > 0: 
                if cycle_id < self._cycle_limit: 
                    cycle_id = cycle_id + 1
                else:
                    break


        # 3 Finish run
        self.log(self.C_LOG_TYPE_I, 'Stop')
        return self._env.done, self._env.broken, cycle_id





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Training(Log):
    """
    This class performs an episodical training on a (multi-)agent in a given environment. Both are 
    expected as parts of a reinforcement learning process (see class Process for more details).
    The class optionally collects all relevant data like environmenal states and rewards or agents
    actions. Furthermore overarching training data will be collected.

    The class provides the three methods run(), run_episode(), run_cycle() that can be called in 
    any order to proceed the training.
    """

    C_TYPE                  = 'Training'
    C_NAME                  = 'RL'

    C_FNAME_TRAINING        = 'training'
    C_FNAME_ENV_STATES      = 'env_states'
    C_FNAME_AGENT_ACTIONS   = 'agent_actions'
    C_FNAME_ENV_REWARDS     = 'env_rewards'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scenario:Scenario, p_episode_limit=50, p_cycle_limit=0, p_collect_states=True, p_collect_actions=True, p_collect_rewards=True, p_collect_training=True, p_logging=True):
        """
        Parmeters:
            p_scenario              RL scenario object
            p_episode_limit         Maximum number of episodes
            p_cycle_limit           Naximum number of cycles within an episode (0 means: no limit)
            p_collect_states        If True, the environment states will be collected
            p_collect_actions       If True, the agent actions will be collected
            p_collect_rewards       If True, the environment reward will be collected
            p_collect_training      If True, global training data will be collected
            p_logging               Boolean switch for logging
        """

        super().__init__(p_logging=p_logging)

        self._scenario      = p_scenario
        self._env           = self._scenario.get_env()
        self._agent         = self._scenario.get_agent()

        self._episode_id    = 0
        self._episode_limit = p_episode_limit
        self._cycle_id      = 0
        self._cycle_limit   = p_cycle_limit

        if p_collect_states:
            self._ds_states   = RLDataStoring(self._env.get_state_space())
        else:
            self._ds_states   = None

        if p_collect_actions:
            self._ds_actions  = RLDataStoring(self._env.get_action_space())
        else:
            self._ds_actions  = None

        if p_collect_rewards:
            reward_type = self._env.get_reward_type()

            if ( reward_type == Reward.C_TYPE_OVERALL ) or ( reward_type == Reward.C_TYPE_EVERY_AGENT ):
                reward_space = Set()
                try:
                    agents = self._agent.get_agents()
                except:
                    agents = [ [self._agent, 1.0] ]

                for agent, weight in agents:
                    reward_space.add_dim(Dimension(agent.get_id(), agent.get_name()))

                if reward_space.get_num_dim() > 0:
                    self._ds_rewards  = RLDataStoring(reward_space)

            else:
                # Futher reward type not yet supported
                self._ds_rewards  = None

        else:
            self._ds_rewards  = None

        if p_collect_training:
            self._ds_training = RLDataStoring()
        else:
            self._ds_training = None


## -------------------------------------------------------------------------------------------------
    def run_cycle(self):
        """
        Runs next training cycle.
        """

        # 1 Begin of new episode? Reset agent and environment 
        if self._cycle_id == 0:
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self._episode_id, 'started...')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n')
            self._scenario.reset()
 
            # 1.1 Init frame for next episode in data storage objects
            if self._ds_training is not None: self._ds_training.add_episode(self._episode_id)
            if self._ds_states is not None: self._ds_states.add_episode(self._episode_id)
            if self._ds_actions is not None: self._ds_actions.add_episode(self._episode_id)
            if self._ds_rewards is not None: self._ds_rewards.add_episode(self._episode_id)


        # 2 Run a cycle
        self._scenario.run_cycle(self._cycle_id, p_ds_states=self._ds_states, p_ds_actions=self._ds_actions, p_ds_rewards=self._ds_rewards)


        # 3 Update training counters
        if self._env.done or self._env.broken or ( self._cycle_id == (self._cycle_limit-1) ):
            # 3.1 Episode finished
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self._episode_id, 'finished after', self._cycle_id + 1, 'cycles')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n\n')

            # 3.1.1 Update global training data storage
            if self._ds_training is not None:
                if self._env.done==True:
                    done_num = 1
                else:
                    done_num = 0

                if self._env.broken==True:
                    broken_num = 1
                else:
                    broken_num = 0

                self._ds_training.memorize(RLDataStoring.C_VAR_NUM_CYLCLES, self._episode_id, self._cycle_id + 1)
                self._ds_training.memorize(RLDataStoring.C_VAR_ENV_DONE, self._episode_id, done_num)
                self._ds_training.memorize(RLDataStoring.C_VAR_ENV_BROKEN, self._episode_id, broken_num)
 
            # 3.1.2 Prepare next episode
            self._episode_id   += 1
            self._cycle_id      = 0

        else:
            # 3.2 Prepare next cycle
            self._cycle_id     += 1


## -------------------------------------------------------------------------------------------------
    def run_episode(self):
        """
        Runs/finishes current training episode.
        """

        current_episode_id = self._episode_id
        while self._episode_id == current_episode_id: self.run_cycle()


## -------------------------------------------------------------------------------------------------
    def run(self):
        """
        Runs/finishes entire training.
        """

        while self._episode_id < self._episode_limit: self.run_episode()


## -------------------------------------------------------------------------------------------------
    def get_data(self):
        return self._ds_training, self._ds_states, self._ds_actions, self._ds_rewards


## -------------------------------------------------------------------------------------------------
    def save_data(self, p_path, p_delimiter):
        result      = True
        num_files   = 0

        if self._ds_training is not None:
            if self._ds_training.save_data(p_path, self.C_FNAME_TRAINING, p_delimiter):
                self.log(self.C_LOG_TYPE_I, 'Saved training data to file "' + self.C_FNAME_TRAINING + '" in "' + p_path + '"')
                num_files  += 1
                result      = result and True
            else:
                result      = False

        if self._ds_states is not None:
            if self._ds_states.save_data(p_path, self.C_FNAME_ENV_STATES, p_delimiter):
                self.log(self.C_LOG_TYPE_I, 'Saved environment state data to file "' + self.C_FNAME_ENV_STATES+ '" in "' + p_path + '"')
                num_files  += 1
                result      = result and True
            else:
                result      = False

        if self._ds_actions is not None:
            if self._ds_actions.save_data(p_path, self.C_FNAME_AGENT_ACTIONS, p_delimiter):
                self.log(self.C_LOG_TYPE_I, 'Saved agent action data to file "' + self.C_FNAME_AGENT_ACTIONS + '" in "' + p_path + '"')
                num_files  += 1
                result      = result and True
            else:
                result      = False

        if self._ds_rewards is not None:
            if self._ds_rewards.save_data(p_path, self.C_FNAME_ENV_REWARDS, p_delimiter):
                self.log(self.C_LOG_TYPE_I, 'Saved environment reward data to file "' + self.C_FNAME_ENV_REWARDS + '" in "' + p_path + '"')
                num_files  += 1
                result      = result and True
            else:
                result      = False

        if num_files > 0: return result
        return False






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HPTuningRL(HyperParamTuning):
    """
    Hyperparameter tuning for reinforcement learning.
    """

    C_NAME              = 'RL'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scenario:Scenario, p_path:str, p_episode_limit=50, p_cycle_limit=0, p_logging=True):
        super().__init__(p_path, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def optimize(self, *p_hp):
        # 1 Set hyperparameters
        # 2 Create and process a training
        # 3 Return overall number of cycles a the value to be mininmized

        return 0
