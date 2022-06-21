## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl
## -- Module  : models_env.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-18  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-06-25  1.0.1     DA       New method Environment.get_reward_type();
## -- 2021-08-26  1.1.0     DA       New classes: EnvBase, EnvModel, SARBuffer, SARBufferelement, 
## -- 2021-08-28  1.1.1     DA       Bugfixes and minor improvements
## -- 2021-09-11  1.1.2     MRD      Change Header information to match our new library name
## -- 2021-10-05  1.1.3     DA       Introduction of method Environment.get_cycle_limit()
## -- 2021-10-05  1.1.4     SY       Bugfixes and minor improvements
## -- 2021-10-25  1.1.5     SY       Enhancement of class EnvBase by adding ScientificObject.
## -- 2021-12-03  1.2.0     DA       Redesign:
## --                                - Introduction of special adaptive function classes AFct*
## --                                - Rework of classes EnvBase, Environment, EnvModel
## -- 2021-12-06  1.2.1     DA       Class AFctBase: correction by removing own method adapt()
## -- 2021-12-10  1.2.2     DA       Code optimization and bugfixes
## -- 2021-12-12  1.2.3     DA       New method EnvBase.get_last_reward()
## -- 2021-12-19  1.3.0     DA       Replaced term 'done' by 'success'
## -- 2021-12-21  1.4.0     DA       - Class EnvBase: 
## --                                    - new custom method _reset()
## --                                    - new custom method get_cycle_limit()
## --                                    - timeout detection
## --                                - Class EnvModel: cycle limit detection
## -- 2022-01-21  1.4.1     DA       Class EnvBase, method process_action(): a success/terminal state
## --                                avoids the timeout labelling
## -- 2022-02-28  1.4.2     SY       - Class EnvModel : redefine method _init_hyperparam()
## --                                - Refactoring due to auto generated ID in class Dimension
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.2 (2022-02-28)

This module provides model classes for environments and environment models.
"""

from mlpro.sl.models import AdaptiveFunction
from mlpro.rl.models_sar import *


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctBase(Model):
    """
    Base class for all special adaptive functions (state transition, reward, success, broken). 

    Parameters
    ----------
    p_afct_cls 
        Adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space of an environment or observation space of an agent
    p_action_space : MSpace
        Action space of an environment or agent
    p_input_space_cls
        Space class that is used for the generated input space of the embedded adaptive function (compatible to class
        MSpace)
    p_output_space_cls
        Space class that is used for the generated output space of the embedded adaptive function (compatible to class
        MSpace)
    p_output_elem_cls 
        Output element class (compatible to/inherited from class Element)
    p_threshold : float
        Threshold for the difference between a set point and a computed output. Computed outputs with
        a difference less than this threshold will be assessed as 'good' outputs. Default = 0.
    p_buffer_size : int
        Initial size of internal data buffer. Default = 0 (no buffering).
    p_ada : bool
        Boolean switch for adaptivity. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_par : Dict
        Further model specific parameters (to be specified in child class).

    Attributes
    ----------
    _state_space : MSpace
        State space
    _action_space : MSpace
        Action space
    _input_space : MSpace
        Input space of embedded adaptive function
    _output_space : MSpace
        Output space oof embedded adaptive function
    _afct : AdaptiveFunction
        Embedded adaptive function

    """

    C_TYPE = 'AFct Base'

    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_afct_cls,
                 p_state_space: MSpace,
                 p_action_space: MSpace,
                 p_input_space_cls=ESpace,
                 p_output_space_cls=ESpace,
                 p_output_elem_cls=Element,
                 p_threshold=0,
                 p_buffer_size=0,
                 p_ada=True,
                 p_logging=Log.C_LOG_ALL,
                 **p_par):

        self._state_space = p_state_space
        self._action_space = p_action_space
        self._input_space = p_input_space_cls()
        self._output_space = p_output_space_cls()

        self._setup_spaces(self._state_space, self._action_space, self._input_space, self._output_space)

        try:
            self._afct = p_afct_cls(p_input_space=self._input_space,
                                    p_output_space=self._output_space,
                                    p_output_elem_cls=p_output_elem_cls,
                                    p_threshold=p_threshold,
                                    p_buffer_size=p_buffer_size,
                                    p_ada=p_ada,
                                    p_logging=p_logging,
                                    **p_par)
        except:
            raise ParamError('Class ' + str(p_afct_cls) + ' is not compatible to class AdaptiveFunction')

        super().__init__(p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)

    ## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace, p_input_space: MSpace,
                      p_output_space: MSpace):
        """
        Custom method to set up the input and output space of the embedded adaptive function. Use the
        method add_dimension() of the empty spaces p_input_space and p_output_space to enrich them
        with suitable dimensions.

        Parameters
        ----------
        p_state_space : MSpace
            State space of an environment respectively observation space of an agent.
        p_action_space : MSpace
            Action space of an environment or agent.
        p_input_space : MSpace
            Empty input space of embedded adaptive function to be enriched with dimension.
        p_output_space : MSpace
            Empty output space of embedded adaptive function to be enriched with dimension.

        """

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def get_afct(self) -> AdaptiveFunction:
        return self._afct

    ## -------------------------------------------------------------------------------------------------
    def get_state_space(self) -> MSpace:
        return self._state_space

    ## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return self._action_space

    ## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):
        pass

    ## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
        return self._afct.get_hyperparam()

    ## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        super().switch_adaptivity(p_ada)
        self._afct.switch_adaptivity(p_ada)

    ## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        if self._afct is not None:
            self._afct.switch_logging(p_logging)

    ## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self._afct.set_random_seed(p_seed=p_seed)

    ## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        return self._afct.get_adapted()

    ## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._afct.clear_buffer()

    ## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        return self._afct.get_maturity()

    ## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        self._afct.init_plot(p_figure=p_figure)

    ## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        self._afct.update_plot()


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctSTrans(AFctBase):
    C_TYPE = 'AFct STrans'

    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_afct_cls,
                 p_state_space: MSpace,
                 p_action_space: MSpace,
                 p_input_space_cls=ESpace,
                 p_output_space_cls=ESpace,
                 p_output_elem_cls=State,  # Specific output element type
                 p_threshold=0,
                 p_buffer_size=0,
                 p_ada=True,
                 p_logging=Log.C_LOG_ALL,
                 **p_par):
        super().__init__(p_afct_cls,
                         p_state_space,
                         p_action_space,
                         p_input_space_cls=p_input_space_cls,
                         p_output_space_cls=p_output_space_cls,
                         p_output_elem_cls=p_output_elem_cls,
                         p_threshold=p_threshold,
                         p_buffer_size=p_buffer_size,
                         p_ada=p_ada,
                         p_logging=p_logging,
                         **p_par)

    ## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace, p_input_space: MSpace,
                      p_output_space: MSpace):
        # 1 Setup input space
        p_input_space.append(p_state_space)
        p_input_space.append(p_action_space)

        # 2 Setup output space
        p_output_space.append(p_state_space)

    ## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:
        # 1 Create input vector from given state and action
        input_values = p_state.get_values().copy()
        input_values.extend(p_action.get_sorted_values())
        input = Element(self._input_space)
        input.set_values(input_values)

        # 2 Compute and return new state
        return self._afct.map(input)

    ## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state: State, p_action: Action, p_state_new: State) -> bool:
        """
        Triggers adaptation of the embedded adaptive function.

        Parameters
        ----------
        p_state : State
            State.
        p_action : Action
            Action
        p_state_new : State
            New state

        Returns
        -------
        bool
            True, if something was adapted. False otherwise.
        """

        input_values = p_state.get_values().copy()
        input_values.extend(p_action.get_sorted_values())
        input = Element(self._input_space)
        input.set_values(input_values)

        return self._afct.adapt(input, p_state_new)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctReward(AFctBase):
    C_TYPE = 'AFct Reward'

    ## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace, p_input_space: MSpace,
                      p_output_space: MSpace):
        # 1 Setup input space
        p_input_space.append(p_state_space)
        p_input_space.append(p_state_space)

        # 2 Setup output space
        p_output_space.add_dim(
            Dimension(p_name_short='Rwd', p_base_set=Dimension.C_BASE_SET_R, p_name_long='Reward'))

    ## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state: State = None, p_state_new: State = None) -> Reward:
        if (p_state is None) or (p_state_new is None):
            raise ParamError('Both parameters p_state and p_state_new are needed to compute the reward')

        # 1 Create input vector from both states
        input_values = p_state.get_values().copy()
        input_values.append(p_state_new.get_values())
        input = Element(self._input_space)
        input.set_values(input_values)

        # 2 Compute and return reward
        output = self._afct.map(input)
        reward = output.get_values()[0]
        return Reward(p_type=Reward.C_TYPE_OVERALL, p_value=reward)

    ## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state: State, p_state_new: State, p_reward: Reward) -> bool:
        """
        Triggers adaptation of the embedded adaptive function.

        Parameters
        ----------
        p_state : State
            Previous state.
        p_state_new : State
            New state.
        p_reward : Reward
            Setpoint reward.

        Returns
        -------
        bool
            True, if something was adapted. False otherwise.
        """

        # 1 Create input vector from both states
        input_values = p_state.get_values().copy()
        input_values.append(p_state_new.get_values())
        input = Element(self._input_space)
        input.set_values(input_values)

        # 2 Create setpoint output vector
        output = Element(self._output_space)
        ids_ = output.get_dim_ids()
        output.set_value(ids_[0], p_reward.get_overall_reward())

        # 3 Trigger adaptation of embedded adaptive function
        return self._afct.adapt(input, output)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctSuccess(AFctBase):
    C_TYPE = 'AFct Success'

    ## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace, p_input_space: MSpace,
                      p_output_space: MSpace):

        # 1 Setup input space
        p_input_space.append(p_state_space)

        # 2 Setup output space
        p_output_space.add_dim(
            Dimension(p_name_short='Success', p_base_set=Dimension.C_BASE_SET_R, p_name_long='Success',
                      p_boundaries=[0, 1]))

    ## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        output = self._afct.map(p_state)

        if output.get_values()[0] >= 0.5:
            return True
        return False

    ## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state: State) -> bool:
        output = Element(self._output_space)
        ids_ = output.get_dim_ids()
        if p_state.get_success():
            output.set_value(ids_[0], 1)
        else:
            output.set_value(ids_[0], 0)

        return self._afct.adapt(p_state, output)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctBroken(AFctBase):
    C_TYPE = 'AFct Broken'

    ## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace, p_input_space: MSpace,
                      p_output_space: MSpace):

        # 1 Setup input space
        p_input_space.append(p_state_space)

        # 2 Setup output space
        p_output_space.add_dim(
            Dimension(p_name_short='Success', p_base_set=Dimension.C_BASE_SET_R, p_name_long='Success',
                      p_boundaries=[0, 1]))

    ## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        output = self._afct.map(p_state)

        if output.get_values()[0] >= 0.5:
            return True
        return False

    ## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state: State) -> bool:
        output = Element(self._output_space)
        ids_ = output.get_dim_ids()
        if p_state.get_success():
            output.set_value(ids_[0], 1)
        else:
            output.set_value(ids_[0], 0)

        return self._afct.adapt(p_state, output)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvBase(AFctSTrans, AFctReward, AFctSuccess, AFctBroken, Plottable, ScientificObject):
    """
    Base class for all environment classes. It defines the interface and elementary properties for
    an environment in the context of reinforcement learning.

    Parameters
    ----------
    p_latency : timedelta
        Optional latency of environment. If not provided, the internal value of constant C_LATENCY 
        is used by default.
    p_afct_strans : AFctSTrans
        Optional external adaptive function for state transition. 
    p_afct_reward : AFctReward
        Optional external adaptive function for reward computation.
    p_afct_success : AFctSuccess
        Optional external adaptive function for state evaluation 'success'.
    p_afct_broken : AFctBroken
        Optional external adaptive function for state evaluation 'broken'.
    p_logging 
        Log level (see class Log for more details).

    Attributes
    ----------
    _latency : timedelta
        Latency of the environment.
    _state : State
        Current state of environment.
    _state : State
        Previous state of environment.
    _last_action : Action
        Last action.
    _last_reward : Reward
        Last reward.
    _afct_strans : AFctSTrans
        Internal adaptive state transition function.
    _afct_reward : AFctReward
        Internal adaptive reward function.
    _afct_success : AFctSuccess
        Internal adaptive function for state evaluation 'success'.
    _afct_broken : AFctBroken
        Internal adaptive function for state evaluation 'broken'.

    """

    C_TYPE = 'Environment Base'
    C_NAME = '????'

    C_LATENCY = timedelta(0, 1, 0)  # Default latency 1s

    C_REWARD_TYPE = Reward.C_TYPE_OVERALL  # Default reward type for reinforcement learning

    C_SCIREF_TYPE = ScientificObject.C_SCIREF_TYPE_NONE

    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_latency: timedelta = None,
                 p_afct_strans: AFctSTrans = None,
                 p_afct_reward: AFctReward = None,
                 p_afct_success: AFctSuccess = None,
                 p_afct_broken: AFctBroken = None,
                 p_logging=Log.C_LOG_ALL):

        self._afct_strans = p_afct_strans
        self._afct_reward = p_afct_reward
        self._afct_success = p_afct_success
        self._afct_broken = p_afct_broken
        self._state_space = None
        self._action_space = None
        self._state = None
        self._prev_state = None
        self._last_action = None
        self._last_reward = None
        self._num_cycles = 0

        Log.__init__(self, p_logging=p_logging)
        self.set_latency(p_latency)

    ## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        """
        Adaptivity is switched off here.  
        """

        raise NotImplementedError('Classes of type ' + self.C_TYPE + ' are not adaptive!')

    ## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Adaptivity is switched off here. If called, then something went wrong. 
        """

        raise NotImplementedError('Classes of type ' + self.C_TYPE + ' are not adaptive!')

    ## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        """
        Adaptivity is switched off here. If called, then something went wrong. 
        """

        raise NotImplementedError('Classes of type ' + self.C_TYPE + ' are not adaptive!')

    ## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        """
        Maturity computation is switched off here. If called, the something went wrong.
        """

        raise NotImplementedError('Classes of type ' + self.C_TYPE + ' are not adaptive!')

    ## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        Log.switch_logging(self, p_logging)
        if self._afct_strans is not None:
            self._afct_strans.switch_logging(p_logging)
        if self._afct_reward is not None:
            self._afct_reward.switch_logging(p_logging)
        if self._afct_success is not None:
            self._afct_success.switch_logging(p_logging)
        if self._afct_broken is not None:
            self._afct_broken.switch_logging(p_logging)

    ## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        """
        Returns latency of environment.
        """

        return self._latency

    ## -------------------------------------------------------------------------------------------------
    def set_latency(self, p_latency: timedelta = None) -> None:
        """
        Sets latency of environment. If p_latency is None latency will be reset
        to internal value of attribute C_LATENCY.

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
    def get_reward_type(self):
        return self.C_REWARD_TYPE

    ## -------------------------------------------------------------------------------------------------
    def get_state(self) -> State:
        """
        Returns current state of environment.
        """

        return self._state

    ## -------------------------------------------------------------------------------------------------
    def _set_state(self, p_state: State) -> None:
        """
        Explicitly sets the current state of the environment. Internal use only.
        """

        self._state = p_state

    ## -------------------------------------------------------------------------------------------------
    def get_success(self) -> bool:
        if self._state is None:
            return False
        return self._state.get_success()

    ## -------------------------------------------------------------------------------------------------
    def get_broken(self) -> bool:
        if self._state is None:
            return False
        return self._state.get_broken()

    ## -------------------------------------------------------------------------------------------------
    def get_last_reward(self) -> Reward:
        return self._last_reward

    ## -------------------------------------------------------------------------------------------------
    def get_functions(self):
        return self._afct_strans, self._afct_reward, self._afct_success, self._afct_broken

    ## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self) -> int:
        """
        Returns limit of cycles per training episode. To be implemented in child classes.
        """

        raise NotImplementedError

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
        Resets environment to an initial state by calling the related custom method _reset().

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
        Custom method to reset the environment to an initial/defined state. 

        Parameters
        ----------
        p_seed : int
            Seed parameter for an internal random generator

        """

        raise NotImplementedError

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
        else:
            self.log(self.C_LOG_TYPE_E, 'Action processing failed')

        self._num_cycles += 1

        cycle_limit = self.get_cycle_limit()

        state = self.get_state()
        if not (state.get_terminal() and state.get_success()):
            state.set_timeout((cycle_limit > 0) and (self._num_cycles >= cycle_limit))

        return result

    ## -------------------------------------------------------------------------------------------------
    def _process_action(self, p_action: Action) -> bool:
        """
        Custom method for state transition. To be implemented in a child class. See method 
        process_action() for further details.
        """

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State = None, p_action: Action = None) -> State:
        """
        Simulates a state transition based on a state and an action. The simulation step itself is
        carried out either by an internal custom implementation in method _simulate_reaction() or
        by an embedded adaptive function.

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

        if self._afct_strans is not None:
            return self._afct_strans.simulate_reaction(p_state, p_action)
        else:
            return self._simulate_reaction(p_state, p_action)

    ## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """
        Custom implementation to simulate a state transition. See method simulate_reaction() for
        further details.
        """

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        """
        Computes a reward for the state transition, given by two successive states. The reward
        computation itself is carried out either by a custom implementation in method
        _compute_reward() or by an embedded adaptive function.

        Parameters
        ----------
        p_state_old : State
            Optional state before transition. If None the internal previous state of the environment
            is used.
        p_state_new : State
            Optional tate after transition. If None the internal current state of the environment
            is used.

        Returns
        -------
        Reward
            Reward object.

        """

        if p_state_old is not None:
            state_old = p_state_old
        else:
            state_old = self._prev_state

        if state_old is None:
            return None

        if p_state_new is not None:
            state_new = p_state_new
        else:
            state_new = self.get_state()

        if state_new is None:
            return None

        if self._afct_reward is not None:
            self._last_reward = self._afct_reward.compute_reward(p_state=state_old, p_state_new=state_new)
        else:
            self._last_reward = self._compute_reward(p_state_old=state_old, p_state_new=state_new)

        return self._last_reward

    ## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        """
        Custom reward computation method. See method compute_reward() for further details.
        """

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """
        Assesses the given state whether it is a 'success' state. Assessment is carried out either by
        a custom implementation in method _compute_success() or by an embedded adaptive function.

        Parameters
        ----------
        p_state : State
            State to be assessed.

        Returns
        -------
        bool
            True, if the given state is a 'success' state. False otherwise.

        """

        if self._afct_success is not None:
            return self._afct_success.compute_success(p_state)
        else:
            return self._compute_success(p_state)

    ## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        """
        Custom method for state evaluation 'success'. See method compute_success() for further details.
        """

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """
        Assesses the given state whether it is a 'broken' state. Assessment is carried out either by
        a custom implementation in method _compute_broken() or by an embedded adaptive function.

        Parameters
        ----------
        p_state : State
            State to be assessed.

        Returns
        -------
        bool
            True, if the given state is a 'broken' state. False otherwise.
            
        """

        if self._afct_broken is not None:
            return self._afct_broken.compute_broken(p_state)
        else:
            return self._compute_broken(p_state)

    ## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        """
        Custom method for state evaluation 'broken'. See method compute_broken() for further details.
        """

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        pass


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Environment(EnvBase, Mode):
    """
    This class represents the central environment model to be reused/inherited in own rl projects.

    Parameters
    ----------
    p_mode 
        Mode of environment. Possible values are Mode.C_MODE_SIM(default) or Mode.C_MODE_REAL.
    p_latency : timedelta
        Optional latency of environment. If not provided, the internal value of constant C_LATENCY 
        is used by default.
    p_afct_strans : AFctSTrans
        Optional external adaptive function for state transition 
    p_afct_reward : AFctReward
        Optional external adaptive function for reward computation
    p_afct_success : AFctSuccess
        Optional external adaptive function for state evaluation 'success'
    p_afct_broken : AFctBroken
        Optional external adaptive function for state evaluation 'broken'
    p_logging 
        Log level (see class Log for more details)

    """

    C_TYPE = 'Environment'

    C_CYCLE_LIMIT = 0  # Recommended cycle limit for training episodes

    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_latency: timedelta = None,
                 p_afct_strans: AFctSTrans = None,
                 p_afct_reward: AFctReward = None,
                 p_afct_success: AFctSuccess = None,
                 p_afct_broken: AFctBroken = None,
                 p_logging=Log.C_LOG_ALL):

        EnvBase.__init__(self,
                         p_latency=p_latency,
                         p_afct_strans=p_afct_strans,
                         p_afct_reward=p_afct_reward,
                         p_afct_success=p_afct_success,
                         p_afct_broken=p_afct_broken,
                         p_logging=p_logging)

        Mode.__init__(self, p_mode, p_logging)
        self._state_space, self._action_space = self.setup_spaces()
        self._num_cylces = 0

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

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self) -> int:
        """
        Returns limit of cycles per training episode.
        """

        if self.get_mode() == Mode.C_MODE_SIM:
            return self.C_CYCLE_LIMIT
        else:
            # In real operation mode there is no cycle limit
            return 0

    ## -------------------------------------------------------------------------------------------------
    def _process_action(self, p_action: Action) -> bool:
        """
        Processes given action and updates the state of the environment.

        Parameters
        ----------
        p_action : Action
            Action to be processed

        Returns
        -------
        bool
            True, if action processing was successfully. False otherwise.
        """

        # 0 Intro
        for agent in p_action.get_elem_ids():
            self.log(self.C_LOG_TYPE_I, 'Actions of agent', agent, '=', p_action.get_elem(agent).get_values())

        # 1 State transition
        if self.\
                _mode == self.C_MODE_SIM:
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
    def _export_action(self, p_action: Action) -> bool:
        """
        Mode C_MODE_REAL only: exports given action to be processed externally 
        (for instance by a real hardware). Please redefine. 

        Parameters
        ----------
        p_action : Action
            Action to be exported

        Returns
        -------
        bool
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
        bool
            True, if state import was successful. False otherwise.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvModel(EnvBase, Model):
    """
    Environment model class as part of a model-based agent.

    Parameters
    ----------
    p_observation_space : MSpace
        Observation space of related agent.
    p_action_space : MSpace
        Action space of related agent.
    p_latency : timedelta
        Latency of related environment.
    p_afct_strans : AFctSTrans
        Mandatory external adaptive function for state transition. 
    p_afct_reward : AFctReward
        Optional external adaptive function for reward computation.
    p_afct_success : AFctSuccess
        Optional external adaptive function for state assessment 'success'.
    p_afct_broken : AFctBroken
        Optional external adaptive function for state assessment 'broken'.
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    """

    C_TYPE = 'EnvModel'

    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_observation_space: MSpace,
                 p_action_space: MSpace,
                 p_latency: timedelta,
                 p_afct_strans: AFctSTrans,
                 p_afct_reward: AFctReward = None,
                 p_afct_success: AFctSuccess = None,
                 p_afct_broken: AFctBroken = None,
                 p_ada=True,
                 p_logging=Log.C_LOG_ALL):

        # 1 Intro
        EnvBase.__init__(self,
                         p_latency=p_latency,
                         p_afct_strans=p_afct_strans,
                         p_afct_reward=p_afct_reward,
                         p_afct_success=p_afct_success,
                         p_afct_broken=p_afct_broken,
                         p_logging=p_logging)

        Model.__init__(self, p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)

        self._state_space = p_observation_space
        self._action_space = p_action_space

        self._afct_strans = p_afct_strans
        self._afct_reward = p_afct_reward
        self._afct_success = p_afct_success
        self._afct_broken = p_afct_broken
        self._cycle_limit = 0

        # 2 Check adaptive functions for compatibility with agent

        # 2.1 Check state transition function
        try:
            if self._afct_strans.get_state_space() != self._state_space:
                raise ParamError(
                    'Observation spaces of environment model and adaptive state transition function are not equal')
            if self._afct_strans.get_action_space() != self._action_space:
                raise ParamError(
                    'Action spaces of environment model and adaptive state transition function are not equal')
        except:
            raise ParamError('Adaptive state transition function is mandatory')

        # 2.2 Check reward function
        if (self._afct_reward is not None) and (self._afct_reward.get_state_space() != self._state_space):
            raise ParamError(
                'Observation spaces of environment model and adaptive function for reward computation are not equal')

        # 2.3 Check function 'success'
        if (self._afct_success is not None) and (self._afct_success.get_state_space() != self._state_space):
            raise ParamError(
                'Observation spaces of environment model and adaptive function for assessment success are not equal')

        # 2.4 Check function 'broken'
        if (self._afct_broken is not None) and (self._afct_broken.get_state_space() != self._state_space):
            raise ParamError(
                'Observation spaces of environment model and adaptive function for assessment broken are not equal')

    ## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):

        # 1 Create overall hyperparameter space of all adaptive components inside
        hyperparam_space_init = False
        try:
            self._hyperparam_space = self._afct_strans.get_hyperparam().get_related_set().copy(p_new_dim_ids=False)
            hyperparam_space_init = True
        except:
            pass
        
        try:
            if hyperparam_space_init:
                self._hyperparam_space.append(self._afct_reward.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            else:
                self._hyperparam_space = self._afct_reward.get_hyperparam().get_related_set().copy(p_new_dim_ids=False)
                hyperparam_space_init = True
        except:
            pass
        
        try:
            if hyperparam_space_init:
                self._hyperparam_space.append(self._afct_success.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            else:
                self._hyperparam_space = self._afct_success.get_hyperparam().get_related_set().copy(p_new_dim_ids=False)
                hyperparam_space_init = True
        except:
            pass
        
        try:
            if hyperparam_space_init:
                self._hyperparam_space.append(self._afct_broken.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            else:
                self._hyperparam_space = self._afct_broken.get_hyperparam().get_related_set().copy(p_new_dim_ids=False)
                hyperparam_space_init = True
        except:
            pass
        
        # 2 Create overall hyperparameter (dispatcher) tuple
        self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)
        try:
            self._hyperparam_tuple.add_hp_tuple(self._afct_strans.get_hyperparam())
        except:
            pass
        
        try:
            self._hyperparam_tuple.add_hp_tuple(self._afct_reward.get_hyperparam())
        except:
            pass
        
        try:
            self._hyperparam_tuple.add_hp_tuple(self._afct_success.get_hyperparam())
        except:
            pass
        
        try:
            self._hyperparam_tuple.add_hp_tuple(self._afct_broken.get_hyperparam())
        except:
            pass
        
        
    ## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        self.set_random_seed(p_seed=p_seed)
        self._state = None
        self._prev_state = None

    ## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self) -> int:
        """
        Returns limit of cycles per training episode.
        """

        return self._cycle_limit

    ## -------------------------------------------------------------------------------------------------
    def _process_action(self, p_action: Action) -> bool:

        # 1 State transition
        self._set_state(self.simulate_reaction(self.get_state(), p_action))

        # 2 State evaluation
        state = self.get_state()
        state.set_success(self.compute_success(state))
        state.set_broken(self.compute_broken(state))

        return True

    ## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        Model.switch_adaptivity(self, p_ada)

        self._afct_strans.switch_adaptivity(p_ada)

        try:
            if self._afct_reward is not None:
                self._afct_reward.switch_adaptivity(p_ada)
        except:
            pass

        try:
            if self._afct_success is not None:
                self._afct_success.switch_adaptivity(p_ada)
        except:
            pass

        try:
            if self._afct_broken is not None:
                self._afct_broken.switch_adaptivity(p_ada)
        except:
            pass

    ## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Reactivated adaptation mechanism. See method Model.adapt() for further details.
        """

        return Model.adapt(self, *p_args)

    ## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Adapts the environment model based on State-Action-Reward-State (SARS) data.

        Parameters:
            p_args[0]           Object of type SARSElement
        """

        try:
            sars_dict = p_args[0].get_data()
            state = sars_dict['state']
            action = sars_dict['action']
            reward = sars_dict['reward']
            state_new = sars_dict['state_new']
        except:
            raise ParamError('Parameter must be of type SARSElement')

        adapted = self._afct_strans.adapt(state, action, state_new)

        if self._afct_reward is not None:
            adapted = adapted or self._afct_reward.adapt(state, state_new, reward)

        if self._afct_success is not None:
            adapted = adapted or self._afct_success.adapt(state_new)

        if self._afct_broken is not None:
            adapted = adapted or self._afct_broken.adapt(state_new)

        if (self._cycle_limit == 0) and state_new.get_timeout():
            # First timeout state defines the cycle limit
            self._cycle_limit = self._num_cycles

        return adapted

    ## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        return Model.get_adapted(self)

    ## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        """
        Returns maturity of environment model as average maturity of the embedded adaptive functions.
        """

        maturity = self._afct_strans.get_maturity()
        num_afct = 1

        try:
            if self._afct_reward is not None:
                maturity += self._afct_reward.get_maturity()
                num_afct += 1
        except:
            pass

        try:
            if self._afct_success is not None:
                maturity += self._afct_success.get_maturity()
                num_afct += 1
        except:
            pass

        try:
            if self._afct_broken is not None:
                maturity += self._afct_broken.get_maturity()
                num_afct += 1
        except:
            pass

        return maturity / num_afct

    ## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._afct_strans.clear_buffer()
        if self._afct_reward is not None:
            self._afct_reward.clear_buffer()
        if self._afct_success is not None:
            self._afct_success.clear_buffer()
        if self._afct_broken is not None:
            self._afct_broken.clear_buffer()
