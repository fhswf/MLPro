## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
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
## -- 2021-11-26  1.2.0     DA       Redesign:
## --                                - Introduction of special adaptive function classes AFct*
## --                                - Rework of classes EnvBase, Environment, EnvModel
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2021-11-26)

This module provides model classes for environments and environnment models.
"""


from mlpro.rl.models_sar import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctSTrans (Model):
    """
    Special adaptive function for state transition prediction.

    Parameters
    ----------
    p_afct_cls : str
        Name of an adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space    
    p_action_space: MSpace
        Action space
    p_threshold : float
        See description of class AdaptiveFunction
    p_buffer_size: int
        Initial size of internal data buffer (0=no buffering)
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    Attributes
    ----------
    _state_space : MSpace
        State space
    _action_space : MSpace
        Action space

    """

    C_TYPE          = 'AFct STrans'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_afct_cls, p_state_space:MSpace, p_action_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=Log.C_LOG_ALL):
         
        self._state_space   = p_state_space
        self._action_space  = p_action_space

        # concatenate state and action space to input space
        # ...
        input_space = None 

        self._afct = p_afct_cls(p_input_space=input_space, p_output_space=p_state_space, p_output_elem_cls=State, p_threshold=p_threshold, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_state_space(self) -> MSpace:
        return self._state_space


## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return self._action_space


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state:State, p_action:Action) -> State:
        input = None
        return super().map(input)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State, p_action:Action) -> bool:
        # to be implemented...
        # 
        #
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctReward (Model):
    """
    Special adaptive function for reward prediction.

    Parameters
    ----------
    p_afct_cls : str
        Name of an adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space    
    p_threshold : float
        See description of class AdaptiveFunction
    p_buffer_size: int
        Initial size of internal data buffer (0=no buffering)
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    Attributes
    ----------
    _state_space : MSpace
        State space

    """

    C_TYPE          = 'AFct Reward'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_afct_cls, p_state_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=Log.C_LOG_ALL):

        self._state_space   = p_state_space

        # concatenate state and action space to input space
        # ...
        input_space     = None 
        output_space    = None

        self._afct = p_afct_cls(p_input_space=input_space, p_output_space=output_space, p_output_elem_cls=Element, p_threshold=p_threshold, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_state_space(self) -> MSpace:
        return self._state_space


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        """
        Predicts the reward based on two consecutive states using the given adaptive function.

        Parameters
        ----------
        p_state_old : State
            State before last action
        p_state_new : State
            State after last action

        Returns
        -------
        Reward
            Object of type Reward
        """

        # to be implemented...
        # 
        #
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State, p_action:Action, p_reward:Reward) -> bool:
        """
        Adapts the adaptive function inside.

        Parameters
        ----------
        p_state : State
            State
        p_action : Action
            Action
        p_reward : Reward
            Target value for the reward
        """

        # to be implemented...
        # 
        #
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctDone (Model):
    """
    Special adaptive function for environment done state prediction.

    Parameters
    ----------
    p_afct_cls : str
        Name of an adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space    
    p_threshold : float
        See description of class AdaptiveFunction
    p_buffer_size: int
        Initial size of internal data buffer (0=no buffering)
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    Attributes
    ----------
    _state_space : MSpace
        State space

    """

    C_TYPE          = 'AFct Done'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_afct_cls, p_state_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=True):

        self._state_space   = p_state_space

        # concatenate state and action space to input space
        # ...
        output_space = None 

        self._afct = p_afct_cls(p_input_space=p_state_space, p_output_space=output_space, p_output_elem_cls=Element, p_threshold=p_threshold, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_state_space(self) -> MSpace:
        return self._state_space


# -------------------------------------------------------------------------------------------------
    def compute_done(self, p_state:State) -> bool:
        # to be implemented...
        # 
        #
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        # to be implemented...
        # 
        #
        pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctBroken (Model):
    """
    Special adaptive function for environment broken state prediction.

    Parameters
    ----------
    p_afct_cls : str
        Name of an adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space    
    p_threshold : float
        See description of class AdaptiveFunction
    p_buffer_size: int
        Initial size of internal data buffer (0=no buffering)
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    Attributes
    ----------
    _state_space : MSpace
        State space

    """

    C_TYPE          = 'AFct Broken'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_afct_cls, p_state_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=True):

        self._state_space   = p_state_space

        # concatenate state and action space to input space
        # ...
        output_space = None 

        self._afct = p_afct_cls(p_input_space=p_state_space, p_output_space=output_space, p_output_elem_cls=Element, p_threshold=p_threshold, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_state_space(self) -> MSpace:
        return self._state_space


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state:State) -> bool:
        # to be implemented...
        # 
        #
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        # to be implemented...
        # 
        #
        pass


       


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvBase (AFctSTrans, AFctReward, AFctDone, AFctBroken, Plottable, ScientificObject):
    """
    Base class for all environment classes. It defines the interface and elementry properties for
    an environment in the context of reinforcement learning.

    Parameters
    ----------
    p_latency : timedelta
        Optional latency of environment. If not provided, the internal value of constant C_LATENCY 
        is used by default.
    p_afct_strans : AFctSTrans
        Optional external adaptive function for state transition 
    p_afct_reward : AFctReward
        Optional external adaptive function for reward computation
    p_afct_done : AFctDone
        Optional external adaptive function for state evaluation 'done'
    p_afct_broken : AFctBroken
        Optional external adaptive function for state evaluation 'broken'
    p_logging 
        Log level (see class Log for more details)

    Attributes
    ----------
    _state : State
        Current state of environment
    _latency : timedelta
        Latency of the environment
    _last_action : Action
        Last action
    _afct_strans : AFctSTrans
        Internal adaptive state transition function
    _afct_reward : AFctReward
        Internal adaptive reward function
    _afct_done : AFctDone
        Internal adaptive function for state evaluation 'done'
    _afct_broken : AFctBroken
        Internal adaptive function for state evaluation 'broken'

    """

    C_TYPE          = 'Environment Base'
    C_NAME          = '????'

    C_LATENCY       = timedelta(0,1,0)              # Default latency 1s

    C_REWARD_TYPE   = Reward.C_TYPE_OVERALL         # Default reward type for reinforcement learning

    C_SCIREF_TYPE   = ScientificObject.C_SCIREF_TYPE_NONE

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_latency:timedelta=None, 
                 p_afct_strans:AFctSTrans=None,
                 p_afct_reward:AFctReward=None,
                 p_afct_done:AFctDone=None,
                 p_afct_broken:AFctBroken=None,     
                 p_logging=Log.C_LOG_ALL): 

        Log.__init__(self, p_logging=p_logging)
        self._afct_strans       = p_afct_strans
        self._afct_reward       = p_afct_reward
        self._afct_done         = p_afct_done
        self._afct_broken       = p_afct_broken
        self._state_space       = None
        self._action_space      = None
        self._state             = None
        self._last_action       = None
        self.set_latency(p_latency)


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        """
        Returns latency of environment.
        """

        return self._latency


## -------------------------------------------------------------------------------------------------
    def set_latency(self, p_latency:timedelta=None) -> None:
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
    def _set_state(self, p_state:State) -> None:
        """
        Explicitely sets the current state of the environment. Internal use only.
        """

        self._state = p_state


## -------------------------------------------------------------------------------------------------
    def get_done(self) -> bool:
        if self._state is None: return False
        return self._state.get_done()


## -------------------------------------------------------------------------------------------------
    def get_broken(self) -> bool:
        if self._state is None: return False
        return self._state.get_broken()


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.
        """
        
        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=None) -> None:
        """
        Resets environment to initial state. Please redefine.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action:Action) -> bool:
        """
        Processes given action and updates the state of the environment.

        Parameters
        ----------
        p_action : Action
            Action to be processed

        Returns
        -------
        bool
            True, if action processing was successfull. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Adaptivity is switched off here. 
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Environment (EnvBase, Mode):
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
    p_afct_done : AFctDone
        Optional external adaptive function for state evaluation 'done'
    p_afct_broken : AFctBroken
        Optional external adaptive function for state evaluation 'broken'
    p_logging 
        Log level (see class Log for more details)

    """

    C_TYPE          = 'Environment'
 
    C_CYCLE_LIMIT   = 0             # Recommended cycle limit for training episodes

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_mode=Mode.C_MODE_SIM,        
                 p_latency:timedelta=None, 
                 p_afct_strans:AFctSTrans=None,
                 p_afct_reward:AFctReward=None,
                 p_afct_done:AFctDone=None,
                 p_afct_broken:AFctBroken=None,     
                 p_logging=Log.C_LOG_ALL): 

        EnvBase.__init__(self, 
                         p_latency=p_latency, 
                         p_afct_strans=p_afct_strans, 
                         p_afct_reward=p_afct_reward,
                         p_afct_done=p_afct_done,
                         p_afct_broken=p_afct_broken,
                         p_logging=p_logging)

        Mode.__init__(self, p_mode, p_logging)
        self._state_space, self._action_space = self.setup_spaces()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        """
        Static template method to setup and return state and action space of environment.
        
        Returns
        -------
        MSpace
            State space object
        MSpace
            Action space object

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        """
        Returns limit of cycles per training episode.
        """

        return self.C_CYCLE_LIMIT


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action:Action) -> bool:
        """
        Processes given action and updates the state of the environment.

        Parameters
        ----------
        p_action : Action
            Action to be processed

        Returns
        -------
        bool
            True, if action processing was successfull. False otherwise.
        """

        # 0 Intro
        self.last_action = p_action
        self.log(self.C_LOG_TYPE_I, 'Start processing action')
        for agent in p_action.get_elem_ids():
            self.log(self.C_LOG_TYPE_I, 'Actions of agent', agent, '=', p_action.get_elem(agent).get_values())


        # 1 State transition
        if self._mode == self.C_MODE_SIM:
            # 1.1 Simulated state transition
            self._set_state( self.simulate_reaction( self.get_state(), p_action ) )

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
        state.set_done(self.compute_done(state))
        state.set_broken(self.compute_broken(state))
        

        # 3 Outro
        self.log(self.C_LOG_TYPE_I, 'Action processing finished successfully')
        return True


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state:State, p_action:Action) -> State:
        """
        Simulates a state transition by either calling the pretrained adaptive function or by calling
        a custom implementation in method _simulate_reaction().

        Parameters
        ----------
        p_state : State
            Current state of environment
        p_action : Action
            Action of an agent
        
        Returns
        -------
        State
            Subsequent state
        """

        if self._afct_strans is not None:
            return self._afct_strans.simulate_reaction(p_state, p_action)
        else:
            return self._simulate_reaction(p_state, p_action)


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action) -> State:
        """
        Custom implementation for a simulated state transition. To be redefined in own environment.

        Parameters
        ----------
        p_state : State
            Current state of environment
        p_action : Action
            Action of an agent
        
        Returns
        -------
        State
            Subsequent state
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _export_action(self, p_action:Action) -> bool:
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
    def compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        """
        Computes a reward by either calling the pretrained adaptive function or by calling a custom 
        implementation in method _compute_reward().

        Parameters
        ----------
        p_state_old : State
            Old state (before state transition)
        p_state_new : State
            New state (after state transition)
        
        Returns
        -------
        Reward
            Reward object
        """

        if self._afct_reward is not None:
            return self._afct_reward.compute_reward(p_state_old, p_state_new)
        else:
            return self._compute_reward(p_state_old, p_state_new)


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        """
        Custom implementation for reward computation. To be redefined in own environment.

        Parameters
        ----------
        p_state_old : State
            Old state (before state transition)
        p_state_new : State
            New state (after state transition)
        
        Returns
        -------
        Reward
            Reward object
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def compute_done(self, p_state:State) -> bool:
        """
        Evaluates a given state as done state by either calling the pretrained adaptive function or
        by calling a custom implementation in method _compute_done().

        Parameters
        ----------
        p_state : State
            State of environment
        
        Returns
        -------
        bool
            True, if given state is a done state. False otherwise.

        """

        if self._afct_done is not None:
            return self._afct_done.compute_done(p_state)
        else:
            return self._compute_done(p_state)


## -------------------------------------------------------------------------------------------------
    def _compute_done(self, p_state:State) -> bool:
        """
        Custom implementation for state evaluation 'done'. To be redefined in own environment.

        Parameters
        ----------
        p_state : State
            State of environment
        
        Returns
        -------
        bool
            True, if given state is a done state. False otherwise.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state:State) -> bool:
        """
        Evaluates a given state as broken state by either calling the pretrained adaptive function or
        by calling a custom implementation in method _compute_broken().

        Parameters
        ----------
        p_state : State
            State of environment
        
        Returns
        -------
        bool
            True, if given state is a broken state. False otherwise.

        """

        if self._afct_broken is not None:
            return self._afct_broken.compute_broken(p_state)
        else:
            return self._compute_broken(p_state)


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        """
        Custom implementation for state evaluation 'broken'. To be redefined in own environment.

        Parameters
        ----------
        p_state : State
            State of environment
        
        Returns
        -------
        bool
            True, if given state is a broken state. False otherwise.

        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvModel(EnvBase, Model):
    """
    Environment model class as part of a model-based agent.

    Parameters
    ----------
    p_latency : timedelta
        Latency of related environment.
    p_afct_strans : AFctSTrans
        Optional external adaptive function for state transition 
    p_afct_reward : AFctReward
        Optional external adaptive function for reward computation
    p_afct_done : AFctDone
        Optional external adaptive function for state evaluation 'done'
    p_afct_broken : AFctBroken
        Optional external adaptive function for state evaluation 'broken'
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    """

    C_TYPE          = 'EnvModel'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_latency:timedelta,
                 p_afct_strans:AFctSTrans, 
                 p_afct_reward:AFctReward, 
                 p_afct_done:AFctDone, 
                 p_afct_broken:AFctBroken, 
                 p_ada=True, 
                 p_logging=Log.C_LOG_ALL):

        EnvBase.__init__(self, 
                         p_latency=p_latency, 
                         p_afct_strans=p_afct_strans,
                         p_afct_reward=p_afct_reward,
                         p_afct_done=p_afct_done,
                         p_afct_broken=p_afct_broken,
                         p_logging=p_logging )

        Model.__init__(self, p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)

        self._afct_strans   = p_afct_strans
        self._afct_reward   = p_afct_reward
        self._afct_done     = p_afct_done
        self._afct_broken   = p_afct_broken

        self._state_space   = self._afct_strans.get_state_space()
        self._action_space  = self._afct_strans.get_action_space()


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action: Action) -> bool:
        return super().process_action(p_action)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Adapts the internal predictive functions based on State-Action-Reward-State (SARS) data.

        Parameters:
            p_arg[0]           Object of type SARSElement
        """

        # ... to be implemented
        pass


## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        """
        Returns maturity of environment model.
        """

        return min(self._afct_strans.get_maturity(), self._afct_reward.get_maturity(), self._afct_done.get_maturity(), self._afct_broken.get_maturity())


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._afct_strans.clear_buffer()
        self._afct_reward.clear_buffer()
        self._afct_done.clear_buffer()
        self._afct_broken.clear_buffer()


## -------------------------------------------------------------------------------------------------
    def get_functions(self):
        return self._afct_strans, self._afct_reward, self._afct_done, self._afct_broken


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action: Action) -> bool:

        # 1 Concatenate internal state and given action to input element of state transition fct
        # ...
        state_action = None

        # 2 Predict next state
        self._set_state(self._afct_strans.map(state_action))

        return True


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Adapts the environment model based on State-Action-Reward-State (SARS) data.

        Parameters:
            p_arg[0]           Object of type SARSElement
        """

        raise NotImplementedError