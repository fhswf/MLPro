## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
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
## -- 2022-08-15  1.4.3     SY       Renaming maturity to accuracy
## -- 2022-10-06  1.4.4     SY       - Handling numpy array on _adapt method of AFctSTrans class
## --                                - Same issue on simulate_reaction method of AFctSTrans class
## -- 2022-11-01  1.5.0     DA       - Classes EnvBase, Environment, EnvModel: new param p_visualize
## --                                - Cleaned the code a bit
## -- 2022-11-02  1.6.0     DA       Refactoring: methods adapt(), _adapt()
## -- 2022-11-09  1.6.1     DA       Refactoring due to changes on plot systematics
## -- 2022-11-29  1.7.0     DA       - Refactoring due to new underlying module mlpro.bf.systems
## --                                - Adaptive parts moved to new module models_env_ada.py 
## -- 2023-01-27  1.7.1     MRD      Add optional argument for Environment to integrate MuJoCo
## -- 2023-02-02  1.7.2     DA       All methods compute_reward(): unified name of first parameter 
## --                                to p_state_old
## -- 2023-02-13  1.7.3     MRD       Simplify State Space and Action Space generation
## -- 2023-05-30  1.7.4     LSB      Redefining the inheritence order in EnvBase to resolve MRO in OAEnv
## -- 2025-07-17  1.8.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.8.0 (2025-07-17) 

This module provides model classes for environments.
"""

from datetime import timedelta

from mlpro.bf import Log, Mode, ParamError
from mlpro.bf.various import TStamp
from mlpro.bf.systems import *



# Export list for public API
__all__ = [ 'Reward',
            'FctReward',
            'EnvBase',
            'Environment' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Reward (TStamp):
    """
    Objects of this class represent rewards of environments. The internal structure
    depends on the reward type. Three types are supported as listed below.

    Parameters
    ----------
    p_type          
        Reward type (default: C_TYPE_OVERALL)
    p_value         
        Overall reward value (reward type C_TYPE_OVERALL only)
    """

    C_TYPE_OVERALL          = 0  # Reward is a scalar (default)
    C_TYPE_EVERY_AGENT      = 1  # Reward is a scalar for every agent
    C_TYPE_EVERY_ACTION     = 2  # Reward is a scalar for every agent and action

    C_VALID_TYPES           = [ C_TYPE_OVERALL, C_TYPE_EVERY_AGENT, C_TYPE_EVERY_ACTION ]

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_type=C_TYPE_OVERALL, p_value=0):
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
class FctReward (Log):
    """
    Template class for reward functions.

    Parameters
    ----------
    p_logging 
        Log level (see class Log for more details).
    """

    C_TYPE = 'Fct Reward'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL ):
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        """
        Computes a reward based on a predecessor and successor state. Custom method _compute_reward()
        is called.

        Parameters
        ----------
        p_state_old : State
            Predecessor state.
        p_state_new : State
            Successor state.

        Returns
        -------
        r : Reward
            Reward
        """

        self.log(Log.C_LOG_TYPE_I, 'Computing reward...')
        return self._compute_reward( p_state_old = p_state_old, 
                                     p_state_new = p_state_new )


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        """
        Custom reward method. See method compute_reward() for further details.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvBase (FctReward, System):
    """
    Base class for all environment classes. It defines the interface and elementary properties for
    an environment in the context of reinforcement learning.

    Parameters
    ----------
    p_latency : timedelta
        Optional latency of environment. If not provided, the internal value of constant C_LATENCY 
        is used by default.
    p_fct_strans : FctSTrans
        Optional external function for state transition. 
    p_fct_reward : FctReward
        Optional external function for reward computation.
    p_fct_success : FctSuccess
        Optional external function for state evaluation 'success'.
    p_fct_broken : FctBroken
        Optional external function for state evaluation 'broken'.
    p_mujoco_file
        Path to XML file for MuJoCo model.
    p_frame_skip : int
        Frame to be skipped every step. Default = 1.
    p_state_mapping
        State mapping if the MLPro state and MuJoCo state have different naming.
    p_action_mapping
        Action mapping if the MLPro action and MuJoCo action have different naming.
    p_use_radian : bool
        Use radian if the action and the state based on radian unit. Default = True.
    p_camera_conf : tuple
        Default camera configuration on MuJoCo Simulation (xyz position, elevation, distance).
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
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

    C_TYPE          = 'Environment Base'

    C_REWARD_TYPE   = Reward.C_TYPE_OVERALL  # Default reward type for reinforcement learning

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_mode = Mode.C_MODE_SIM,
                  p_latency : timedelta = None,
                  p_fct_strans : FctSTrans = None,
                  p_fct_reward : FctReward = None,
                  p_fct_success : FctSuccess = None,
                  p_fct_broken : FctBroken = None,
                  p_mujoco_file = None,
                  p_frame_skip : int = 1,
                  p_state_mapping = None,
                  p_action_mapping = None,
                  p_camera_conf : tuple = (None, None, None),
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL ):

        System.__init__( self,
                         p_mode = p_mode,
                         p_latency = p_latency,
                         p_fct_strans = p_fct_strans,
                         p_fct_success = p_fct_success,
                         p_fct_broken = p_fct_broken,
                         p_mujoco_file = p_mujoco_file,
                         p_frame_skip = p_frame_skip,
                         p_state_mapping = p_state_mapping,
                         p_action_mapping = p_action_mapping,
                         p_camera_conf = p_camera_conf,
                         p_visualize = p_visualize,
                         p_logging = p_logging )

        self._fct_reward = p_fct_reward
        self._num_cycles = 0


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        System.switch_logging( self, p_logging=p_logging )

        if self._fct_reward is not None:
            self._fct_reward.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def get_reward_type(self):
        return self.C_REWARD_TYPE


## -------------------------------------------------------------------------------------------------
    def get_last_reward(self) -> Reward:
        return self._last_reward


## -------------------------------------------------------------------------------------------------
    def get_functions(self):
        return self._fct_strans, self._fct_reward, self._fct_success, self._fct_broken


## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self) -> int:
        """
        Returns limit of cycles per training episode. To be implemented in child classes.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _process_action(self, p_action: Action) -> bool:
        """
        Custom method for state transition. To be implemented in a child class. See method 
        process_action() for further details.
        """

        result = System._process_action( self, p_action=p_action)

        self._num_cycles += 1

        cycle_limit = self.get_cycle_limit()

        state = self.get_state()
        if not (state.get_terminal() and state.get_success()):
            state.set_timeout((cycle_limit > 0) and (self._num_cycles >= cycle_limit))

        return result


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

        if self._fct_reward is not None:
            self._last_reward = self._fct_reward.compute_reward(p_state_old=state_old, p_state_new=state_new)
        else:
            self._last_reward = self._compute_reward(p_state_old=state_old, p_state_new=state_new)

        return self._last_reward





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Environment (EnvBase):
    """
    This class represents the central environment model to be reused/inherited in own rl projects.

    Parameters
    ----------
    p_mode 
        Mode of environment. Possible values are Mode.C_MODE_SIM(default) or Mode.C_MODE_REAL.
    p_latency : timedelta
        Optional latency of environment. If not provided, the internal value of constant C_LATENCY 
        is used by default.
    p_fct_strans : FctSTrans
        Optional external function for state transition. 
    p_fct_reward : FctReward
        Optional external function for reward computation.
    p_fct_success : FctSuccess
        Optional external function for state evaluation 'success'.
    p_fct_broken : FctBroken
        Optional external function for state evaluation 'broken'.
    p_mujoco_file
        Path to XML file for MuJoCo model.
    p_frame_skip : int
        Frame to be skipped every step. Default = 1.
    p_state_mapping
        State mapping if the MLPro state and MuJoCo state have different naming.
    p_action_mapping
        Action mapping if the MLPro action and MuJoCo action have different naming.
    p_use_radian : bool
        Use radian if the action and the state based on radian unit. Default = True.
    p_camera_conf : tuple
        Default camera configuration on MuJoCo Simulation (xyz position, elevation, distance).
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = True.
    p_logging 
        Log level (see class Log for more details)
    """

    C_TYPE          = 'Environment'

    C_CYCLE_LIMIT   = 0  # Recommended cycle limit for training episodes

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_mode = Mode.C_MODE_SIM,
                  p_latency : timedelta = None,
                  p_fct_strans : FctSTrans = None,
                  p_fct_reward : FctReward = None,
                  p_fct_success : FctSuccess = None,
                  p_fct_broken : FctBroken = None,
                  p_mujoco_file = None,
                  p_frame_skip : int = 1,
                  p_state_mapping = None,
                  p_action_mapping = None,
                  p_camera_conf : tuple = (None, None, None),
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL ):

        super().__init__(p_mode = p_mode,
                         p_latency = p_latency,
                         p_fct_strans = p_fct_strans,
                         p_fct_reward = p_fct_reward,
                         p_fct_success = p_fct_success,
                         p_fct_broken = p_fct_broken,
                         p_mujoco_file = p_mujoco_file,
                         p_frame_skip = p_frame_skip,
                         p_state_mapping = p_state_mapping,
                         p_action_mapping = p_action_mapping,
                         p_camera_conf = p_camera_conf,
                         p_visualize = p_visualize,
                         p_logging = p_logging )


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