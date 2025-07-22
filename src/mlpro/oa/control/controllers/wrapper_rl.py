## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.controller
## -- Module  : wrapper_rl.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-19  0.1.0     DA       Initial implementation of class OAControllerRL
## -- 2024-10-09  0.2.0     DA       Refactoring
## -- 2024-10-13  0.3.0     DA       Refactoring
## -- 2024-10-16  0.3.1     DA/ASP   Bugfix in method OAControllerRL._adapt()
## -- 2024-12-05  0.4.0     DA       Class OAControllerRL: 
## --                                - implementation of plot methods
## .-                                - redefintion of method assign_so()
## -- 2024-12-06  0.4.1     DA       Method OAControllerRL.__init__(): handling of the own name
## -- 2025-06-02  0.4.2     DA       - Bugfix in OAControllerRL.__init__()
## --                                - Plot methods of OAControllerRL disabled
## -- 2025-06-11  0.5.0     DA       Refactoring
## -- 2025-07-22  0.6.0     DA       Refactoring: __all__ export list, docstring, imports
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.6.0 (2025-07-22)

This module provides a wrapper class for MLPro's RL policies.

"""


from mlpro.bf import Log
from mlpro.bf.mt import Task
from mlpro.bf.control import ControlError, ControlVariable
from mlpro.bf.systems import State, Action

from mlpro.oa.control import OAController
from mlpro.rl import SARSElement, FctReward, Policy



# Export list for public API
__all__ = [ 'OAControllerRL' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControllerRL (OAController):
    """
    Wrapper class for online-adaptive closed-loop controllers reusing reinforcement learning objects
    like a policy and a reward function.

    Parameters
    ----------
    p_rl_policy : Policy
        RL policy object.
    p_rl_fct_reward : FctReward
        RL reward function.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters. 
    """

    C_TYPE          = 'OA Controller RL'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_rl_policy : Policy,
                  p_rl_fct_reward : FctReward,
                  p_ada : bool = True,
                  p_name: str = None, 
                  p_range_max=Task.C_RANGE_THREAD, 
                  p_visualize: bool = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        self._rl_policy : Policy        = p_rl_policy
        self._rl_fct_reward : FctReward = p_rl_fct_reward
        self._state_old : State         = None
        self._action_old : Action       = None

        if p_name is not None:
            name = p_name
        else:
            name = p_rl_policy.get_name()

        super().__init__( p_input_space = p_rl_policy.get_observation_space(),
                          p_output_space = p_rl_policy.get_action_space(),
                          p_ada = p_ada,
                          p_name = name, 
                          p_range_max = p_range_max, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )
        

## -------------------------------------------------------------------------------------------------
    def assign_so(self, p_so):
        super().assign_so( p_so = p_so )
        self._rl_policy.assign_so( p_so = p_so )


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        super().switch_adaptivity(p_ada=p_ada)
        self._rl_policy.switch_adaptivity(p_ada=p_ada)


## -------------------------------------------------------------------------------------------------
    def compute_output( self, p_ctrl_error: ControlError ) -> ControlVariable:

        # 1 Convert control error to RL state
        state = State( p_state_space = p_ctrl_error.value_space )
        state.values = p_ctrl_error.values
        state.id = p_ctrl_error.id
        state.tstamp = p_ctrl_error.tstamp

        # 2 Let the RL policy compute th next action
        action = self._rl_policy.compute_action( p_obs = state )

        # 3 Convert RL action to control variable and return
        return ControlVariable( p_id = self.get_so().get_next_inst_id(),
                                p_value_space = action.get_feature_data().get_related_set(),
                                p_values = action.get_feature_data().get_values(),
                                p_tstamp = self.get_so().get_tstamp() )


## -------------------------------------------------------------------------------------------------

    def _adapt(self, p_ctrl_error: ControlError, p_ctrl_var: ControlVariable) -> bool:
       
        # 0 Intro
        adapted = False

        # 1 Convert control error to RL state
        state_new        = State( p_state_space = p_ctrl_error.value_space )
        state_new.id     = p_ctrl_error.id
        state_new.tstamp = p_ctrl_error.tstamp
        state_new.values = p_ctrl_error.values

        # 2 Adaptation from the second cycle on
        if self._state_old is None: 
            self._state_old = state_new
            return False

        # 3 Call reward function
        reward = self._rl_fct_reward.compute_reward( p_state_old = self._state_old, p_state_new = state_new )

        # 4 Setup SARS element
        sars_elem = SARSElement( p_state = self._state_old,
                                 p_action = self._action_old,
                                 p_reward = reward,
                                 p_state_new = state_new )

        # 5 Trigger adaptation of the RL policy
        adapted = self._rl_policy.adapt( p_sars_elem = sars_elem )

        # 6 Buffering of new state and action
        self._state_old  = state_new
        self._action_old = Action( p_agent_id = self.id,
                                   p_action_space = p_ctrl_var.value_space,
                                   p_values = p_ctrl_var.values,
                                   p_tstamp = p_ctrl_var.tstamp )

        # 7 Outro
        return adapted
    

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure = None, p_plot_settings = None):
        super().init_plot( p_figure = p_figure, p_plot_settings = p_plot_settings )
        self._rl_policy.init_plot( p_figure = p_figure, p_plot_settings = p_plot_settings )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_instances = None, **p_kwargs):
        super().update_plot(p_instances = p_instances, **p_kwargs)
        self._rl_policy.update_plot( **p_kwargs )
