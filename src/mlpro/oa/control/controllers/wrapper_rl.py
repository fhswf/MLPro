## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.controller
## -- Module  : wrapper_rl.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-19  0.1.0     DA       Initial implementation of class OAControllerRL
## -- 2024-10-09  0.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-10-09)

This module provides a wrapper class for MLPro's RL policies.

"""


from mlpro.bf.control.basics import ControlError, SetPoint
from mlpro.bf.math.basics import Log
from mlpro.bf.mt import Log, Task
from mlpro.bf.systems.basics import ControlVariable, ControlledVariable
from mlpro.oa.control import OAController
from mlpro.rl import SARSElement, FctReward, Policy




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
    C_NAME          = ''

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
        
        self._rl_policy : Policy             = p_rl_policy
        self._rl_fct_reward : FctReward      = p_rl_fct_reward
        self._error_old : ControlError       = None
        self._ctrl_var_old : ControlVariable = None

        super().__init__( p_ada = p_ada,
                          p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        super().switch_adaptivity(p_ada=p_ada)
        self._rl_policy.switch_adaptivity(p_ada=p_ada)


## -------------------------------------------------------------------------------------------------
    def compute_output(self, p_ctrl_error: ControlError, p_ctrl_var : ControlVariable = None ) -> ControlVariable:
        return self._rl_policy.compute_action( p_obs = p_ctrl_error )


## -------------------------------------------------------------------------------------------------
    def _adapt( self, 
                p_ctrl_error: ControlError, 
                p_ctrl_var: ControlVariable ) -> bool:
        
        # 0 Intro
        adapted = False


        # 1 Adaptation from the second cycle on
        if self._error_old is not None:

            # 1.1 Call reward function
            reward = self._rl_fct_reward.compute_reward( p_state_old = self._error_old, p_state_new = p_ctrl_error )


            # 1.2 Setup SARS element
            sars_elem : SARSElement = SARSElement( p_state = self._error_old,
                                                   p_action = self._ctrl_var_old,
                                                   p_reward = reward,
                                                   p_state_new = p_ctrl_error )


            # 1.3 Trigger adaptation of the RL policy
            adapted = self._rl_policy.adapt( p_sars_elem = sars_elem )


        # 2 Buffering of current control variable and control error
        self._error_old    = p_ctrl_error
        self._ctrl_var_old = p_ctrl_var 


        # 3 Outro
        return adapted