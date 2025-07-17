## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.dynamicgames
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-21  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-06-25  1.0.1     DA       Class GameBoard: introduction of internal constant C_REWARD_TYPE
## -- 2021-08-28  1.0.2     DA       Adjustments after changings on rl models
## -- 2021-09-11  1.0.3     MRD      Change Header information to match our new library name
## -- 2021-11-16  1.1.0     DA       Refactoring
## -- 2021-12-03  1.1.1     DA       Refactoring
## -- 2021-12-07  1.1.2     DA       Refactoring
## -- 2021-12-09  1.1.3     DA       Class GTTraining: refactoring
## -- 2022-11-01  1.2.0     DA       Refactoring and code cleaning
## -- 2023-03-30  2.0.0     SY       Introduction as MLPro-GT-DP
## -- 2023-04-03  2.0.1     SY       Remove Potential Games and refactoring
## -- 2023-04-12  2.1.0     SY       MLPro-GT-DP to MLPro-GT-DG (Dynamic Games)
## -- 2023-05-11  2.2.0     SY       Remove PGameBoard and refactoring
## -- 2023-08-22  2.2.1     SY       Refactoring due to compatibility
## -- 2023-09-25  2.3.0     SY       Introduction of GTTrainingResults
## -- 2025-07-17  2.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.4.0 (2025-07-17) 

This module provides model classes for tasks related to Game Theory in dynamic games.
"""

from mlpro.bf.systems import State
from mlpro.rl import *



# Export list for public API
__all__ = [ 'GameBoard',
            'Player',
            'MultiPlayer',
            'Game',
            'GTTrainingResults',
            'GTTraining' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GameBoard (Environment):
    """
    Model class for a game theoretical game board. See super class for more information.
    """

    C_TYPE          = 'Game Board'
    C_REWARD_TYPE   = Reward.C_TYPE_EVERY_AGENT

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        reward = Reward(self.get_reward_type())

        for player_id in self._last_action.get_agent_ids():
            reward.add_agent_reward(player_id, self._utility_fct(p_state_new, player_id))

        return reward


## -------------------------------------------------------------------------------------------------
    def _utility_fct(self, p_state: State, p_player_id):
        """
        Computes utility of given player. To be redefined.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Player (Agent):
    """
    This class implements a game theoretical player model. See super class for more information.
    """

    C_TYPE = 'Player'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiPlayer (MultiAgent):
    """
    This class implements a game theoretical model for a team of players. See super class for more 
    information.
    """

    C_TYPE = 'Multi-Player'

## -------------------------------------------------------------------------------------------------
    def add_player(self, p_player: Player, p_weight=1.0) -> None:
        super().add_agent(p_agent=p_player, p_weight=p_weight)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Game (RLScenario):
    """
    This class implements a game consisting of a game board and a (multi-)player. See super class for 
    more information.
    """

    C_TYPE = 'Game'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTTrainingResults (RLTrainingResults):
    """
    Results of a GT training.

    """

    C_NAME = 'GT'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTTraining (RLTraining):
    """
    This class implements a standardized episodical training process. See super class for more 
    information.

    Parameters
    ----------
    p_game_cls 
        Name of GT game class, compatible to/inherited from class Game.
    p_cycle_limit : int
        Maximum number of training cycles (0=no limit). Default = 0.
    p_cycles_per_epi_limit : int
        Optional limit of cycles per episode (0=no limit, -1=get environment limit). Default = -1.    
    p_adaptation_limit : int
        Maximum number of adaptations (0=no limit). Default = 0.
    p_stagnation_limit : int
        Optional limit of consecutive evaluations without training progress. Default = 0.
    p_eval_frequency : int
        Optional evaluation frequency (0=no evaluation). Default = 0.
    p_eval_grp_size : int
        Number of evaluation episodes (eval group). Default = 0.
    p_hpt : HyperParamTuner
        Optional hyperparameter tuner (see class mlpro.bf.ml.HyperParamTuner). Default = None.
    p_hpt_trials : int
        Optional number of hyperparameter tuning trials. Default = 0. Must be > 0 if p_hpt is supplied.
    p_path : str
        Optional destination path to store training data. Default = None.
    p_collect_states : bool
        If True, the environment states will be collected. Default = True.
    p_collect_actions : bool
        If True, the agent actions will be collected. Default = True.
    p_collect_rewards : bool
        If True, the environment reward will be collected. Default = True.
    p_collect_training : bool
        If True, global training data will be collected. Default = True.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_WE.

    """

    C_NAME = 'GT'

    C_CLS_RESULTS = GTTrainingResults

## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):
        kwargs = p_kwargs.copy()
        kwargs['p_scenario_cls'] = kwargs['p_game_cls']
        kwargs.pop('p_game_cls')
        super().__init__(**kwargs)
