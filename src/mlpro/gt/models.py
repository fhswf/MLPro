## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.gt
## -- Module  : models.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-21  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-06-25  1.0.1     DA       Class GameBoard: introduction of internal constant C_REWARD_TYPE
## -- 2021-07-01  1.0.2     DA       Class Training: removed obsolete parameter p_path
## -- 2021-08-28  1.0.3     DA       Adjustments after changings on rl models
## -- 2021-09-11  1.0.3     MRD      Change Header information to match our new library name
## -- 2021-11-16  1.1.0     DA       Refactoring
## -- 2021-12-03  1.1.1     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2021-12-03)

This module provides model classes for tasks related to cooperative Game Theory.
"""


from datetime import timedelta
from mlpro.bf.various import Log
from mlpro.rl.models import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GameBoard (Environment):
    """
    Model class for a game theoretical game board. See super class for more information.
    """

    C_TYPE              = 'Game Board'
    C_REWARD_TYPE       = Reward.C_TYPE_EVERY_AGENT

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        reward = Reward(self.get_reward_type())

        for player_id in self._last_action.get_agent_ids():
            reward.add_agent_reward(player_id, self._utility_fct(p_state_new, player_id))

        return reward


## -------------------------------------------------------------------------------------------------
    def _utility_fct(self, p_state:State, p_player_id):
        """
        Computes utility of given player. To be redefined.
        """
        
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PGameBoard (GameBoard):
    """
    Model class for a potential game theoretical game board. See super class for more information.
    """

    C_TYPE      = 'Potential Game Board'

## -------------------------------------------------------------------------------------------------
    def compute_potential(self):
        """
        Computes (weighted) potential level of the game board.
        """

        if self._last_action == None: return 0
        self.potential = 0

        for player_id in self._last_action.get_agent_ids():
            self.potential = self.potential + ( self._utility_fct(player_id) * self._last_action.get_elem(player_id).get_weight() )
        
        return self.potential





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Player (Agent):
    """
    This class implements a game theoretical player model. See super class for more information.
    """

    C_TYPE      = 'Player'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiPlayer (MultiAgent):
    """
    This class implements a game theoretical model for a team of players. See super class for more 
    information.
    """

    C_TYPE      = 'Multi-Player'

## -------------------------------------------------------------------------------------------------
    def add_player(self, p_player:Player, p_weight=1.0) -> None:
        super().add_agent(p_agent=p_player, p_weight=p_weight)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Game (RLScenario):
    """
    This class implements a game consisting of a game board and a (multi-)player. See super class for 
    more information.
    """

    C_TYPE      = 'Game'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTTraining (RLTraining): 
    """
    This class implements a standardized episodical training process. See super class for more 
    information.
    """

    C_NAME      = 'GT'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_game:Game, 
                 p_cycle_limit=0, 
                 p_max_cycles_per_episode=-1, 
                 p_max_adaptations=0, 
                 p_max_stagnations=5, 
                 p_eval_frequency=100, 
                 p_eval_grp_size=50, 
                 p_hpt:HyperParamTuner=None, 
                 p_hpt_trials=0, 
                 p_path=None, 
                 p_collect_states=True, 
                 p_collect_actions=True, 
                 p_collect_rewards=True, 
                 p_collect_training=True, 
                 p_logging=Log.C_LOG_ALL):

        super().__init__(p_scenario=p_game, 
                         p_cycle_limit=p_cycle_limit, 
                         p_max_cycles_per_episode=p_max_cycles_per_episode, 
                         p_max_adaptations=p_max_adaptations, 
                         p_max_stagnations=p_max_stagnations, 
                         p_eval_frequency=p_eval_frequency, 
                         p_eval_grp_size=p_eval_grp_size, 
                         p_hpt=p_hpt, 
                         p_hpt_trials=p_hpt_trials, 
                         p_path=p_path, 
                         p_collect_states=p_collect_states, 
                         p_collect_actions=p_collect_actions, 
                         p_collect_rewards=p_collect_rewards, 
                         p_collect_training=p_collect_training, 
                         p_logging=p_logging)