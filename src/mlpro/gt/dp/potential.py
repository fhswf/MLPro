## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.gt.dp
## -- Module  : potential.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-03  0.0.0     SY       Creation
## -- 2023-04-03  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-04-03)

This module provides model classes for Potential Games in dynamic programming.
"""

from mlpro.gt.dp.basics import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PGameBoard (GameBoard):
    """
    Model class for a potential game theoretical game board. See super class for more information.
    """

    C_TYPE = 'Potential Game Board'

## -------------------------------------------------------------------------------------------------
    def compute_potential(self):
        """
        Computes (weighted) potential level of the game board.
        """

        if self._last_action == None: return 0
        self.potential = 0

        for player_id in self._last_action.get_agent_ids():
            self.potential = self.potential + (
                        self._utility_fct(player_id) * self._last_action.get_elem(player_id).get_weight())

        return self.potential