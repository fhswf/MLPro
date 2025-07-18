## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.dynamicgames
## -- Module  : potential.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-03  0.0.0     SY       Creation
## -- 2023-04-12  1.0.0     SY       Release of first version
## -- 2023-05-11  1.1.0     SY       Refactoring
## -- 2023-09-25  1.1.1     SY       Refactoring
## -- 2025-07-17  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-07-17) 

This module provides model classes for Potential Games in dynamic programming.
"""

from mlpro.gt.dynamicgames.basics import *



# Export list for public API
__all__ = [ 'PGameBoard' ]




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