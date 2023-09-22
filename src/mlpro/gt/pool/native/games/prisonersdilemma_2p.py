## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.pool.native.games
## -- Module  : prisonersdilemma_2p
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-21  0.0.0     SY       Creation
## -- 2023-09-22  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-09-22)

This module provides a 2-player game of Prisoners' Dilemma with random solver. In the near future,
we are going to add more solvers and this howto is going to be updated accordingly.
"""

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffFunction_PD2P (GTFunction):


## -------------------------------------------------------------------------------------------------
    def _setup_payoff_matrix(self):

        self._add_payoff_matrix(
            p_idx=1,
            p_payoff_matrix=np.array([[2, 8], [1, 5]])
        )

        self._add_payoff_matrix(
            p_idx=2,
            p_payoff_matrix=np.array([[2, 1], [8, 5]])
        )
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PrisonersDilemma2PGame (GTGame):

    C_NAME  = 'PrisonersDilemma2PGame'


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:

        solver1 = RandomSolver(
            p_strategy_space=MSpace().add_dim('RStr','Z','Random Strategy','','','',[0,1]),
            p_id=1,
            p_visualize=p_visualize,
            p_logging=p_logging
        )


        p1 = GTPlayer(
            p_solver=solver1,
            p_name="Player of Prisoner 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        coal1 = GTCoalition(
            p_name="Coalition of Prisoner 1",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal1.add_player(p1)


        solver2 = RandomSolver(
            p_strategy_space=MSpace().add_dim('RStr','Z','Random Strategy','','','',[0,1]),
            p_id=2,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        p2 = GTPlayer(
            p_solver=solver2,
            p_name="Player of Prisoner 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        coal2 = GTCoalition(
            p_name="Coalition of Prisoner 2",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal2.add_player(p2)


        competition = GTCompetition(
            p_name="Prisoner's Dilemma Competition",
            p_logging=p_logging
            )
        competition.add_coalition(coal1)
        competition.add_coalition(coal2)
        
        return competition
        



