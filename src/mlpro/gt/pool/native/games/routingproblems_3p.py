## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.pool.native.games
## -- Module  : routingproblems_3p.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-01-12  0.0.0     SY       Creation
## -- 2024-01-18  1.0.0     SY       Release of first version
## -- 2024-01-28  1.0.1     SY       Refactoring
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18) 

This module provides a 3-player game of routing problems, where each player has to move simultaneously
from the starting node to the target node. We also set up this game as a congestion game, which means
that if multiple players select the same path, then the travelling time for both players are increased.

Node S represents the starting node, while node T denotes the target node. Here are the information of
the routing network:
Note -> [initial node to next node] : [x/y/z]
     -> x = the travelling time, if only one player chooses this path
     -> y = the travelling time, if two players choose this path simulateneously
     -> z = the travelling time, if three players choose this path simulateneously
    1. Node S to Node 1 : [4/6/10]
    2. Node S to Node 2 : [3/4/5]
    3. Node 1 to Node 2 : [1/2/5]
    4. Node 1 to Node 3 : [3/5/6]
    5. Node 2 to Node 3 : [4/5/6]
    6. Node 2 to Node 4 : [3/6/9]
    7. Node 3 to Node T : [2/4/6]
    8. Node 4 to Node 3 : [1/2/7]
    8. Node 4 to Node T : [2/8/10]

The main objective of each player is to reach the target points as fast as possible, while trying to
avoid taking same actions with other players. This game represents a common scenario in industries,
e.g. AGV routing plan, mobile robots, logistics, and many more.

7 potential pathways can be selected by each player, such as:
    1) S -> 1 -> 2 -> 3 -> T
    2) S -> 1 -> 3 -> T
    3) S -> 1 -> 2 -> 4 -> 3 -> T
    4) S -> 1 -> 2 -> 4 -> T
    5) S -> 2 -> 4 -> T
    6) S -> 2 -> 4 -> 3 -> T
    7) S -> 2 -> 3 -> T

In this example, we are going to apply different solvers for each player, where Player 1 utilizes 
a min greedy policy, Player 2 utilizes a combination of a min greedy policy and a random policy, and
Player 3 utilizes a random policy. In the near future, we are going to add more solvers and
this game is going to be updated accordingly.

"""

import numpy as np

from mlpro.bf.math import Dimension, MSpace
from mlpro.bf.physics import TransferFunction
from mlpro.bf.ml import Model  

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
from mlpro.gt.pool.native.solvers.greedypolicy import MinGreedyPolicy
         
        
        
# Export list for public API
__all__ = [ 'PayoffFunction_Routing3P',
            'PayoffMatrix_Routing3P',
            'TransferFunction_Routing3P',
            'MinGreedyPolicy_Routing3P',
            'Routing_3P' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TransferFunction_Routing3P(TransferFunction):

## -------------------------------------------------------------------------------------------------
    def _set_function_parameters(self, p_args) -> bool:

        return True


## -------------------------------------------------------------------------------------------------
    def _custom_function(self, p_input, p_range=None):

        path = {
            1 : ['S','1','2','3','T'],
            2 : ['S','1','3','T'],
            3 : ['S','1','2','4','3','T'],
            4 : ['S','1','2','4','T'],
            5 : ['S','2','4','T'],
            6 : ['S','2','4','3','T'],
            7 : ['S','2','3','T']
        }
        
        time_matrix = {
            'S_1' : [4,6,10],
            'S_2' : [3,4,5],
            '1_2' : [1,2,5],
            '1_3' : [3,5,6],
            '2_3' : [4,5,6],
            '2_4' : [3,6,9],
            '3_T' : [2,4,6],
            '4_3' : [1,2,7],
            '4_T' : [2,8,10]
        }
        
        time = [0, 0, 0]
        reached = [False, False, False]
        p_input = [int(x) for x in p_input]
        
        n_iter = max(max(len(path[p_input[0]]), len(path[p_input[1]])), len(path[p_input[2]]))
        for x in range(n_iter):
            if x != 0:
                for pl in range(3):
                    if reached[pl] is False:
                        pl_path = path[p_input[pl]]
                        if pl == 0:
                            n1_path = path[p_input[1]]
                            n2_path = path[p_input[2]]
                            n1_reached = reached[1]
                            n2_reached = reached[2]
                        elif pl == 1:
                            n1_path = path[p_input[0]]
                            n2_path = path[p_input[2]]
                            n1_reached = reached[0]
                            n2_reached = reached[2]
                        elif pl == 2:
                            n1_path = path[p_input[0]]
                            n2_path = path[p_input[1]] 
                            n1_reached = reached[0]
                            n2_reached = reached[1]               
                        str_time = pl_path[x-1]+'_'+pl_path[x]

                        if n1_reached and n2_reached:
                            time[pl] += time_matrix[str_time][0]
                        elif (not n1_reached) and n2_reached:
                            if (pl_path[x-1]==n1_path[x-1]) and (pl_path[x]==n1_path[x]):
                                time[pl] += time_matrix[str_time][1]
                            else:
                                time[pl] += time_matrix[str_time][0]
                        elif (not n2_reached) and n1_reached:
                            if (pl_path[x-1]==n2_path[x-1]) and (pl_path[x]==n2_path[x]):
                                time[pl] += time_matrix[str_time][1]
                            else:
                                time[pl] += time_matrix[str_time][0]
                        else:
                            if (pl_path[x-1]==n1_path[x-1]==n2_path[x-1]) and (pl_path[x]==n1_path[x]==n2_path[x]):
                                time[pl] += time_matrix[str_time][2]
                            elif (pl_path[x-1]==n1_path[x-1]) and (pl_path[x]==n1_path[x]):
                                time[pl] += time_matrix[str_time][1]
                            elif (pl_path[x-1]==n2_path[x-1]) and (pl_path[x]==n2_path[x]):
                                time[pl] += time_matrix[str_time][1]
                            else:
                                time[pl] += time_matrix[str_time][0]
                                
                for pl in range(3):
                    if reached[pl] is False:
                        pl_path = path[p_input[pl]]
                        if pl_path[x] == 'T':
                            reached[pl] = True

        return time
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffFunction_Routing3P(GTFunction):

## -------------------------------------------------------------------------------------------------
    def _setup_transfer_functions(self):

        TF = TransferFunction_Routing3P(
            p_name="TransferFunction_Routing3P",
            p_type=TransferFunction.C_TRF_FUNC_CUSTOM
        )

        self._add_transfer_function(p_idx=0, p_transfer_fct=TF)
        self._add_transfer_function(p_idx=1, p_transfer_fct=TF)
        self._add_transfer_function(p_idx=2, p_transfer_fct=TF)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffMatrix_Routing3P(GTPayoffMatrix):

## -------------------------------------------------------------------------------------------------
    def _call_mapping(self, p_input:str, p_strategies:GTStrategy) -> float:

        self._elem_ids  = p_strategies.get_elem_ids()
        idx             = self._elem_ids.index(p_input)

        return self._function(p_input, p_strategies)[idx]


## -------------------------------------------------------------------------------------------------
    def _call_best_response(self, p_element_id:str) -> float:

        # S -> 2 -> 4 -> T without congestion
        return 8


## -------------------------------------------------------------------------------------------------
    def _call_zero_sum(self) -> bool:
        
        return False
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MinGreedyPolicy_Routing3P(MinGreedyPolicy):

## -------------------------------------------------------------------------------------------------
    def _call_compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        
        # S -> 2 -> 4 -> T without congestion
        stg_values  = np.zeros(self._strategy_space.get_num_dim())
        stg_values[0] = 5
        
        return GTStrategy(self._id, self._strategy_space, stg_values)

         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Routing_3P (GTGame):

    C_NAME  = 'Routing_3P'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        
        _strategy_space = MSpace()
        _strategy_space.add_dim(Dimension('RStr','Z','Strategy','','','',[1,7]))
        
        solver1 = MinGreedyPolicy_Routing3P(
            p_strategy_space=_strategy_space,
            p_id=0,
            p_name="Min Greedy Solver",
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        p1 = GTPlayer(
            p_solver=solver1,
            p_name="Player 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=True
        )

        coal1 = GTCoalition(
            p_name="Coalition of Player 1",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal1.add_player(p1)
        
        solver2a = MinGreedyPolicy_Routing3P(
            p_strategy_space=_strategy_space,
            p_id=1,
            p_name="Min Greedy Solver",
            p_visualize=p_visualize,
            p_logging=p_logging
        )
        
        solver2b = RandomSolver(
            p_strategy_space=_strategy_space,
            p_id=1,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        p2 = GTPlayer(
            p_solver=[solver2a, solver2b],
            p_name="Player 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=True
        )

        coal2 = GTCoalition(
            p_name="Coalition of Player 2",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal2.add_player(p2)
        
        solver3 = RandomSolver(
            p_strategy_space=_strategy_space,
            p_id=2,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        p3 = GTPlayer(
            p_solver=solver3,
            p_name="Player 3",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=True
        )

        coal3 = GTCoalition(
            p_name="Coalition of Player 3",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal3.add_player(p3)

        competition = GTCompetition(
            p_name="3P Routing Competition",
            p_logging=p_logging
            )
        competition.add_coalition(coal1)
        competition.add_coalition(coal2)
        competition.add_coalition(coal3)
        
        coal_ids = competition.get_coalitions_ids()

        self._payoff = PayoffMatrix_Routing3P(
            p_function=PayoffFunction_Routing3P(
                p_func_type=GTFunction.C_FUNC_TRANSFER_FCTS
                ),
            p_player_ids=coal_ids
        )
        
        return competition