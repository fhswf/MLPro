## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.pool.native.games
## -- Module  : supplydemand_3p
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-12-12  0.0.0     SY       Creation
## -- 2023-12-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-12-12)

This module provides a 3-player game of supply and demand games, where each player represents a seller.
Each seller has the capability to produce the same product with a quantity between 1-5 items everyday.
The production cost of producing 1-5 items are constant, which is 5€. Therefore, the seller needs to 
sell an item with higher price, if they produce less amount.

The tables below show the sales price based on the quantity produced by each seller:
            Seller 1                       Seller 2                       Seller 3            
    Price (€)       Quantity        Price (€)       Quantity        Price (€)       Quantity    
        15              1               10              1               8               1
        12              2               8               2               7               2
        9               3               6               3               6               3
        6               4               4               4               5               4
        3               5               2               5               4               5

The market demand of the products is 10 products/day. The buyer will always firstly buy the products
with less prices. Therefore, each player needs an individual strategy in the competitive manner to 
select the quantity of the produced products in a day in order to maximize their profit.

In this example, we are going to apply different solvers for each seller, where Seller 1 and 2 utilize
max greedy policy. This means that they always select the best possible outcomes without caring what
the other sellers are doing. Meanwhile, Seller 3 utilized random solver. In the near future, we are
going to add more solvers and this game is going to be updated accordingly.

"""

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
from mlpro.gt.pool.native.solvers.greedypolicy import MaxGreedyPolicy
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffFunction_SD3P (GTFunction):


## -------------------------------------------------------------------------------------------------
    def _setup_transfer_functions(self):

        raise NotImplementedError
        
        

