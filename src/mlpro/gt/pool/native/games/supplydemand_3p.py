## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.pool.native.games
## -- Module  : supplydemand_3p.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-12-12  0.0.0     SY       Creation
## -- 2023-12-12  1.0.0     SY       Release of first version
## -- 2024-01-12  1.0.1     SY       Refactoring: Module Name
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18) 

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

import numpy as np

from mlpro.bf.math import Dimension, MSpace
from mlpro.bf.physics import TransferFunction
from mlpro.bf.ml import Model  

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
from mlpro.gt.pool.native.solvers.greedypolicy import MaxGreedyPolicy
         


# Export list for public API
__all__ = [ 'PayoffFunction_SD3P',
            'PayoffMatrix_SD3P',
            'TransferFunction_SD3P',
            'MaxGreedyPolicy_SD3P_P1',
            'MaxGreedyPolicy_SD3P_P2',
            'SupplyDemand_3P' ]
  
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TransferFunction_SD3P(TransferFunction):


## -------------------------------------------------------------------------------------------------
    def _set_function_parameters(self, p_args) -> bool:

        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.demand         = p_args['p_demand']
                self.price_seller_1 = p_args['p_price_seller_1']
                self.price_seller_2 = p_args['p_price_seller_2']
                self.price_seller_3 = p_args['p_price_seller_3']
                self.prod_cost      = p_args['p_prod_cost']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True


## -------------------------------------------------------------------------------------------------
    def _custom_function(self, p_input, p_range=None):

        total_production = sum(p_input)

        if total_production <= self.demand:

            profit_seller_1 = p_input[0]*self.price_seller_1[int(p_input[0]-1)]-self.prod_cost
            profit_seller_2 = p_input[1]*self.price_seller_2[int(p_input[1]-1)]-self.prod_cost
            profit_seller_3 = p_input[2]*self.price_seller_3[int(p_input[2]-1)]-self.prod_cost

            return [profit_seller_1, profit_seller_2, profit_seller_3]
        
        else:

            price_seller_1  = self.price_seller_1[int(p_input[0]-1)]
            price_seller_2  = self.price_seller_2[int(p_input[1]-1)]
            price_seller_3  = self.price_seller_3[int(p_input[2]-1)]
            list_of_prices  = [price_seller_1, price_seller_2, price_seller_3]
            
            sold_quantity   = 0
            profit_seller   = [-self.prod_cost, -self.prod_cost, -self.prod_cost]

            for x in range(len(list_of_prices)):

                if sold_quantity < self.demand:
                    min_price = min(list_of_prices)
                    idx = list_of_prices.index(min_price)

                    if (sold_quantity+p_input[idx]) <= self.demand:
                        sold_quantity += p_input[idx]
                        profit_seller[idx] += (p_input[idx]*list_of_prices[idx])
                        list_of_prices[idx] = 100
                    else:
                        quantity = self.demand-sold_quantity
                        sold_quantity += quantity
                        profit_seller[idx] += quantity*list_of_prices[idx]

            return profit_seller
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffFunction_SD3P (GTFunction):


## -------------------------------------------------------------------------------------------------
    def _setup_transfer_functions(self):

        TF = TransferFunction_SD3P(
            p_name="TransferFunction_SD3P",
            p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
            p_demand=10,
            p_price_seller_1=[15,12,9,6,3],
            p_price_seller_2=[10,8,6,4,2],
            p_price_seller_3=[8,7,6,5,4],
            p_prod_cost=5
            )

        self._add_transfer_function(p_idx=0, p_transfer_fct=TF)
        self._add_transfer_function(p_idx=1, p_transfer_fct=TF)
        self._add_transfer_function(p_idx=2, p_transfer_fct=TF)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffMatrix_SD3P (GTPayoffMatrix):


## -------------------------------------------------------------------------------------------------
    def _call_mapping(self, p_input:str, p_strategies:GTStrategy) -> float:

        self._elem_ids  = p_strategies.get_elem_ids()
        idx             = self._elem_ids.index(p_input)

        return self._function(p_input, p_strategies)[idx]


## -------------------------------------------------------------------------------------------------
    def _call_best_response(self, p_element_id:str) -> float:

        idx = self._elem_ids.index(p_element_id)
        tf  = self._function._payoff_map[idx]

        if idx == 0:
            prices = tf.price_seller_1
        elif idx == 1:
            prices = tf.price_seller_2
        elif idx == 2:
            prices = tf.price_seller_3
        
        response = []
        for x, y in enumerate(prices):
            response.append(y*(x+1))
        
        return max(response)



## -------------------------------------------------------------------------------------------------
    def _call_zero_sum(self) -> bool:
        
        return False
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MaxGreedyPolicy_SD3P_P1(MaxGreedyPolicy):


## -------------------------------------------------------------------------------------------------
    def _call_compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        
        stg_values  = np.zeros(self._strategy_space.get_num_dim())

        tf          = p_payoff._function._payoff_map[0]
        prices      = tf.price_seller_1
        
        response    = []
        for x, y in enumerate(prices):
            response.append(y*(x+1))
        
        stg_values[0] = response.index(max(response))+1
        
        return GTStrategy(self._id, self._strategy_space, stg_values)
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MaxGreedyPolicy_SD3P_P2(MaxGreedyPolicy):


## -------------------------------------------------------------------------------------------------
    def _call_compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        
        stg_values  = np.zeros(self._strategy_space.get_num_dim())

        tf      = p_payoff._function._payoff_map[1]
        prices  = tf.price_seller_2
        
        response = []
        for x, y in enumerate(prices):
            response.append(y*(x+1))
        
        stg_values[0] = response.index(max(response))+1
        
        return GTStrategy(self._id, self._strategy_space, stg_values)

         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SupplyDemand_3P (GTGame):

    C_NAME  = 'SupplyDemand_3P'


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        
        _strategy_space = MSpace()
        _strategy_space.add_dim(Dimension('RStr','Z','Strategy','','','',[1,5]))
        
        solver1 = MaxGreedyPolicy_SD3P_P1(
            p_strategy_space=_strategy_space,
            p_id=0,
            p_name="Max Greedy Solver",
            p_visualize=p_visualize,
            p_logging=p_logging
        )


        p1 = GTPlayer(
            p_solver=solver1,
            p_name="Seller 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=True
        )

        coal1 = GTCoalition(
            p_name="Coalition of Seller 1",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal1.add_player(p1)
        
        solver2 = MaxGreedyPolicy_SD3P_P2(
            p_strategy_space=_strategy_space,
            p_id=1,
            p_name="Max Greedy Solver",
            p_visualize=p_visualize,
            p_logging=p_logging
        )


        p2 = GTPlayer(
            p_solver=solver2,
            p_name="Seller 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=True
        )

        coal2 = GTCoalition(
            p_name="Coalition of Seller 2",
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
            p_name="Seller 3",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=True
        )

        coal3 = GTCoalition(
            p_name="Coalition of Seller 3",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal3.add_player(p3)

        competition = GTCompetition(
            p_name="Supply Demand 3 Sellers Competition",
            p_logging=p_logging
            )
        competition.add_coalition(coal1)
        competition.add_coalition(coal2)
        competition.add_coalition(coal3)
        
        coal_ids = competition.get_coalitions_ids()

        self._payoff = PayoffMatrix_SD3P(
            p_function=PayoffFunction_SD3P(
                p_func_type=GTFunction.C_FUNC_TRANSFER_FCTS
                ),
            p_player_ids=coal_ids
        )
        
        return competition
        
        
        

