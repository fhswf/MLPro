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
from mlpro.bf.physics.basics import *
from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
from mlpro.gt.pool.native.solvers.greedypolicy import MaxGreedyPolicy
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TransferFunction_SD3P(TransferFunction):


## -------------------------------------------------------------------------------------------------
    def _set_function_parameters(self, p_args) -> bool:

        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.demand         = p_args['demand']
                self.price_seller_1 = p_args['price_seller_1']
                self.price_seller_2 = p_args['price_seller_2']
                self.price_seller_3 = p_args['price_seller_3']
                self.prod_cost      = p_args['prod_cost']
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
            price_seller_2  = self.price_seller_2[int(p_input[0]-1)]
            price_seller_3  = self.price_seller_3[int(p_input[0]-1)]
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

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffMatrix_SD3P (GTPayoffMatrix):


## -------------------------------------------------------------------------------------------------
    def _call_mapping(self, p_input:str, p_strategies:GTStrategy) -> float:
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _call_best_response(self, p_element_id:str) -> float:
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _call_zero_sum(self) -> bool:
        
        raise NotImplementedError
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SupplyDemand_3P (GTGame):

    C_NAME  = 'SupplyDemand_3P'


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        
        raise NotImplementedError
        
        

