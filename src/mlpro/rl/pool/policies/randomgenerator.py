## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.pool.policies
## -- Module  : randomgenerator
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-05-19  0.0.0     SY       Creation
## -- 2022-05-19  1.0.0     SY       Release of first version
## -- 2022-05-20  1.0.1     SY       Remove constructor and raise error for undefined boundaries
## -- 2022-09-19  1.0.2     SY       Minor improvements: False operation for integers
## -- 2022-10-08  1.0.3     SY       Bug fixing
## -- 2022-11-02  1.0.4     DA       Refactoring: method _adapt()
## -- 2023-09-21  1.0.5     SY       Typo on description
## -- 2025-07-17  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-17) 

This module provides random genarator for multi purposes, e.g. testing environment, etc..
"""

import random

import numpy as np

from mlpro.bf import ParamError
from mlpro.bf.systems import State, Action
from mlpro.rl import Policy
         
        
        
# Export list for public API
__all__ = [ 'RandomGenerator' ]



        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RandomGenerator (Policy):
    """
    A random policy that generates random actions for each dimension of the underlying action space.

    See class mlpro.rl.Policy for further details.

    """

    C_NAME      = 'RandomGenerator'

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        # 1 Create an empty numpy array
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 2 Generating random actions based on the dimension of ation space
        ids = self._action_space.get_dim_ids()
        for d in range(self._action_space.get_num_dim()):
            try:
                base_set = self._action_space.get_dim(ids[d]).get_base_set()
            except:
                raise ParamError('Mandatory base set is not defined.')
                
            try:
                if len(self._action_space.get_dim(ids[d]).get_boundaries()) == 1:
                    lower_boundaries = 0
                    upper_boundaries = self._action_space.get_dim(ids[d]).get_boundaries()[0]
                else:
                    lower_boundaries = self._action_space.get_dim(ids[d]).get_boundaries()[0]
                    upper_boundaries = self._action_space.get_dim(ids[d]).get_boundaries()[1]
                if base_set == 'Z' or base_set == 'N':
                    my_action_values[d] = random.randint(lower_boundaries, upper_boundaries)
                elif base_set == 'R' or base_set == 'DO':
                    my_action_values[d] = random.uniform(lower_boundaries, upper_boundaries)
            except:
                raise ParamError('Mandatory boundaries are not defined.')

        # 3 Return an action object with the generated random values
        return Action(self._id, self._action_space, my_action_values)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am not adapting anything!')
        return False