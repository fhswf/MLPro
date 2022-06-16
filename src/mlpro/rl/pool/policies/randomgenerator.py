## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.pool.policies
## -- Module  : randomgenerator
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-05-19  0.0.0     SY       Creation
## -- 2022-05-19  1.0.0     SY       Release of first version
## -- 2022-05-20  1.0.1     SY       Remove constructor and raise error for undefined boundaries
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-05-20)

This module providew random genarator for multi purposes, e.g. testing environment, etc..
"""

from mlpro.rl.models import *
import random
         
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RandomGenerator(Policy):
    """
    A random policy that generates random actions for each dimension
    of the underlying action space.

    Parameters
    ----------
    p_observation_space : MSpace     
        Subspace of an environment that is observed by the policy.
    p_action_space : MSpace
        Action space object.
    p_buffer_size : int           
        Size of internal buffer. Default = 1.
    p_ada : bool               
        Boolean switch for adaptivity. Default = True.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.
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
                if base_set == 'Z' and base_set == 'N':
                    my_action_values[d] = random.randint(lower_boundaries, upper_boundaries)
                elif base_set == 'R' or base_set == 'DO':
                    my_action_values[d] = random.uniform(lower_boundaries, upper_boundaries)
            except:
                raise ParamError('Mandatory boundaries are not defined.')

        # 3 Return an action object with the generated random values
        return Action(self._id, self._action_space, my_action_values)

## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am not adapting anything!')
        return False