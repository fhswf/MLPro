## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.pool.policies
## -- Module  : randomgenerator
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-05-19  0.0.0     SY       Creation
## -- 2022-05-19  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-05-19)

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
    p_seed
        Initial seed for randomizer. Default = 0.
    """

    C_NAME      = 'RandomGenerator'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_observation_space: MSpace,
                 p_action_space: MSpace,
                 p_buffer_size=1,
                 p_ada=True,
                 p_logging=Log.C_LOG_ALL,
                 p_seed=0):
        super().__init__(p_observation_space=p_observation_space,
                         p_action_space=p_action_space,
                         p_buffer_size=p_buffer_size,
                         p_ada=p_ada,
                         p_logging=p_logging)
        if p_seed == None:
            raise ParamError('Please provide seeding parameter p_seed!')
        else:
            self.set_random_seed(p_seed)

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        # 1 Create an empty numpy array
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 2 Generating random actions based on the dimension of ation space
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random()

        # 3 Return an action object with the generated random values
        return Action(self._id, self._action_space, my_action_values)

## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am not adapting anything!')
        return False