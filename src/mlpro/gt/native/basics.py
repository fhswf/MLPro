## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.gt
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-30  0.0.0     SY       Creation
## -- 2023-??-??  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-??-??)

This module provides model classes for tasks related to a Native Game Theory.
"""


from mlpro.bf.various import *
from mlpro.bf.systems import *
from mlpro.bf.ml import *
from mlpro.bf.mt import *
from mlpro.bf.math import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTGame (Scenario):


## -------------------------------------------------------------------------------------------------
    def __init__(self):
        pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTPayoffMatrix (TStamp):


## -------------------------------------------------------------------------------------------------
    def __init__(self):
        pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTMultiPlayer (GTPlayer):


## -------------------------------------------------------------------------------------------------
    def __init__(self):
        pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTStrategy (Action):

    C_TYPE          = 'GTStrategy'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_player_id = 0, 
                 p_strategy_space : Set = None,
                 p_values: np.ndarray = None):
        
        super().__init__(p_agent_id=p_player_id,
                         p_action_space=p_strategy_space,
                         p_values=p_values)


## -------------------------------------------------------------------------------------------------
    def get_player_ids(self) -> list:
        return self.get_agent_ids()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTSolver (Task, ScientificObject):

    C_TYPE          = 'GTSolver'
    C_NAME          = '????'

    C_SCIREF_TYPE   = ScientificObject.C_SCIREF_TYPE_NONE


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_strategy_space:MSpace,
                 p_id = None,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 **p_param):
        
        Task.__init__(self,
                      p_id = p_id,
                      p_name = None,
                      p_range_max = Async.C_RANGE_PROCESS,
                      p_autorun = Task.C_AUTORUN_NONE,
                      p_class_shared = None,
                      p_visualize = p_visualize,
                      p_logging = p_logging)

        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tuple  = None
        self._init_hyperparam(**p_param)

        self._strategy_space = p_strategy_space

        try:
            self._setup_solver()
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_param):
        """
        Implementation specific hyperparameters can be added here. Please follow these steps:
        a) Add each hyperparameter as an object of type HyperParam to the internal hyperparameter
           space object self._hyperparam_space
        b) Create hyperparameter tuple and bind to self._hyperparam_tuple
        c) Set default value for each hyperparameter

        Parameters
        ----------
        p_par : Dict
            Further model specific hyperparameters, that are passed through constructor.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _setup_solver(self):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
        """
        Returns the internal hyperparameter tuple to get access to single values.
        """

        return self._hyperparam_tuple


## -------------------------------------------------------------------------------------------------
    def get_strategy_space(self) -> MSpace:
        return self._strategy_space


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.
        """

        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        
        return self._compute_strategy(p_payoff)


## -------------------------------------------------------------------------------------------------
    def _compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTPlayer (GTSolver):

    C_TYPE = 'GTPlayer'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_solver: GTSolver,
                 p_name='',
                 p_visualize:bool=True,
                 p_logging=Log.C_LOG_ALL,
                 **p_param):

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)

        self._solver = p_solver

        GTSolver.__init__(self,
                          p_strategy_space = self._solver.get_strategy_space(),
                          p_id = self._solver.get_id(),
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_param)


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_param):

        # 1 Create a dispatcher hyperparameter tuple for the player
        self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)

        # 2 Extend agent's hp space and tuple from policy
        try:
            self._hyperparam_space.append( self._solver.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            self._hyperparam_tuple.add_hp_tuple(self._solver.get_hyperparam())
        except:
            pass

        
## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        self._solver.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def get_strategy_space(self) -> MSpace:
        return self._solver.get_strategy_space()


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self._solver.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        return self._solver.compute_strategy(p_payoff)