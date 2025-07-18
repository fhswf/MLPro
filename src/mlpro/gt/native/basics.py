## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.native
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-30  0.0.0     SY       Creation
## -- 2023-12-12  1.0.0     SY       Release of first version
## -- 2023-12-27  1.0.1     SY       Adding Docstring
## -- 2025-07-17  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-17) 

This module provides model classes for tasks related to a Native Game Theory.

"""

import random
from datetime import timedelta
from typing import Union
import statistics as st

import numpy as np

from mlpro.bf import *
from mlpro.bf.various import Persistent, TStamp, ScientificObject
from mlpro.bf.data import DataStoring

from mlpro.bf.mt import *
from mlpro.bf.math import *
from mlpro.bf.physics import *
from mlpro.bf.systems import Action
from mlpro.bf.ml import *
        
        

# Export list for public API
__all__ = [ 'GTStrategy',
            'GTFunction',
            'GTPayoffMatrix',
            'GTSolver',
            'GTPlayer',
            'GTCoalition',
            'GTCompetition',
            'GTDataStoring',
            'GTGame',
            'GTTrainingResults',
            'GTTraining' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTStrategy (Action):
    """
    A class representing a strategy for a player in game theory.
    Objects of this class representations of (multi-)players. Every element
    of the internal list is related to a player, and its partial subsection.
    Strategy values for the first player can be added while object instantiation.
    Strategy values of further player can be added by using method self.add_elem().

    Parameters
    ----------
    p_player_id 
        Unique id of (first) player to be added
    p_strategy_space : Set
        Strategy space of (first) player to be added. Default = None.
    p_values : np.ndarray
        Strategy values of (first) player to be added. Default = None.
        
    """

    C_TYPE          = 'GT Strategy'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_player_id = 0, 
                 p_strategy_space:Set = None,
                 p_values:np.ndarray = None):
        
        super().__init__(p_agent_id=p_player_id,
                         p_action_space=p_strategy_space,
                         p_values=p_values)


## -------------------------------------------------------------------------------------------------
    def get_player_ids(self) -> list:
        """
        A method to get the ids of (multi-)players that have been added to this class.

        Returns
        ----------
        list
            A list of players' ids.
            
        """

        return self.get_agent_ids()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTFunction (Persistent):
    """
    A class representing a mapping functionality between strategies and payoffs in two possible
    forms, such as payoff matrices or transfer functions.

    Parameters
    ----------
    p_func_type : int 
        Type of functions, either C_FUNC_PAYOFF_MATRIX or C_FUNC_TRANSFER_FCTS.
    p_dim_elems : list
        The dimension of payoff matrix. For transfer function, this can be avoided. Default = None.
    p_num_coalisions : int
        Number of coalisions. For transfer function, this can be avoided. Default = None.
    p_logging
        Log level (see constants C_LOG_*). Default: Log.C_LOG_ALL.
        
    """

    C_TYPE                  = 'GT Function'

    C_FUNCTION_TYPE         = None
    C_FUNC_PAYOFF_MATRIX    = 0
    C_FUNC_TRANSFER_FCTS    = 1


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_func_type:int, p_dim_elems:list=None, p_num_coalisions:int=None, p_logging=Log.C_LOG_ALL):
        
        super().__init__(p_id=None, p_logging=p_logging)
        
        self.C_FUNCTION_TYPE        = p_func_type
        self._elem_ids              = None

        if self.C_FUNCTION_TYPE == self.C_FUNC_PAYOFF_MATRIX:

            if p_dim_elems is None:
                raise ParamError("p_dim_elems is not defined!")

            if p_num_coalisions is None:
                raise ParamError("p_num_coalisions is not defined!")
            
            self._num_coals         = p_num_coalisions
            dim_elems               = [self._num_coals]
            dim_elems.extend(p_dim_elems)

            self._payoff_map        = np.zeros(dim_elems)
            self._mapping_matrix    = self._setup_mapping_matrix()
            self._setup_payoff_matrix()
        
        elif self.C_FUNCTION_TYPE == self.C_FUNC_TRANSFER_FCTS:

            self._payoff_map        = {}
            self._setup_transfer_functions()


## -------------------------------------------------------------------------------------------------
    def _setup_mapping_matrix(self) -> np.ndarray:
        """
        A method to setup the mapping of the payoff matrix between strategies and payoffs.
        This is only applicable for C_FUNC_PAYOFF_MATRIX.
        This method needs to be redefined based on the setup of the game.

        Returns
        ----------
        np.ndarray
            Resulted mapping.
            
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _setup_payoff_matrix(self):
        """
        A method to setup payoff matrices. This is only applicable for C_FUNC_PAYOFF_MATRIX.
        This method needs to be redefined based on the setup of the game.
        
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _add_payoff_matrix(self, p_idx:int, p_payoff_matrix:np.ndarray):
        """
        A method to add a payoff matrix to the GTFunction class. This method is called during the
        redifinition of _setup_payoff_matrix(). This is only applicable for C_FUNC_PAYOFF_MATRIX.
 
        Parameters
        ----------
        p_idx : int
            Id of the payoff matrix in the same order as the index of the players' ids in the list.
        p_payoff_matrix : np.ndarray
            Defined payoff matrix.

        """

        if p_payoff_matrix.shape != self._payoff_map[p_idx-1].shape:
                raise ParamError("The shape between p_payoff_matrix and each element of self._payoff_map does not match!")
        
        self._payoff_map[p_idx] = p_payoff_matrix


## -------------------------------------------------------------------------------------------------
    def _setup_transfer_functions(self):
        """
        A method to setup transfer functions. This is only applicable for C_FUNC_TRANSFER_FCTS.
        This method needs to be redefined based on the setup of the game.
        
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _add_transfer_function(self, p_idx:int, p_transfer_fct:TransferFunction):
        """
        A method to add a transfer function to the GTFunction class. This method is called during
        the redifinition of _setup_transfer_functions().
        This is only applicable for C_FUNC_TRANSFER_FCTS.
 
        Parameters
        ----------
        p_idx : int
            Id of the payoff matrix in the same order as the index of the players' ids in the list.
        p_transfer_fct : TransferFunction
            Defined transfer function.

        """

        self._payoff_map[p_idx] = p_transfer_fct


## -------------------------------------------------------------------------------------------------
    def best_response(self, p_element_id:str) -> float:
        """
        A method to measure the highest possible payoff of a player/coalition in the related payoff
        map.

        Parameters
        ----------
        p_element_id : str
            Id of a specific player/coalition.

        Returns
        -------
        float
            The highest possible payoff of a player/coalition.

        """

        if self.C_FUNCTION_TYPE == self.C_FUNC_TRANSFER_FCTS:

            return 0
        
        elif self.C_FUNCTION_TYPE == self.C_FUNC_PAYOFF_MATRIX:

            if self._elem_ids is None:
                raise ParamError("self._elem_ids is None! Please instantiate this or use __call__() method!")
                
            idx = self._elem_ids.index(p_element_id)
            return np.max(self._payoff_map[idx])


## -------------------------------------------------------------------------------------------------
    def zero_sum(self) -> bool:
        """
        A method to check whether the game is a zero sum game by considering the payoff maps.

        Returns
        -------
        bool
            True means it is a zero sum game, otherwise no.

        """

        if self.C_FUNCTION_TYPE == self.C_FUNC_TRANSFER_FCTS:

            return False
        
        elif self.C_FUNCTION_TYPE == self.C_FUNC_PAYOFF_MATRIX:

            for pl in range(self._num_coals):
                if pl == 0:
                    payoff = self._payoff_map[pl]
                else:
                    payoff += self._payoff_map[pl]
            
            if np.count_nonzero(payoff) > 0:
                return False
            else:
                return True


## -------------------------------------------------------------------------------------------------
    def __call__(self, p_element_id:str, p_strategies:GTStrategy) -> float:
        """
        A method to apply the mapping between the selected strategy and the payoff, in which it
        returns the payoff for a player/coalition with respect to the selected strategy.

        Parameters
        ----------
        p_element_id : str
            Id of the player/coalition.
        p_strategies : GTStrategy
            Selected strategies by all players/coalitions.

        Returns
        -------
        float
            Payoff of the player/coalition.

        """

        if self.C_FUNCTION_TYPE == self.C_FUNC_PAYOFF_MATRIX:

            if self._elem_ids is None:
                self._elem_ids = p_strategies.get_elem_ids()

                if self._num_coals != len(self._elem_ids):
                    raise ParamError("The number of elements in p_dim_elems and p_elem_ids does not match!")
                
            idx     = self._elem_ids.index(p_element_id)
            payoff  = self._payoff_map[idx]
            
            val = p_strategies.get_sorted_values()
            
            for p, x in enumerate(self._mapping_matrix.tolist()):
                try:
                    y = self._mapping_matrix.tolist()[p].index(val.tolist())
                    return payoff[p][y]
                except:
                    pass
                    
            raise ParamError("The selected p_strategies has no matching in self._mapping_matrix!")
        

        elif self.C_FUNCTION_TYPE == self.C_FUNC_TRANSFER_FCTS:

            if self._elem_ids is None:
                self._elem_ids = p_strategies.get_elem_ids()
                
            idx = self._elem_ids.index(p_element_id)
            tf  = self._payoff_map[idx]

            el_strategy = []
            for el in self._elem_ids:
                val = p_strategies.get_elem(el).get_values()
                el_strategy.append(val)

            return tf(el_strategy)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTPayoffMatrix (TStamp, Persistent):
    """
    A class representing a payoff matrix for a set of players in game theory, where it includes
    GTFunction that has a mapping functionality.
    
    Parameters
    ----------
    p_function : GTFunction, optional
        Defined GTFunction for mapping functionality. The default is None.
    p_player_ids : list, optional
        List of players ids. The default is None.
    p_logging :
        Logging functionality. The default is Log.C_LOG_ALL.
    
    """

    C_TYPE          = 'GT Payoff Matrix'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_function:GTFunction = None,
                 p_player_ids:list = None,
                 p_logging=Log.C_LOG_ALL):
                     
        Persistent.__init__(self, p_id=None, p_logging=p_logging)
        TStamp.__init__(self)

        self._function      = p_function
        self._player_ids    = p_player_ids


## -------------------------------------------------------------------------------------------------
    def get_payoff(self, p_strategies:GTStrategy, p_element_id:str) -> float:
        """
        A method to get the payoff for a player/coalition with respect to the selected strategies.

        Parameters
        ----------
        p_strategies : GTStrategy
            Selected strategies by all players/coalitions.
        p_element_id : str
            ID of a specific player/coalition.

        Returns
        -------
        float
            Payoff value.

        """

        return self.call_mapping(p_element_id, p_strategies)


## -------------------------------------------------------------------------------------------------
    def call_mapping(self, p_input:str, p_strategies:GTStrategy) -> float:
        """
        A method to run the mapping from the payoff matrix.

        Parameters
        ----------
        p_input : str
            inputs of the payoff matrix.
        p_strategies : GTStrategy
            Selected strategies by all players/coalitions.

        Returns
        -------
        float
            Payoff of the player/coalition.

        """
        
        if self._function.C_FUNCTION_TYPE == self._function.C_FUNC_PAYOFF_MATRIX:
            return self._function(p_input, p_strategies)
        else:
            return self._call_mapping(p_input, p_strategies)


## -------------------------------------------------------------------------------------------------
    def _call_mapping(self, p_input:str, p_strategies:GTStrategy) -> float:
        """
        If the payoff matrix does not use the standardized matrix by MLPro-GT. This method can be 
        used by redefining it.

        Parameters
        ----------
        p_input : str
            inputs of the payoff matrix.
        p_strategies : GTStrategy
            Selected strategies by all players/coalitions.

        Returns
        -------
        float
            Payoff of the player/coalition.

        """
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def best_response_value(self, p_strategies:GTStrategy, p_element_id:str) -> float:
        """
        A method to calculate the gap between the payoff of the taken strategy to the best response
        value.

        Parameters
        ----------
        p_strategies : GTStrategy
            Selected strategies by all players/coalitions.
        p_element_id : str
            Id of a specific player/coalition.

        Returns
        -------
        float
            Current payoff - payoff from best response.

        """

        payoff = self.get_payoff(p_strategies, p_element_id)
        if self._function.C_FUNCTION_TYPE == self._function.C_FUNC_PAYOFF_MATRIX:
            best_payoff = self._function.best_response(p_element_id)
        else:
            best_payoff = self._call_best_response(p_element_id)

        if payoff < best_payoff:
            self.log(self.C_LOG_TYPE_I, 'Player/Coalition %s has not the best response!'%(p_element_id))
            return payoff-best_payoff
        else:
            self.log(self.C_LOG_TYPE_I, 'Player/Coalition %s has the best response!'%(p_element_id))
            return 0


## -------------------------------------------------------------------------------------------------
    def _call_best_response(self, p_element_id:str) -> float:
        """
        If the payoff matrix does not use the standardized matrix by MLPro-GT. This method can be 
        used to get the best response value by redefining it.

        Parameters
        ----------
        p_element_id : str
            Id of a specific player/coalition.

        Returns
        -------
        float
            The highest possible payoff of a player/coalition.

        """
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def zero_sum(self) -> bool:
        """
        A method to check whether the game is a zero sum game by considering the payoff maps.

        Returns
        -------
        bool
            True means it is a zero sum game, otherwise no.

        """
        
        if self._function.C_FUNCTION_TYPE == self._function.C_FUNC_PAYOFF_MATRIX:
            return self._function.zero_sum()
        else:
            return self._call_zero_sum()


## -------------------------------------------------------------------------------------------------
    def _call_zero_sum(self) -> bool:
        """
        If the payoff matrix does not use the standardized matrix by MLPro-GT. This method can be 
        used to get the detect zero sum games by redefining it.

        Returns
        -------
        bool
            True means it is a zero sum game, otherwise no.

        """
        
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTSolver (Task, ScientificObject):
    """
    A class representing a solver (policy) in game theory.
    
    Parameters
    ----------
    p_strategy_space : MSpace
        Strategy space of (first) player to be added. Default = None.
    p_id :
        Id of a player. The default is None.
    p_visualize : bool, optional
        Allowing visualization. The default is False.
    p_logging :
        Logging setup. The default is Log.C_LOG_ALL.
    **p_param :
        additional parameters related to the policy.
    
    """

    C_TYPE          = 'GT Solver'
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
        p_param : Dict
            Further model specific hyperparameters, that are passed through constructor.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _setup_solver(self):
        """
        A method to setup a solver. This needs to be redefined based on each policy, but remains
        optional.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
        """
        Returns the internal hyperparameter tuple to get access to single values.
        
        """

        return self._hyperparam_tuple


## -------------------------------------------------------------------------------------------------
    def get_strategy_space(self) -> MSpace:
        """
        A method to get the strategy space of a solver.

        Returns
        -------
        MSpace
            Strategy space.

        """
        
        return self._strategy_space


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.

        Parameters
        ----------
        p_seed :
            Seeding.
        
        """

        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        """
        A method to compute a strategy from the solver.

        Parameters
        ----------
        p_payoff : GTPayoffMatrix
            Payoff matrix of a specific player.

        Returns
        -------
        GTStrategy
            The computed strategy.

        """
        
        return self._compute_strategy(p_payoff)


## -------------------------------------------------------------------------------------------------
    def _compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        """
        A method to compute a strategy from the solver. This method needs to be redefined.

        Parameters
        ----------
        p_payoff : GTPayoffMatrix
            Payoff matrix of a specific player.

        Returns
        -------
        GTStrategy
            The computed strategy.

        """
        
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTPlayer (GTSolver):
    """
    A class representing a player in game theory with at least one specific defined solver.
    
    Parameters
    ----------
    p_solver : Union[list, GTSolver]
        A list of solvers or a solver.
    p_name :
        Name of the player. The default is ''.
    p_visualize : bool, optional
        Allowing visualization. The default is False.
    p_logging :
        Logging setup. The default is Log.C_LOG_ALL.
    p_random_solver : bool, optional
        Allowing random solver. The default is False.
    **p_param :
        Additional parameters for the player.
    
    """

    C_TYPE = 'GT Player'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_solver:Union[list, GTSolver],
                 p_name='',
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 p_random_solver:bool=False,
                 **p_param):

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)

        self._idx_solvers = 0
        if isinstance(p_solver, list):
            self._random_solver = p_random_solver
            self._list_solvers  = p_solver
        else:
            self._random_solver = False
            self._list_solvers  = [p_solver]
        self._num_solvers = len(self._list_solvers)

        self._visualize = p_visualize
        self._logging   = p_logging
        self._param     = p_param
        
        GTSolver.__init__(self,
                          p_strategy_space = self._list_solvers[0].get_strategy_space(),
                          p_id = self._list_solvers[0].get_id(),
                          p_visualize = self._visualize,
                          p_logging = self._logging,
                          **self._param)

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        
        self.switch_solver()

        
## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_param):
        """
        A method to initiate the related hyperparameters.

        Parameters
        ----------
        **p_param :
            Additional parameters for the player.

        """
        
        self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)
        
        try:
            self._hyperparam_space.append(self.get_solver().get_hyperparam().get_related_set(),
                                          p_new_dim_ids=False)
            self._hyperparam_tuple.add_hp_tuple(self.get_solver().get_hyperparam())
        except:
            pass

        
## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        """
        A metod to swith logging setup

        Parameters
        ----------
        p_logging :
            Loggin setup.

        """

        super().switch_logging(p_logging)
        self.get_solver().switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        """
        A method to set the logging level

        Parameters
        ----------
        p_level :
            Logging level.

        """

        super().set_log_level(p_level)
        self.get_solver().set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def get_strategy_space(self) -> MSpace:
        """
        A method to get the strategy space of a player.

        Returns
        -------
        MSpace
            Strategy space.

        """

        return self.get_solver().get_strategy_space()


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.

        Parameters
        ----------
        p_seed :
            Seeding.
        
        """

        self.get_solver().set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        """
        A method to compute a strategy from the solver.

        Parameters
        ----------
        p_payoff : GTPayoffMatrix
            Payoff matrix of a specific player.

        Returns
        -------
        GTStrategy
            The computed strategy.

        """

        return self.get_solver().compute_strategy(p_payoff)


## -------------------------------------------------------------------------------------------------
    def get_solver(self) -> GTSolver:
        """
        A method to get the solver of the player.

        Returns
        -------
        GTSolver
            Solver.

        """

        return self._solver


## -------------------------------------------------------------------------------------------------
    def switch_solver(self):
        """
        A method to switch the solver, if the player has multiple solvers.

        """

        if len(self._list_solvers) == 1:
            self._solver = self._list_solvers[0]
            self.log(self.C_LOG_TYPE_I, '%s is keeping the same solver %s'%(self.get_name(), self._solver.get_id()))
        else:
            if self._random_solver:
                rnd             = random.randint(0, self._num_solvers-1)
                self._solver    = self._list_solvers[rnd]
            else:
                if self._idx_solvers == self._num_solvers:
                    self._idx_solvers = 0
                self._solver    = self._list_solvers[self._idx_solvers]
                self._idx_solvers += 1
                
            p_name = self.get_name()

            GTSolver.__init__(self,
                            p_strategy_space = self._solver.get_strategy_space(),
                            p_id = self._solver.get_id(),
                            p_visualize = self._visualize,
                            p_logging = self._logging,
                            **self._param)
            
            if p_name != '':
                self.set_name(p_name)
            else:
                self.set_name(self.C_NAME)
            
            try:
                self.log(self.C_LOG_TYPE_I, 'Player %s is switching to solver %s'%(self._id, self._solver.get_name()))
            except:
                self.log(self.C_LOG_TYPE_I, 'Player %s is switching to solver %s'%(self._id, self._solver.get_id()))





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTCoalition (GTPlayer):
    """
    A class representing a colation in game theory that contains a set of players or at least one
    player.
    
    Parameters
    ----------
    p_name : str, optional
        Name of a coalition. The default is " ".
    p_coalition_type :
        Type of coalitions. The default is None.
    p_logging :
        Logging setup. The default is Log.C_LOG_ALL.
    
    """

    C_TYPE  = 'GT Coalition'
    C_NAME  = ''

    C_COALITION_STRATEGY    = None
    C_COALITION_MEAN        = 0
    C_COALITION_SUM         = 1
    C_COALITION_MIN         = 2
    C_COALITION_MAX         = 3
    C_COALITION_MEDIAN      = 4
    C_COALITION_MODE        = 5
    C_COALITION_CUSTOM      = 6


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = " ",
                 p_coalition_type = None,
                 p_logging = Log.C_LOG_ALL):

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        
        self._coop_players      = []
        self._coop_players_ids  = []

        if p_coalition_type is None:
            raise ParamError("Please add a coalition strategy!")
        else:
            self._co_strategy = p_coalition_type
        
        self.switch_logging(p_logging)

        Model.__init__(self,
                       p_ada=False,
                       p_name=p_name,
                       p_visualize=False,
                       p_logging=p_logging)

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging) -> None:
        """
        A metod to swith logging setup

        Parameters
        ----------
        p_logging :
            Loggin setup.

        """
        
        Log.switch_logging(self, p_logging=p_logging)

        for pl in self._coop_players:
            pl.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """
        A method to switch adaptivity. In native GT, this is not necessary.

        Parameters
        ----------
        p_ada : bool
            adaptivity.

        """
        pass

    
## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        """
        A method to set the logging level

        Parameters
        ----------
        p_level :
            Logging level.

        """
        
        Log.set_log_level(self, p_level)

        for pl in self._coop_players:
            pl.set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def add_player(self, p_player:GTPlayer):
        """
        A method to add a player to the coalition.

        Parameters
        ----------
        p_player : GTPlayer
            A GT player.

        """

        self._coop_players.append(p_player)
        self._coop_players_ids.append(p_player.get_id())

        self.log(Log.C_LOG_TYPE_I, p_player.get_name() + ' added.')

        if p_player.get_solver().get_hyperparam() is not None:
            self._hyperparam_space.append(p_set=p_player._solver.get_hyperparam().get_related_set(),
                                          p_new_dim_ids=False,
                                          p_ignore_duplicates=True)
        
        if self._hyperparam_tuple is None:
            self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)
            
        self._hyperparam_tuple.add_hp_tuple(p_player.get_hyperparam())
                        

## -------------------------------------------------------------------------------------------------
    def get_players(self) -> list:
        """
        A method to get a list of players in the coaltiion.

        Returns
        -------
        list
            List of GT Players.

        """
        
        return self._coop_players
                        

## -------------------------------------------------------------------------------------------------
    def get_players_ids(self) -> list:
        """
        A method to get the players' ids in the coalition.

        Returns
        -------
        list
            List of ids.

        """
        
        return self._coop_players_ids
                        

## -------------------------------------------------------------------------------------------------
    def get_player(self, p_player_id) -> GTPlayer:
        """
        A method to get the object of a specific player.

        Parameters
        ----------
        p_player_id :
            Id of a player.

        Returns
        -------
        GTPlayer
            Object of the player.

        """
        
        return self._coop_players[self._coop_players_ids.index(p_player_id)]
                        

## -------------------------------------------------------------------------------------------------
    def get_coalition_strategy(self) -> int:
        """
        A methof to get the coalition strategy.

        Returns
        -------
        int
            Coalition strategy.

        """
        
        return self._co_strategy

    
## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.

        Parameters
        ----------
        p_seed :
            Seeding.
        
        """
        
        for pl in self._coop_players:
            pl.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def get_strategy_space(self) -> ESpace:
        """
        A method to get the strategy space of the coalition.

        Returns
        -------
        ESpace
            Strategy space.

        """
        
        espace = ESpace()
        espace.add_dim(Dimension( p_name_short='CoStr', p_name_long='Coalition Strategy', p_boundaries=[-np.inf,np.inf]))
        return espace


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        """
        A method to compute a combined strategy from the players in the coalition.

        Parameters
        ----------
        p_payoff : GTPayoffMatrix
            Payoff matrix of the coalition.

        Returns
        -------
        GTStrategy
            The computed strategy.

        """

        if self.get_coalition_strategy() == self.C_COALITION_CUSTOM:

            coalition_strategy  = self._custom_coalition_strategy(p_payoff)

        else:

            for pl in self._coop_players:

                if self._coop_players.index(pl) == 0:
                    strategy_pl = pl.compute_strategy(p_payoff).get_sorted_values()
                    strategies_array = None
                    
                else:
                    if self.get_coalition_strategy() == self.C_COALITION_MEAN:
                        if strategies_array is not None:
                            strategies_array = np.append(strategies_array, strategy_pl)
                        else:
                            strategies_array = np.copy(strategy_pl)
                            
                    elif self.get_coalition_strategy() == self.C_COALITION_SUM:
                        strategy_pl += pl.compute_strategy(p_payoff).get_sorted_values()
                        
                    elif self.get_coalition_strategy() == self.C_COALITION_MIN:
                        strategy_pl = np.minimum(strategy_pl, pl.compute_strategy(p_payoff).get_sorted_values())
                    
                    elif self.get_coalition_strategy() == self.C_COALITION_MAX:
                        strategy_pl = np.maximum(strategy_pl, pl.compute_strategy(p_payoff).get_sorted_values())
                    
                    elif self.get_coalition_strategy() == self.C_COALITION_MEDIAN:
                        if strategies_array is not None:
                            strategies_array = np.append(strategies_array, strategy_pl)
                        else:
                            strategies_array = np.copy(strategy_pl)
                    
                    elif self.get_coalition_strategy() == self.C_COALITION_MODE:
                        if strategies_array is not None:
                            strategies_array = np.append(strategies_array, strategy_pl)
                        else:
                            strategies_array = np.copy(strategy_pl)
                
            if (self.get_coalition_strategy() == self.C_COALITION_MEAN) and (strategies_array is not None):
                strategy_pl = np.mean(strategies_array)
            elif (self.get_coalition_strategy() == self.C_COALITION_MEDIAN) and (strategies_array is not None):
                strategy_pl = np.median(strategies_array)
            elif (self.get_coalition_strategy() == self.C_COALITION_MODE) and (strategies_array is not None):
                strategy_pl = st.mode(strategies_array)
                
            if type(strategy_pl) is np.ndarray:
                coalition_strategy = GTStrategy(self.get_id(), self.get_strategy_space(), strategy_pl)
            else:
                coalition_strategy = GTStrategy(self.get_id(), self.get_strategy_space(), np.array([strategy_pl]))
        
        return coalition_strategy


## -------------------------------------------------------------------------------------------------
    def _custom_coalition_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        """
        A method for customizing the coalition strategy.

        Parameters
        ----------
        p_payoff : GTPayoffMatrix
            The payoff matrix of the coalition.

        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTCompetition (GTCoalition):
    """
    A class representing a competition in game theory that contains a set of coalitions. This suits
    for a competitive game.
    
    Parameters
    ----------
    p_name : str, optional
        Name of the competition. The default is " ".
    p_logging :
        Logging setup. The default is Log.C_LOG_ALL.
    
    Returns
    -------
    None.
    
    """

    C_TYPE  = 'GT Competition'
    C_NAME  = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = " ",
                 p_logging = Log.C_LOG_ALL):

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        
        self._coalitions      = []
        self._coalitions_ids  = []
        
        self.switch_logging(p_logging)
        
        super().__init__(p_name=p_name,
                         p_coalition_type=GTCoalition.C_COALITION_CUSTOM,
                         p_logging=p_logging)

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging) -> None:
        """
        A metod to swith logging setup

        Parameters
        ----------
        p_logging :
            Loggin setup.

        """
        
        Log.switch_logging(self, p_logging=p_logging)

        for coal in self._coalitions:
            coal.switch_loggin(p_logging)

    
## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        """
        A method to set the logging level

        Parameters
        ----------
        p_level :
            Logging level.

        """
        
        Log.set_log_level(self, p_level)

        for coal in self._coalitions:
            coal.set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def add_coalition(self, p_coalition:GTCoalition):
        """
        A method to add a coaltion to the competition.

        Parameters
        ----------
        p_coalition : GTCoalition
            A coalition.

        """

        self._coalitions.append(p_coalition)
        self._coalitions_ids.append(p_coalition.get_id())

        self.log(Log.C_LOG_TYPE_I, p_coalition.get_name() + ' added.')

        for coal in self.get_coalitions():
            for pl in coal.get_players():
                if pl.get_solver().get_hyperparam() is not None:
                    self._hyperparam_space.append(p_set=pl._solver.get_hyperparam().get_related_set(),
                                                  p_new_dim_ids=False,
                                                  p_ignore_duplicates=True)
                
                if self._hyperparam_tuple is None:
                    self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)
                    
                self._hyperparam_tuple.add_hp_tuple(pl.get_hyperparam())
                            

## -------------------------------------------------------------------------------------------------
    def get_coalitions(self) -> list:
        """
        A method to get the list of coalitions in the competition.

        Returns
        -------
        list
            List of coalitions.

        """
        
        return self._coalitions
                        

## -------------------------------------------------------------------------------------------------
    def get_coalitions_ids(self) -> list:
        """
        A method to get the list of coalitions' ids in the competition.

        Returns
        -------
        list
            List of coalitions' ids.

        """
        
        return self._coalitions_ids
                        

## -------------------------------------------------------------------------------------------------
    def get_coalition(self, p_coalition_id) -> GTCoalition:
        """
        A method to get the object of a specific coalition.

        Parameters
        ----------
        p_coalition_id :
            Coalition id.

        Returns
        -------
        GTCoalition
            Coalition object.

        """
        return self._coalitions[self._coalitions_ids.index(p_coalition_id)]
                        

## -------------------------------------------------------------------------------------------------
    def get_players(self) -> list:
        """
        A method to get the list of players in the competition.

        Returns
        -------
        list
            List of players.

        """

        players = []
        for coal in self._coalitions:
            players.extend(coal.get_players())

        return players
                        

## -------------------------------------------------------------------------------------------------
    def get_players_ids(self) -> list:
        """
        A method to get the list of players' ids in the competition.

        Returns
        -------
        list
            List of players' ids.

        """

        ids = []
        for coal in self._coalitions:
            ids.extend(coal.get_players_ids())

        return ids
                        

## -------------------------------------------------------------------------------------------------
    def get_player(self, p_player_id) -> GTPlayer:
        """
        A method to get the object of a player.

        Parameters
        ----------
        p_player_id :
            Player id.

        Returns
        -------
        GTPlayer
            Object of the player.

        """

        return self.get_players()[self.get_players_ids().index(p_player_id)]

    
## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.

        Parameters
        ----------
        p_seed :
            Seeding.
        
        """

        for coal in self._coalitions:
            coal.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        """
        A method to compute the strategy of each coaltiion.

        Parameters
        ----------
        p_payoff : GTPayoffMatrix
            Payoff matrices.

        Returns
        -------
        GTStrategy
            The computed strategy.

        """

        strategy = GTStrategy()

        for coal in self._coalitions:
            strategy_coal = coal.compute_strategy(p_payoff)
            
            strategy_elem = strategy_coal.get_elem(coal.get_id())
            strategy.add_elem(coal.get_id(), strategy_elem)

        return strategy


## -------------------------------------------------------------------------------------------------
    def get_strategy_space(self) -> ESpace:
        """
        A method to get the strategy space of the competition.

        Returns
        -------
        ESpace
            Strategy space.

        """
        
        espace = ESpace()
        
        for x in range(len(self.get_coalitions())):
            name_short = 'Coal' + str(x+1)
            name_long = 'Coalition' + str(x+1)
            espace.add_dim(Dimension( p_name_short=name_short, p_name_long=name_long, p_boundaries=[-np.inf,np.inf]))
        
        return espace





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTDataStoring (DataStoring):
    """
    A method for data storing of the game.

    Parameters
    ----------
    p_space : Set, optional
        Spaces. The default is None.

    """

    # Frame ID renamed
    C_VAR0          = 'Trial'

    # Variables for episodic detail data storage
    C_VAR_CYCLE     = 'Cycle'
    C_VAR_DAY       = 'Day'
    C_VAR_SEC       = 'Second'
    C_VAR_MICROSEC  = 'Microsecond'


 ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_space:Set = None):
        
        self.space = p_space

        # Initialization as an episodic detail data storage
        self.variables = [self.C_VAR_CYCLE, self.C_VAR_DAY, self.C_VAR_SEC, self.C_VAR_MICROSEC]
        self.var_space = []

        for dim_id in self.space.get_dim_ids():
            dim = self.space.get_dim(dim_id)
            self.var_space.append(dim.get_name_short())

        self.variables.extend(self.var_space)

        super().__init__(self.variables)


## -------------------------------------------------------------------------------------------------
    def get_variables(self):
        """
        A method to get variables.

        """
        
        return self.variables


## -------------------------------------------------------------------------------------------------
    def get_space(self):
        """
        A method to get space..

        """
        
        return self.space


## -------------------------------------------------------------------------------------------------
    def add_trial(self, p_trial_id):
        """
        A method to add a trial id into the data storing.

        Parameters
        ----------
        p_trial_id : 
            Trial id.

        """
        self.add_frame(p_trial_id)
        self.current_trial = p_trial_id


## -------------------------------------------------------------------------------------------------
    def memorize_row(self, p_cycle_id, p_tstamp: timedelta, p_data):
        """
        A method to add data to the data storing

        Parameters
        ----------
        p_cycle_id :
            Cyle id.
        p_tstamp : timedelta
            Time stamp.
        p_data :
            Data to be stored.

        """

        self.memorize(self.C_VAR_CYCLE, self.current_trial, p_cycle_id)
        self.memorize(self.C_VAR_DAY, self.current_trial, p_tstamp.days)
        self.memorize(self.C_VAR_SEC, self.current_trial, p_tstamp.seconds)
        self.memorize(self.C_VAR_MICROSEC, self.current_trial, p_tstamp.microseconds)

        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_trial, p_data[i])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTGame (Scenario):
    """
    A class representing a game in game theory.
    
    Parameters
    ----------
    p_mode :
        Operation mode. See bf.ops.Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_ada :
        Boolean switch for adaptivity. In the native GT, this is always switched off.
        The default is False.
    p_cycle_limit :
        Maximum number of cycles.. The default is 1.
    p_visualize : bool, optional
        Boolean switch for visualisation. The default is False.
    p_logging :
        Log level (see constants of class Log). The default is Log.C_LOG_ALL.
    
    """

    C_TYPE      = 'GT Game'
    C_NAME      = ''
    
    C_LATENCY   = timedelta(0, 1, 0)


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_ada=False,
                 p_cycle_limit=1,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL):
        
        super().__init__(p_mode=p_mode,
                         p_ada=p_ada,
                         p_cycle_limit=p_cycle_limit,
                         p_visualize=p_visualize,
                         p_logging=p_logging)
        self._strategies = None

        self.connect_data_logger()


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        """
        Custom setup of GT Game. Payoff matrix has to be defined here as self._payoff.

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM
        p_ada : bool
            Boolean switch for adaptivity.
        p_visualize : bool
            Boolean switch for visualisation. 
        p_logging
            Log level (see constants of class Log). 

        Returns
        -------
        player : GTPlayer
            GTPlayer model (object of type GTPlayer, GTCoalition or GTCompetition).
            
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):
        """
        A method to run a cycle in the defined game

        Returns
        -------
        False, False, False, False
        
        """

        self.log(self.C_LOG_TYPE_I, 'Switch solvers...')

        pl_ids = self._model.get_players()

        for pl in pl_ids:
            pl.switch_solver()

        self.log(self.C_LOG_TYPE_I, 'Compute strategies...')

        self._strategies = self._model.compute_strategy(self._payoff)

        if self._ds_strategies is not None:
            ts = self._timer.get_time()
            self._ds_strategies.memorize_row(self._cycle_id, ts, self._strategies.get_sorted_values())

        if self._ds_payoffs is not None:
            ts = self._timer.get_time()
            if isinstance(self._model, GTCompetition):
                payoff = []
                for coal in self._model.get_coalitions():
                    payoff.append(self._get_evaluation(p_coalition_id=coal.get_id(),
                                                       p_coalition=coal)
                                  )
            else:
                payoff = self._get_evaluation(p_coalition_id=self._model.get_id(),
                                              p_coalition=self._model)
            self._ds_payoffs.memorize_row(self._cycle_id, ts, payoff)

        return False, False, False, False


## -------------------------------------------------------------------------------------------------
    def _get_evaluation(self, p_coalition_id:str, p_coalition:GTCoalition) -> Union[float,list]:
        """
        A method to get the evaluation of a coalition in the form of payoff matrix.

        Parameters
        ----------
        p_coalition_id : str
            Coalition id.
        p_coalition : GTCoalition
            Coalition object.

        Returns
        -------
        Union[float,list]
            Payoff of the respective coalition.

        """
        
        self._is_bestresponse(p_coalition_id, p_coalition)
        return self._payoff.get_payoff(self._strategies, p_coalition_id)


## -------------------------------------------------------------------------------------------------
    def connect_data_logger(self,
                            p_ds_strategies:GTDataStoring = None,
                            p_ds_payoffs:GTDataStoring = None):
        """
        A method to connect connect with the data logger from GTDataStoring.

        Parameters
        ----------
        p_ds_strategies : GTDataStoring, optional
            Object of GTDataStoring of strategies. The default is None.
        p_ds_payoffs : GTDataStoring, optional
            Object of GTDataStoring of payoffs. The default is None.

        """
        
        self._ds_strategies = p_ds_strategies
        self._ds_payoffs    = p_ds_payoffs


## -------------------------------------------------------------------------------------------------
    def is_zerosum(self) -> bool:
        """
        A method to identify whether it is a zero-sum game.

        Returns
        -------
        bool
            True means zero-sum game, otherwise not.

        """
        
        return self._payoff.zero_sum()


## -------------------------------------------------------------------------------------------------
    def _is_bestresponse(self, p_coalition_id:str, p_coalition:GTCoalition) -> float:
        """
        A method to identify whether the best response value of a coaltion.
        

        Parameters
        ----------
        p_coalition_id : str
            Coalition id.
        p_coalition : GTCoalition
            Coalition object.

        Returns
        -------
        float
            The best response value.

        """

        return self._payoff.best_response_value(self._strategies, p_coalition_id)


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        """
        A method to get the latency of the game

        Returns
        -------
        timedelta
            Latency.

        """

        return self.C_LATENCY
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTTrainingResults (TrainingResults):
    """
    Results of a native GT training.

    Parameters
    ----------
    p_scenario : GTScenario
        Related native GT scenario.
    p_run : int
        Run id.
    p_cycle_id : int
        Id of first cycle of this run.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_NAME                  = 'GT Training Results'

    C_FNAME_COAL_STRATEGIES = 'stategies'
    C_FNAME_COAL_PAYOFFS    = 'payoffs'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_scenario:GTGame,
                 p_run:int,
                 p_cycle_id:int,
                 p_logging=Log.C_LOG_WE):
        super().__init__(p_scenario=p_scenario,
                         p_run=p_run,
                         p_cycle_id=p_cycle_id,
                         p_logging=p_logging)

        self.ds_strategies  = None
        self.ds_payoffs     = None
        self.num_trials     = 0


## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename='summary.csv') -> bool:
        """
        A method to save the training results

        Parameters
        ----------
        p_path :
            Saving path.
        p_filename :
            Name and format of the file. The default is 'summary.csv'.

        Returns
        -------
        bool
            True means successful, otherwise failed.

        """
        
        if not super().save(p_path, p_filename=p_filename):
            return False

        if self.ds_strategies is not None:
            self.ds_strategies.save_data(p_path, self.C_FNAME_COAL_STRATEGIES)
        if self.ds_payoffs is not None:
            self.ds_payoffs.save_data(p_path, self.C_FNAME_COAL_PAYOFFS)
            
        return True





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTTraining (Training):
    """
    Template class for a GT training.

    Parameters
    ----------
    p_game_cls 
        Name of GT game class, compatible to/inherited from class GTGame.
    p_cycle_limit : int
        Maximum number of training cycles (0=no limit). Default = 0.
    p_adaptation_limit : int
        Maximum number of adaptations (0=no limit). Default = 0.
    p_hpt : HyperParamTuner
        Optional hyperparameter tuner (see class HyperParamTuner). Default = None.
    p_hpt_trials : int
        Optional number of hyperparameter tuning trials. Default = 0.        
    p_path : str
        Optional destination path to store training data. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_WE.
    p_collect_strategy
        Collect data of selected strategies. Default = False.
    p_collect_payoff
        Collect data of obtained payoffs. Default = False.
    p_init_seed
        Seeding. Default = 0.


    """

    C_TYPE          = 'GT Training'
    C_NAME          = 'Native GT Training'

    C_CLS_RESULTS   = GTTrainingResults


## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):
        
        kwargs = p_kwargs.copy()
        kwargs['p_scenario_cls'] = kwargs['p_game_cls']
        kwargs.pop('p_game_cls')

        super().__init__(**kwargs)

        try:
            self._collect_strategy = self._kwargs['p_collect_strategy']
        except KeyError:
            self._collect_strategy = True
            self._kwargs['p_collect_strategy'] = self._collect_strategy

        try:
            self._collect_payoff = self._kwargs['p_collect_payoff']
        except KeyError:
            self._collect_payoff = True
            self._kwargs['p_collect_payoff'] = self._collect_payoff
            
        try:
            self._seed = kwargs['p_init_seed']
        except:
            self._seed = 0


## -------------------------------------------------------------------------------------------------
    def _init_results(self) -> GTTrainingResults:
        """
        A method to initialise data storing functionality.

        Returns
        -------
        GTTrainingResults
            Object of GTTrainingResults.

        """
        
        results = super()._init_results()

        if self._collect_strategy:
            results.ds_strategies = GTDataStoring(self._scenario._model.get_strategy_space())

        if self._collect_payoff:
            results.ds_payoffs = GTDataStoring(self._scenario._model.get_strategy_space())

        self._scenario.connect_data_logger(p_ds_strategies=results.ds_strategies,
                                           p_ds_payoffs=results.ds_payoffs)

        return results


## -------------------------------------------------------------------------------------------------
    def _init_trial(self):
        """
        A method to initialise a trial.

        """

        self._scenario.reset(p_seed=self._seed)
        self._seed += 1

        if (self._results.ds_strategies and self._scenario._ds_strategies) is not None:
            self._results.ds_strategies.add_trial(self._results.num_trials)

        if (self._results.ds_payoffs and self._scenario._ds_payoffs) is not None:
            self._results.ds_payoffs.add_trial(self._results.num_trials)


## -------------------------------------------------------------------------------------------------
    def _close_trial(self):
        """
        A method to close/stop a trial.

        """

        self._results.num_trials += 1


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:
        """
        A method to run a cycle.

        Returns
        -------
        bool
            False.

        """

        self._init_trial()
        self._scenario.run_cycle()
        self._close_trial()

        return False