## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.gt.native
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-30  0.0.0     SY       Creation
## -- 2023-06-01  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-06-01)

This module provides model classes for tasks related to a Native Game Theory.
"""


from mlpro.bf.various import *
from mlpro.bf.systems import *
from mlpro.bf.ml import *
from mlpro.bf.mt import *
from mlpro.bf.math import *
from typing import Union





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTPayoffMatrix (TStamp):

    C_TYPE          = 'GTPayoffMatrix'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_function:Function = None,
                 p_player_ids:list = None):
        
        TStamp.__init__(self)

        self._function      = p_function
        self._player_ids    = p_player_ids


## -------------------------------------------------------------------------------------------------
    def get_payoff(self,
                   p_strategies:Union[np.ndarray, int, str, float],
                   p_player_ids:Union[str, list]=None) -> Union[float, list]:
        
        if self._function is not None:
            strategies = Element(Set(self._function._input_space))
        else:
            strategies = Element(Set())
        
        strategies.set_values(p_strategies)
        payoffs = self.call_mapping(strategies)

        if p_player_ids is None:
            return payoffs.get_values()
        elif isinstance(p_player_ids, list):
            payoff_values = payoffs.get_values()
            list_payoff = []
            for ids in range(len(p_player_ids)):
                idx = self._player_ids.index(ids)
                list_payoff.append(payoff_values[idx])
            return list_payoff
        else:
            payoff_values = payoffs.get_values()
            idx = self._player_ids.index(p_player_ids)
            return payoff_values[idx]


## -------------------------------------------------------------------------------------------------
    def call_mapping(self, p_input:Element) -> Element:
        
        if self._function is not None:
            return self._function(p_input)
        else:
            return self._call_mapping(p_input)


## -------------------------------------------------------------------------------------------------
    def _call_mapping(self, p_input:Element) -> Element:
        
        raise NotImplementedError
        
        



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTStrategy (Action):

    C_TYPE          = 'GTStrategy'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_player_id = 0, 
                 p_strategy_space: Set = None,
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
                 p_solver:GTSolver,
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
            self._hyperparam_space.append( self.get_solver().get_hyperparam().get_related_set(), p_new_dim_ids=False)
            self._hyperparam_tuple.add_hp_tuple(self.get_solver().get_hyperparam())
        except:
            pass

        
## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        self.get_solver().switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        super().set_log_level(p_level)
        self.get_solver().set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def get_strategy_space(self) -> MSpace:
        return self.get_solver().get_strategy_space()


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self.get_solver().set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        return self.get_solver().compute_strategy(p_payoff)


## -------------------------------------------------------------------------------------------------
    def get_solver(self) -> GTSolver:
        return self._solver





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTCoalition (GTPlayer):

    C_TYPE  = 'GTCoalition'
    C_NAME  = ''

    C_COALITION_STRATEGY    = None
    C_COALITION_CONCATENATE = 0
    C_COALITION_MEAN        = 1
    C_COALITION_SUM         = 2
    C_COALITION_MIN         = 3
    C_COALITION_MAX         = 4


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = " ",
                 p_coalition_type = None,
                 p_logging = Log.C_LOG_ALL):
        
        self._coop_players      = []
        self._coop_players_ids  = []

        if p_coalition_type is None:
            raise ParamError("Please add a coalition strategy!")
        else:
            self._co_strategy = p_coalition_type

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        
        self.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging) -> None:
        Log.switch_logging(self, p_logging=p_logging)

        for pl in self._coop_players:
            pl.switch_logging(p_logging)

    
## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        Log.set_log_level(self, p_level)

        for pl in self._coop_players:
            pl.set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def add_agent(self, p_player:GTPlayer):

        self._coop_players.append(p_player)
        self._coop_players_ids.append(p_player.get_id())

        self.log(Log.C_LOG_TYPE_I, p_player.C_TYPE + ' ' + p_player.get_name() + ' added.')

        if p_player.get_solver().get_hyperparam() is not None:
            self._hyperparam_space.append(p_set=p_player._solver.get_hyperparam().get_related_set(),
                                          p_new_dim_ids=False,
                                          p_ignore_duplicates=True)
        
        if self._hyperparam_tuple is None:
            self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)
            
        self._hyperparam_tuple.add_hp_tuple(p_player.get_hyperparam())
                        

## -------------------------------------------------------------------------------------------------
    def get_players(self) -> list:
        return self._coop_players
                        

## -------------------------------------------------------------------------------------------------
    def get_players_ids(self) -> list:
        return self._coop_players_ids
                        

## -------------------------------------------------------------------------------------------------
    def get_player(self, p_player_id) -> GTPlayer:
        return self._coop_players[self._coop_players_ids.index(p_player_id)]
                        

## -------------------------------------------------------------------------------------------------
    def get_coalition_strategy(self) -> int:
        return self._co_strategy

    
## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        for pl in self._coop_players:
            pl.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def get_strategy_space(self) -> ESpace:

        if self.get_coaltion_strategy == 0:
            return None
        else:
            espace = ESpace()
            espace.add_dim(Dimension( p_name_short='CoStr', p_name_long='Coalition Strategy', p_boundaries=[-np.inf,np.inf]))
            return espace


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:

        strategy = GTStrategy()

        for pl in self._coop_players:
            strategy_pl     = pl.compute_strategy(p_payoff)
            strategy_elem   = strategy_pl.get_elem(pl.get_id())
            strategy.add_elem(pl.get_id(), strategy_elem)

        if self.get_coaltion_strategy == 0:
            return strategy
        else:
            arr = strategy.get_sorted_values()

            if self.get_coaltion_strategy == 1:
                value = arr.mean()
            elif self.get_coaltion_strategy == 2:
                value = arr.sum()
            elif self.get_coaltion_strategy == 3:
                value = arr.min()
            elif self.get_coaltion_strategy == 4:
                value = arr.max()

            coalition_strategy = GTStrategy(self.get_id(), Element(self.get_strategy_space), value)
            return coalition_strategy





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTCompetition (GTCoalition):

    C_TYPE  = 'GTCompetition'
    C_NAME  = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = " ",
                 p_logging = Log.C_LOG_ALL):
        
        self._coalitions      = []
        self._coalitions_ids  = []

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        
        self.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging) -> None:
        Log.switch_logging(self, p_logging=p_logging)

        for coal in self._coalitions:
            coal.switch_loggin(p_logging)

    
## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        Log.set_log_level(self, p_level)

        for coal in self._coalitions:
            coal.set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def add_coalition(self, p_coalition:GTCoalition):

        self._coalitions.append(p_coalition)
        self._coalitions_ids.append(p_coalition.get_id())

        self.log(Log.C_LOG_TYPE_I, p_coalition.C_TYPE + ' ' + p_coalition.get_name() + ' added.')

        for coal in p_coalition.get_coalitions():
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
        return self._coalitions
                        

## -------------------------------------------------------------------------------------------------
    def get_coalitions_ids(self) -> list:
        return self._coalitions_ids
                        

## -------------------------------------------------------------------------------------------------
    def get_coalition(self, p_coalition_id) -> GTCoalition:
        return self._coalitions[self._coalitions_ids.index(p_coalition_id)]
                        

## -------------------------------------------------------------------------------------------------
    def get_players(self) -> list:

        players = []
        for coal in self._coalitions:
            players.extend(coal.get_players())

        return players
                        

## -------------------------------------------------------------------------------------------------
    def get_players_ids(self) -> list:

        ids = []
        for coal in self._coalitions:
            ids.extend(coal.get_players_ids())

        return ids
                        

## -------------------------------------------------------------------------------------------------
    def get_player(self, p_player_id) -> GTPlayer:

        return self.get_players()[self.get_players_ids().index(p_player_id)]

    
## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):

        for coal in self._coalitions:
            coal.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:

        strategy = GTStrategy()

        for coal in self._coalitions:
            strategy_coal   = coal.compute_strategy(p_payoff)
            strategy_elem   = strategy_coal.get_elem(coal.get_id())
            strategy.add_elem(coal.get_id(), strategy_elem)
        
        return strategy




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTDataStoring (DataStoring):

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
        return self.variables


## -------------------------------------------------------------------------------------------------
    def get_space(self):
        return self.space


## -------------------------------------------------------------------------------------------------
    def add_trial(self, p_trial_id):
        self.add_frame(p_trial_id)
        self.current_trial = p_trial_id


## -------------------------------------------------------------------------------------------------
    def memorize_row(self, p_cycle_id, p_tstamp: timedelta, p_data):

        self.memorize(self.C_VAR_CYCLE, self.current_trial, p_cycle_id)
        self.memorize(self.C_VAR_DAY, self.current_trial, p_tstamp.days)
        self.memorize(self.C_VAR_SEC, self.current_trial, p_tstamp.seconds)
        self.memorize(self.C_VAR_MICROSEC, self.current_trial, p_tstamp.microseconds)

        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_trial, p_data[i])



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTGame (Scenario):

    C_TYPE  = 'GTCompetition'
    C_NAME  = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_payoff_matrix:GTPayoffMatrix,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL):
        
        super().__init__(p_mode=Mode.C_MODE_SIM,
                         p_ada=False,
                         p_cycle_limit=1,
                         p_visualize=p_visualize,
                         p_logging=p_logging)
        
        self._payoff        = p_payoff_matrix
        self._strategies    = None

        self.connect_data_logger()


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        """
        Custom setup of ML scenario.

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
        ........

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        adapted : bool
            True, if something within the scenario has adapted something in this cycle. False otherwise.
        end_of_data : bool
            True, if the end of the related data source has been reached. False otherwise.
        """

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
                    payoff.append(self._get_evaluation(p_coalition_id=coal.get_id()))
            else:
                payoff = self._get_evaluation(p_coalition_id=self._model.get_id())
            self._ds_payoffs.memorize_row(self._cycle_id, ts, payoff)

        return False, False, False, False


## -------------------------------------------------------------------------------------------------
    def _get_evaluation(self, p_player_ids:Union[str, list]=None, p_coalition_id=None) -> Union[float, list]:
        
        if (p_player_ids is None) and (p_coalition_id is None):
            raise ParamError("p_player_ids and p_coalition_id are both none! Either of them needs to be defined.")

        if p_player_ids is not None:
            return self._payoff.get_payoff(self._strategies.get_sorted_values(), p_player_ids)
        else:
            if isinstance(self._model, GTCompetition):
                ids = self._model.get_coalition(p_coalition_id)
                pl_ids = ids.get_players()
            else:
                pl_ids = self._model.get_players()

            return self._payoff.get_payoff(self._strategies.get_sorted_values(), pl_ids)


## -------------------------------------------------------------------------------------------------
    def connect_data_logger(self,
                            p_ds_strategies:GTDataStoring=None,
                            p_ds_payoffs:GTDataStoring = None):
        self._ds_strategies = p_ds_strategies
        self._ds_payoffs    = p_ds_payoffs
    




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

    C_NAME                  = 'GTTrainingResults'

    C_FNAME_COAL_STRATEGIES = 'coalitions_stategies'
    C_FNAME_COAL_PAYOFFS    = 'coalitions_payoffs'


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
        if not super().save(p_path, p_filename=p_filename):
            return False

        if self.ds_strategies is not None:
            self.ds_strategies.save_data(p_path, self.C_FNAME_COAL_STRATEGIES)
        if self.ds_payoffs is not None:
            self.ds_payoffs.save_data(p_path, self.C_FNAME_COAL_PAYOFFS)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTTraining (Training):

    C_TYPE          = 'GTTraining'

    C_CLS_RESULTS   = GTTrainingResults


## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):

        super().__init__(**p_kwargs)

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


## -------------------------------------------------------------------------------------------------
    def _init_results(self) -> GTTrainingResults:
        
        results = super()._init_results()

        if self._collect_strategy:
            strategy_space = Set()
            results.ds_strategies = GTDataStoring(strategy_space)

        if self._collect_payoff:
            payoff_space = Set()
            results.ds_payoffs = GTDataStoring(payoff_space)

        self._scenario.connect_data_logger(p_ds_strategies=results.ds_strategies,
                                           p_ds_payoffs=results.ds_payoffs)

        return results


## -------------------------------------------------------------------------------------------------
    def _init_trial(self):

        self._scenario.reset()

        if (self._results.ds_strategies and self._scenario.ds_strategies) is not None:
            self._results.ds_strategies.add_trial(self._results.num_trials)

        if (self._results.ds_payoffs and self._scenario.ds_payoffs) is not None:
            self._results.ds_payoffs.add_trial(self._results.num_trials)


## -------------------------------------------------------------------------------------------------
    def _close_trial(self):

        self._results.num_trials += 1


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:

        self._init_trial()
        self._scenario.run_cycle()
        self._close_trial()

        return False