## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : ml.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-20  0.0.0     DA       Creation 
## -- 2021-08-25  1.0.0     DA       Release of first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-10-06  1.0.1     DA       Extended class Adaptive by new methods _adapt(), get_adapted(),
## --                                _set_adapted(); moved Buffer classes to mlpro.bf.data.py
## -- 2021-10-25  1.0.2     SY       Enhancement of class Adaptive by adding ScientificObject.
## -- 2021-10-26  1.1.0     DA       New class AdaptiveFunction
## -- 2021-10-29  1.1.1     DA       New method Adaptive.set_random_seed()
## -- 2021-11-15  1.2.0     DA       - Class Adaptive renamed to Model
## --                                - New classes Mode, Scenario, TrainingResults, Training, 
## --                                  HyperParamTuner
## -- 2021-11-30  1.2.1     DA       - Classes Model, AdaptiveFunction: new opt. parameters **p_par
## --                                - Docstrings reformatted to numpy style
## --                                - Class Model: new method get_maturity()
## -- 2021-12-07  1.2.2     DA       - Method Scenario.__init__(): param p_cycle_len removed
## --                                - Method Training.__init__(): par p_scenario replaced by p_scenario_cls
## -- 2021-12-08  1.2.3     DA       Moved class AdaptiveFunction to new subtopic package sl
## -- 2021-12-09  1.2.4     DA       Class Training: introduced dynamic parameters **p_kwargs
## -- 2021-12-10  1.2.5     DA       Class HyperParamTuner: changed interface of method maximize()
## -- 2021-12-12  1.3.0     DA       Classes Scenario, Training, TrainingResults: introduced number
## --                                of adapatations
## -- 2021-12-21  1.3.1     DA       - Minor changes on class Training
## --                                - Added log functionality to class TrainingResults
## -- 2022-01-18  1.3.2     MRD      Small optimize on Scenario instantiation in Training class
## --                                Put the self._cycle_limit directly on the parameter argument
## -- 2022-01-27  1.3.3     SY       Class Training: enhanced training with hyperparameter tuning
## -- 2022-01-28  1.3.4     SY       Class HyperParamTuner: add save(), save_line(), HPDataStoring
## -- 2022-02-24  1.3.5     SY       Introduce new class HyperParamDispatcher
## -- 2022-03-02  1.3.6     SY       Refactoring class HyperParamDispatcher
## -- 2022-03-02  1.3.7     DA       Class HyperParamDispatcher:correction of method set_values()
## -- 2022-06-06  1.3.8     MRD      Add additional parameter to Training class, p_env_mode for
## --                                setting up environment mode
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.8 (2022-06-06)
This module provides fundamental machine learning templates, functionalities and properties.
"""


import sys
from typing import Dict
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.data import Buffer
from mlpro.bf.plot import *
import random




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParam (Dimension):
    """
    Hyperparameter definition class. See class Dimension for further descriptions.
    """

## -------------------------------------------------------------------------------------------------
    def register_callback(self, p_cb):
        self._cb = p_cb


## -------------------------------------------------------------------------------------------------
    def callback_on_change(self, p_value):
        try:
            self._cb(p_value)
        except:
            pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamSpace (ESpace):
    """
    Hyperparameter space, which is just an Euclidian space.
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamTuple (Element):
    """
    Tuple of hyperparameters, which is an element of a hyperparameter space
    """

## -------------------------------------------------------------------------------------------------
    def set_value(self, p_dim_id, p_value):
        super().set_value(p_dim_id, p_value)
        self._set.get_dim(p_dim_id).callback_on_change(p_value)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamDispatcher (HyperParamTuple):
    """
    To dispatch multiple hp tuples into one tuple
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_set: Set) -> None:
        super().__init__(p_set)
        self._hp_dict = {}


## -------------------------------------------------------------------------------------------------
    def add_hp_tuple(self, p_hpt:HyperParamTuple):
        for idx in p_hpt.get_dim_ids():
            self._hp_dict[idx] = p_hpt


## -------------------------------------------------------------------------------------------------
    def get_value(self, p_dim_id):
        return self._hp_dict.get(p_dim_id).get_value(p_dim_id)
   

## -------------------------------------------------------------------------------------------------
    def set_value(self, p_dim_id, p_value):
        self._hp_dict.get(p_dim_id).set_value(p_dim_id, p_value)


## -------------------------------------------------------------------------------------------------
    def get_values(self):
        for idx, dim_id in enumerate(self._set.get_dim_ids()):
            self._values[idx] = self.get_value(dim_id)
        return self._values


## -------------------------------------------------------------------------------------------------
    def set_values(self, p_values):
        for idx, dim_id in enumerate(self._set.get_dim_ids()):
            self.set_value(dim_id, p_values[idx])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Model (Log, LoadSave, Plottable, ScientificObject):
    """
    Fundamental template class for adaptive ML models. Supports especially
      - Adaptivity
      - Data buffering
      - Hyperparameter management
      - Plotting
      - Scientific referencing on source code level

    Parameters
    ----------
    p_buffer_size : int
        Initial size of internal data buffer. Defaut = 0 (no buffering).
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_par : Dict
        Futher model specific parameters (to be defined in chhild class).

    """

    C_TYPE          = 'Model'
    C_NAME          = '????'

    C_BUFFER_CLS    = Buffer       

    C_SCIREF_TYPE   = ScientificObject.C_SCIREF_TYPE_NONE     

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_buffer_size=0, p_ada=True, p_logging=Log.C_LOG_ALL, **p_par):  

        Log.__init__(self, p_logging=p_logging)
        self._adapted           = False
        self.switch_adaptivity(p_ada)
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tuple  = None
        self._init_hyperparam(**p_par)

        if p_buffer_size > 0:
            self._buffer = self.C_BUFFER_CLS(p_size=p_buffer_size)
        else:
            self._buffer = None


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):
        """
        Implementation specific hyperparameters can be added here. Please follow these steps:
        a) Add each hyperparameter as an object of type HyperParam to the internal hyperparameter
           space object self._hyperparam_space
        b) Create hyperparameter tuple and bind to self._hyperparam_tuple
        c) Set default value for each hyperparameter

        Parameters
        ----------
        p_par : Dict
            Futher model specific parameters, that are passed through constructor.

        """

        pass


## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
        """
        Returns the internal hyperparameter tuple to get access to single values.
        """

        return self._hyperparam_tuple


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """
        Switches adaption functionality on/off.
        
        Parameters
        ----------
        p_ada : bool
            Boolean switch for adaptivity

        """

        self._adaptivity = p_ada
        if self._adaptivity:
            self.log(self.C_LOG_TYPE_I, 'Adaptivity switched on')
        else:
            self.log(self.C_LOG_TYPE_I, 'Adaptivity switched off')


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.
        """

        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        """
        Returns True, if the model was adapted at least once. False otherwise.
        """

        return self._adapted


## -------------------------------------------------------------------------------------------------
    def _set_adapted(self, p_adapted:bool):
        """
        Sets the adapted flag. Internal use only.
        """

        self._adapted = p_adapted


## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Adapts the model by calling the custom method _adapt().

        Parameters
        ----------
        p_args
            All parameters that are needed for the adaption. Depends on the specific higher context.

        Returns
        -------
        bool
            True, if something has been adapted. False otherwise.

        """

        if not self._adaptivity: return False
        self.log(self.C_LOG_TYPE_I, 'Adaptation started')
        adapted = self._adapt(*p_args)
        self._set_adapted(adapted)
        return adapted
        

## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Custom implementation of the adaptation algorithm. Please describe the type and purpose of 
        all parameters needed by your implementation. This method will be called by public method 
        adapt() if adaptivity is switched on. 

        Parameters
        ----------
        p_args[0]           
            ...
        p_args[1]           
            ...

        Returns
        -------
        bool
            True, if something has been adapted. False otherwise.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        """
        Clears internal buffer (if buffering is active).
        """

        if self._buffer is not None: self._buffer.clear()


## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        """
        Computes the maturity of the model.

        Returns
        -------
        float
            Maturity of the model as a scalar value in interval [0,1]
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Mode (Log):
    """
    Property class that adds a mode and related methods to a child class.

    Parameters
    ----------
    p_mode
        Operation mode. Valid values are stored in constant C_VALID_MODES.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_MODE_INITIAL  = -1
    C_MODE_SIM      = 0
    C_MODE_REAL     = 1

    C_VALID_MODES   = [ C_MODE_SIM, C_MODE_REAL ]

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mode, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)
        self._mode = self.C_MODE_INITIAL
        self.set_mode(p_mode)


## -------------------------------------------------------------------------------------------------
    def get_mode(self):
        """
        Returns current mode.
        """

        return self._mode


## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):
        """
        Sets new mode.

        Parameters
        ----------
        p_mode
            Operation mode. Valid values are stored in constant C_VALID_MODES.

        """

        if not p_mode in self.C_VALID_MODES: raise ParamError('Invalid mode')
        if self._mode == p_mode: return
        self._mode = p_mode
        self.log(self.C_LOG_TYPE_I, 'Operation mode set to', self._mode)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Scenario (Mode, LoadSave, Plottable):
    """
    Template class for a common ML scenario with an adaptive model inside. To be inherited and 
    specialized in higher ML subtopic layers.
    
    The following key features are included:
      - Operation mode
      - Cycle management
      - Timer
      - Latency 
      - Explicit handling of an adaptive ML model inside

    Parameters
    ----------
    p_mode
        Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_ada : bool
        Boolean switch for adaptivity. Default = True.
    p_cycle_limit : int
        Maximum number of cycles. Default = 0 (no limit).
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    
    """

    C_TYPE      = 'Scenario'
    C_NAME      = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_mode=Mode.C_MODE_SIM,       
                 p_ada:bool=True,               
                 p_cycle_limit=0,              
                 p_visualize=True,              
                 p_logging=Log.C_LOG_ALL ):    

        # 0 Intro
        self._cycle_max     = sys.maxsize
        self._cycle_id      = 0
        self._visualize     = p_visualize
        self.set_cycle_limit(p_cycle_limit)


        # 1 Setup entire scenario
        self._model = self._setup(p_mode=p_mode, p_ada=p_ada, p_logging=p_logging)
        if self._model is None: 
            raise ImplementationError('Please return your ML model in method self._setup()')

        super().__init__(p_mode, p_logging)


        # 2 Init timer
        if self.get_mode() == Mode.C_MODE_SIM:
            t_mode = Timer.C_MODE_VIRTUAL
        else:
            t_mode = Timer.C_MODE_REAL

        self._cycle_len = self.get_latency()
        self._timer     = Timer(t_mode, self._cycle_len, self._cycle_limit)


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        self._model.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_logging) -> Model:
        """
        Custom setup of ML scenario.

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM
        p_ada : bool
            Boolean switch for adaptivity.
        p_logging
            Log level (see constants of class Log). 

        Returns
        -------
        Model
            Adaptive model inside the ML scenario
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        self._model.init_plot(p_figure=p_figure)


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        self._model.update_plot()


## -------------------------------------------------------------------------------------------------
    def get_model(self) -> Model:
        """
        Returns the adaptive model object inside the scenario.
        """

        return self._model


## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):
        super().set_mode(p_mode)
        self._set_mode(p_mode)


## -------------------------------------------------------------------------------------------------
    def _set_mode(self, p_mode):
        """
        Redefine this method to switch the scenario between simulation or real operation mode.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        """
        Returns the latency of the scenario. To be implemented in child class.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def set_cycle_limit(self, p_limit):
        self._cycle_limit = p_limit


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=1):
        """
        Resets the scenario and especially the ML model inside. Internal random generators are seed 
        with the given value. Custom reset actions can be implemented in method _reset().

        Parameters
        ----------
        p_seed : int          
            Seed value for internal random generator

        """

        # 0 Intro
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Scenario reset with seed', str(p_seed))

        # 1 Internal ML model reset
        self._model.set_random_seed(p_seed)
        if self._visualize: self._model.init_plot()

        # 2 Custom reset of further scenario-specific components
        self._reset(p_seed)

        # 3 Timer reset
        self._timer.reset()

        # 4 Cycle counter reset
        self._cycle_id = 0


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Custom method to reset further components of the scenario (not the ML model itself) and to 
        set the given random seed value. See method reset() for further details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def run_cycle(self):
        """
        Runs a single process cycle.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        timeout : bool
            True on timeout. False otherwise.
        cycle_limit : bool
            True, if cycle limit has reached. False otherwise.
        adapted : bool
            True, if ml model adapted something in this cycle. False otherwise.

        """

        # 1 Run a single custom cycle
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Start of cycle', str(self._cycle_id))
        success, error, adapted = self._run_cycle()
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': End of cycle', str(self._cycle_id), '\n')


        # 2 Update visualization
        if self._visualize:
            self.update_plot()


        # 3 Update cycle id and check for optional limit
        if ( self._cycle_limit > 0 ) and ( self._cycle_id >= ( self._cycle_limit -1 ) ): 
            limit = True
        else:
            self._cycle_id = ( self._cycle_id + 1 ) & self._cycle_max
            limit = False


        # 4 Wait for next cycle (real mode only)
        if ( self._timer.finish_lap() == False ) and ( self._cycle_len is not None ):
            self.log(self.C_LOG_TYPE_W, 'Process time', self._timer.get_time(), ': Process timed out !!!')
            timeout = True
        else:
            timeout = False


        # 5 Return result of custom cycle execution
        return success, error, timeout, limit, adapted


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):
        """
        Custom implementation of a single process cycle. To be redefined.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        adapted : bool
            True, if ml model adapted something in this cycle. False otherwise.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_cycle_id(self):
        """
        Returns current cycle id.
        """

        return self._cycle_id


## -------------------------------------------------------------------------------------------------
    def run(self, 
            p_term_on_success:bool=True,        
            p_term_on_error:bool=True,          
            p_term_on_timeout:bool=False ):    
        """
        Runs the scenario as a sequence of single process steps until a terminating event occures.

        Parameters
        ----------
        p_term_on_success : bool
            If True, the run terminates on success. Default = True.
        p_term_on_error : bool
            If True, the run terminates on error. Default = True.
        p_term_on_timeout : bool
            If True, the run terminates on timeout. Default = False.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        timeout : bool
            True on timeout. False otherwise.
        cycle_limit : bool
            True, if cycle limit has reached. False otherwise.
        adapted : bool
            True, if ml model adapted something. False otherwise.
        num_cycles: int
            Number of cycles.

        """

        self._cycle_id  = 0
        adapted         = False
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), 'Start of processing')

        while True:
            success, error, timeout, limit, adapted_cycle = self.run_cycle()
            adapted = adapted or adapted_cycle
            if p_term_on_success and success: break
            if p_term_on_error and error: break
            if p_term_on_timeout and timeout: break
            if limit: break

        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), 'End of processing')

        return success, error, timeout, limit, adapted, self._cycle_id





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TrainingResults (Log, Saveable):
    """
    Results of a training (see class Training).

    Parameters
    ----------
    p_scenario : Scenario
        Related scenario.
    p_run : int
        Run id.
    p_cycle_id : int
        Id of first cycle of this run.
    p_path : str
        Optional estination path to store the results.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_TYPE      = 'Results '

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scenario:Scenario, p_run, p_cycle_id, p_path=None, p_logging=Log.C_LOG_WE):
        self.scenario           = p_scenario
        self.run                = p_run
        self.ts_start           = datetime.now()
        self.ts_end             = None
        self.duration           = 0
        self.cycle_id_start     = p_cycle_id
        self.cycle_id_end       = 0
        self.num_cycles         = 0
        self.num_cycles_train   = 0
        self.num_cycles_eval    = 0
        self.num_adaptations    = 0
        self.highscore          = None
        self.path               = p_path

        self.custom_results     = []

        Log.__init__(self, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def add_custom_result(self, p_name, p_value):
        self.custom_results.append([p_name, p_value])


## -------------------------------------------------------------------------------------------------
    def close(self):
        self.ts_end             = datetime.now()
        self.duration           = self.ts_end - self.ts_start
        self.cycle_id_end       = self.cycle_id_start + self.num_cycles -1


## -------------------------------------------------------------------------------------------------
    def log_results(self):
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
        self.log(self.C_LOG_TYPE_W, '-- Training Results of run', self.run)
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
        self._log_results()
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')


## -------------------------------------------------------------------------------------------------
    def _log_results(self):
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
        self.log(self.C_LOG_TYPE_W, '-- Scenario          :', self.scenario.C_TYPE, self.scenario.C_NAME)
        self.log(self.C_LOG_TYPE_W, '-- Model             :', self.scenario.get_model().C_TYPE, self.scenario.get_model().C_NAME)
        self.log(self.C_LOG_TYPE_W, '-- Start time stamp  :', self.ts_start)
        self.log(self.C_LOG_TYPE_W, '-- End time stamp    :', self.ts_end)
        self.log(self.C_LOG_TYPE_W, '-- Duration          :', self.duration)
        self.log(self.C_LOG_TYPE_W, '-- Start cycle id    :', self.cycle_id_start)
        self.log(self.C_LOG_TYPE_W, '-- End cycle id      :', self.cycle_id_end)
        self.log(self.C_LOG_TYPE_W, '-- Training cycles   :', self.num_cycles_train)
        self.log(self.C_LOG_TYPE_W, '-- Evaluation cycles :', self.num_cycles_eval)
        self.log(self.C_LOG_TYPE_W, '-- Adaptations       :', self.num_adaptations)
        self.log(self.C_LOG_TYPE_W, '-- High score        :', self.highscore)
        if self.path is not None:
            self.log(self.C_LOG_TYPE_W, '-- Results stored in : "' + self.path +'"')


## -------------------------------------------------------------------------------------------------
    def _save_line(self, p_file, p_name, p_value):
        value = p_value
        if value is None: value = '-'
        p_file.write(p_name + '\t' + str(value) + '\n')


## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename='summary.csv') -> bool:
        """
        Saves a training summary in the given path.

        Parameters
        ----------
        p_path : str
            Destination folder
        p_filename  :string
            Name of summary file. Default = 'summary.csv'

        Returns
        -------
        success : bool
            True, if summary file was created successfully. False otherwise.

        """

        filename = p_path + os.sep + p_filename
        filename.replace(os.sep + os.sep, os.sep)

        file = open(filename, 'wt')
        if file is None: return False

        self._save_line(file, 'Scenario', '"' + self.scenario.C_TYPE + ' ' + self.scenario.C_NAME + '"')
        self._save_line(file, 'Model', '"' + self.scenario.get_model().C_TYPE + ' ' + self.scenario.get_model().C_NAME + '"')
        self._save_line(file, 'Run', str(self.run))
        self._save_line(file, 'Start', self.ts_start)
        self._save_line(file, 'End', self.ts_end)
        self._save_line(file, 'Duration', self.duration)
        self._save_line(file, 'Start cycle id', self.cycle_id_start)
        self._save_line(file, 'End cycle id', self.cycle_id_end)
        self._save_line(file, 'Training cycles', self.num_cycles_train)
        self._save_line(file, 'Evaluation cycles', self.num_cycles_eval)
        self._save_line(file, 'Adaptations', self.num_adaptations)
        self._save_line(file, 'Highscore', self.highscore)
        self._save_line(file, 'Path', '"' + str(self.path) + '"')

        for name, value in self.custom_results:
            self._save_line(file, name, value)

        file.close()
        return True





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamTuner (Log, Saveable):
    """
    Template class for hyperparameter tuning (HPT).
    """

    C_TYPE      = 'HyperParam Tuner'
    C_NAME      = '????'
    C_VAR_TRIAL = 'Trial'
    C_VAR_SCORE = 'Highscore'

## -------------------------------------------------------------------------------------------------
    def maximize(self, p_training_cls, p_num_trials, p_root_path, **p_training_param ) -> TrainingResults:
        """
        ...

        Parameters
        ----------
        p_training_cls
            Training class to be instantiated/executed 
        p_num_trials : int    
            Number of trials
        p_num_trials : str    
            Root path of the training class
        p_training_param : dictionary
            Training parameters

        Returns
        -------
        results : TrainingResults
            Training results of the best tuned model (see class TrainingResults).

        """

        self._training_cls      = p_training_cls
        self._num_trials        = p_num_trials
        self._root_path         = p_root_path
        self._training_param    = p_training_param
        self.HPDataStoring      = None
        self.variables          = [self.C_VAR_TRIAL, self.C_VAR_SCORE]

        return self._maximize()


## -------------------------------------------------------------------------------------------------
    def _maximize(self) -> TrainingResults:
        raise NotImplementedError
    
## -------------------------------------------------------------------------------------------------
    def _save_line(self, p_file, p_name, p_value):
        value = p_value
        if value is None: value = '-'
        p_file.write(p_name + '\t' + str(value) + '\n')


## -------------------------------------------------------------------------------------------------
    def save(self, p_param, p_result, p_filename='best_parameters.csv') -> bool:
        """
        Saves the best result of the hyperparameter tuning in the root path.

        Parameters
        ----------
        p_param : dict
            A dictionary that consists of list of best parameters
        p_result : float
            Highest score
        p_filename  :str
            Name of summary file. Default = 'best_parameters.csv'

        Returns
        -------
        success : bool
            True, if summary file was created successfully. False otherwise.

        """

        filename = self._root_path + os.sep + p_filename
        filename.replace(os.sep + os.sep, os.sep)

        file = open(filename, 'wt')
        if file is None: return False
  
        self._save_line(file, 'Tuner', '"' + self.C_NAME + '"')    
        self._save_line(file, 'Number of evaluations', self._num_trials)     
        self._save_line(file, 'Highest Score', p_result)
        for key in p_param:
            self._save_line(file, key, p_param[key])

        file.close()
        
        try:
            self.HPDataStoring.save_data(self._root_path, 'tuning_summary', '\t')
            return True
        except:
            return False
        




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Training (Log):
    """
    Template class for a ML training and hyperparameter tuning.

    Parameters
    ----------
    p_scenario_cls 
        Name of ML scenario class, compatible to/inherited from class Scenario.
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
        Boolean switch for env/agent visualisation. Default = False
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_WE.

    """

    C_TYPE          = 'Training'
    C_NAME          = '????'

    C_CLS_RESULTS   = TrainingResults

    C_MODE_TRAIN    = 0 
    C_MODE_EVAL     = 1

    C_LOG_SEPARATOR = '------------------------------------------------------------------------------'

## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):
 
        # 1 Check and completion of parameters
        self._kwargs = p_kwargs.copy()

        # 1.1 Mandatory parameter p_scenario_cls
        try:
            scenario_cls = self._kwargs['p_scenario_cls']
        except:
            raise ParamError('Mandatory parameter p_scenario_cls not supplied')

        # 1.2 Optional parameter p_cycle_limit
        try:
            self._cycle_limit = self._kwargs['p_cycle_limit']
        except:
            self._cycle_limit = 0
            self._kwargs['p_cycle_limit'] = self._cycle_limit

        # 1.3 Optional parameter p_adaptation_limit
        try:
            self._adaptation_limit = self._kwargs['p_adaptation_limit']
        except:
            self._adaptation_limit = 0
            self._kwargs['p_adaptation_limit'] = self._adaptation_limit

        # 1.4 Optional parameter p_hpt
        try:
            self._hpt = self._kwargs['p_hpt']
        except:
            self._hpt = None
            self._kwargs['p_hpt'] = self._hpt

        # 1.5 Optional parameter p_hpt_trials
        try:
            self._hpt_trials = self._kwargs['p_hpt_trials']
        except:
            self._hpt_trials = 0
            self._kwargs['p_hpt_trials'] = self._hpt_trials 

        if ( self._hpt is not None ) and ( self._hpt_trials <= 0 ):
            raise ParamError('Please supply parameter p_hpt_trials with a number >0')

        # 1.6 Optional file path
        try:
            path = self._kwargs['p_path']
        except:
            path = None
            self._kwargs['p_path'] = path

        # 1.7 Optional parameter for visualization
        try:
            visualize = self._kwargs['p_visualize']
        except:
            visualize = False
            self._kwargs['p_visualize'] = visualize

        # 1.8 Optional log level
        try:
            logging = self._kwargs['p_logging']
        except:
            logging = Log.C_LOG_WE
            self._kwargs['p_logging'] = logging

        # 1.9 Optional environment mode
        try:
            env_mode = self._kwargs['p_env_mode']
        except:
            env_mode = Mode.C_MODE_SIM
            self._kwargs['p_env_mode'] = env_mode


        # 2 Initialization
        super().__init__(p_logging=logging)

        self._current_run       = 0
        self._new_run           = True
        self._results           = None
        self._root_path         = self._gen_root_path(path)
        self._current_path      = None
        self._scenario          = None
        self._mode              = self.C_MODE_TRAIN


        # 3 Setup scenario
        if self._hpt is None:
            try:
                self._scenario = scenario_cls( p_mode=env_mode, 
                                               p_ada=True,
                                               p_cycle_limit=self._cycle_limit,
                                               p_visualize=visualize,
                                               p_logging=logging )
            except:
                raise ParamError('Par p_scenario_cls: class "' + scenario_cls.__name__ + '" not compatible')


## -------------------------------------------------------------------------------------------------
    def get_scenario(self) -> Scenario:
        return self._scenario


## -------------------------------------------------------------------------------------------------
    def _gen_root_path(self, p_path) -> str:
        if p_path is None: return None

        now         = datetime.now()
        ts          = '%04d-%02d-%02d  %02d.%02d.%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        root_path   = p_path + str(os.sep) + ts + ' ' + self.C_TYPE + ' ' + self.C_NAME
        root_path.replace(str(os.sep) + str(os.sep), str(os.sep))
        os.mkdir(root_path)
        return root_path


## -------------------------------------------------------------------------------------------------
    def _gen_current_path(self, p_root_path, p_run) -> str:
        if p_root_path is None: return None
        if self._hpt is None: return self._root_path

        now             = datetime.now()
        ts              = '%04d-%02d-%02d  %02d.%02d.%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        current_path    = p_root_path + os.sep + ts + ' Run #' + str(p_run)
        current_path.replace(os.sep + os.sep, os.sep)
        os.mkdir(current_path)
        return current_path


## -------------------------------------------------------------------------------------------------
    def _init_results(self) -> TrainingResults:
        return self.C_CLS_RESULTS(self._scenario, 
                                  self._current_run, 
                                  self._scenario.get_cycle_id(), 
                                  p_path=self._current_path,
                                  p_logging=self._level)


## -------------------------------------------------------------------------------------------------
    def _close_results(self, p_results:TrainingResults):
        p_results.close()
        if self._current_path is not None: p_results.save(self._current_path)


## -------------------------------------------------------------------------------------------------
    def run_cycle(self) -> bool:
        """
        Runs a single training cycle.
        
        Returns
        -------
        termination_event : bool
            True, if training run has finished. False otherwise.

        """

        # 1 Intro
        if self._new_run:
            # 1.1 Start of new training run
            self._current_path  = self._gen_current_path(self._root_path, self._current_run)
            self._results       = self._init_results()
            self.log(self.C_LOG_TYPE_W, self.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, self.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, '-- Training run', self._current_run, 'started...')
            self.log(self.C_LOG_TYPE_W, self.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, self.C_LOG_SEPARATOR, '\n')
            self._new_run = False
            

        # 2 Run a single training cycle
        mode         = self._mode
        run_finished = self._run_cycle()

        if mode == self.C_MODE_TRAIN:
            self._results.num_cycles_train += 1
            self._results.num_cycles += 1
        elif mode == self.C_MODE_EVAL:
            self._results.num_cycles_eval += 1


        # 3 Assess results
        if ( self._cycle_limit > 0 ) and ( self._results.num_cycles_train >= self._cycle_limit ):
            # 3.1 Cycle limit reached
            self.log(self.C_LOG_TYPE_W, 'Training cycle limit', self._cycle_limit, 'reached')
            run_finished = True

        if run_finished:
            # 3.2 Training run finished
            self.log(self.C_LOG_TYPE_W, self.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, self.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, '-- Training run', self._current_run, 'finished')
            self.log(self.C_LOG_TYPE_W, self.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, self.C_LOG_SEPARATOR, '\n')

            self._scenario.get_model().save(self._current_path, 'trained model.pkl')
            self._close_results(self._results)
            self._results.log_results()

            self._current_path = None
            self._new_run       = True

        return run_finished
        

## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:
        """
        Single custom trainig cycle to be redefined. Custom training results can be added to using
        self._results.add_custom_result(p_name, p_value).

        Returns
        -------
        bool
            True, if training has finished. False otherwise.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def run(self) -> TrainingResults:
        """
        Runs a training and returns the results of the best trained/tuned agent.

        Returns
        -------
        TrainingResults
            Object with training results.

        """

        if self._hpt is None:
            # 1 Training without hyperparameter tuning
            self.log(self.C_LOG_TYPE_I, 'Training started (without hyperparameter tuning)')
            self._results = self._run()

        else:
            # 2 Training with hyperparameter tuning
            self.log(self.C_LOG_TYPE_I, 'Training started (with hyperparameter tuning)')
            training_param  = self._kwargs.copy()
            training_param.pop('p_hpt')
            training_param.pop('p_hpt_trials')
            self._results = self._hpt.maximize(p_training_cls=self.__class__, p_num_trials=self._hpt_trials, p_root_path=self._root_path, p_training_param=training_param)

        self.log(self.C_LOG_TYPE_I, 'Training completed')
        return self.get_results()


## -------------------------------------------------------------------------------------------------
    def _run(self) -> TrainingResults:
        self._new_run = True
        while not self.run_cycle(): pass
        return self.get_results()


## -------------------------------------------------------------------------------------------------
    def get_results(self) -> TrainingResults:
        return self._results