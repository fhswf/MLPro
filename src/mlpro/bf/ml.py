## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : ml
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-20  0.0.0     DA       Creation 
## -- 2021-08-25  1.0.0     DA       Release of first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-09-18  1.0.1     MRD      Buffer Class Implementation. Add new parameter buffer
## --                                to the Adaptive Class
## -- 2021-09-19  1.0.1     MRD      Improvement on Buffer Class. Implement new base class
## --                                Buffer Element and BufferRnd
## -- 2021-09-25  1.0.2     MRD      Add __len__ functionality for SARBuffer
## -- 2021-10-06  1.0.3     DA       Extended class Adaptive by new methods _adapt(), get_adapted(),
## --                                _set_adapted(); moved Buffer classes to mlpro.bf.data.py
## -- 2021-10-25  1.0.4     SY       Enhancement of class Adaptive by adding ScientificObject.
## -- 2021-10-26  1.1.0     DA       New class AdaptiveFunction
## -- 2021-10-29  1.1.1     DA       New method Adaptive.set_random_seed()
## -- 2021-11-08  1.2.0     DA       - Class Adaptive renamed to Model
## --                                - New classes Mode, Scenario, TrainingResults, Training, 
## --                                  HyperParamTuner
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2021-11-08)

This module provides fundamental machine learning functionalities and properties.
"""

from os import confstr_names
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.data import Buffer




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParam(Dimension):
    """
    ...
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
class HyperParamSpace(ESpace):
    """
    ...
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamTupel(Element):
    """
    ...
    """

## -------------------------------------------------------------------------------------------------
    def set_value(self, p_dim_id, p_value):
        super().set_value(p_dim_id, p_value)
        self._set.get_dim(p_dim_id).callback_on_change(p_value)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Model(Log, LoadSave, ScientificObject):
    """
    Property class for adapativity. And if something can be adapted it should be loadable and saveable
    so that this class provides load/save properties as well.
    """

    C_TYPE          = 'Model'
    C_NAME          = '????'

    C_BUFFER_CLS    = Buffer            

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_buffer_size=0, p_ada=True, p_logging=True):
        """
        Parameters:
            p_buffer_size       Initial size of internal data buffer (0=no buffering)
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        Log.__init__(self, p_logging=p_logging)
        self._adapted           = False
        self.switch_adaptivity(p_ada)
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tupel  = None
        self._init_hyperparam()

        if p_buffer_size > 0:
            self._buffer = self.C_BUFFER_CLS(p_size=p_buffer_size)
        else:
            self._buffer = None

        self._attrib_hp1        = 0


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self):
        """
        Implementation specific hyperparameters can be added here. Please follow these steps:
        a) Add each hyperparameter as an object of type HyperParam to the internal hyperparameter
           space object self._hyperparam_space
        b) Create hyperparameter tuple and bind to self._hyperparam_tupel
        c) Set default value for each hyperparameter
        """


## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTupel:
        return self._hyperparam_tupel


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """
        Switches adaption functionality on/off.
        
        Parameters:
            p_ada               Boolean switch for adaptivity
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

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        return self._adapted


## -------------------------------------------------------------------------------------------------
    def _set_adapted(self, p_adapted:bool):
        self._adapted = p_adapted


## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Adapts something inside in the sense of machine learning. Please redefine and describe the 
        protected method _adapt() that is called here. 

        Parameters:
            p_args          All parameters that are needed for the adaption. 

        Returns:
            True, if something has been adapted. False otherwise.
        """

        if not self._adaptivity: return False
        self.log(self.C_LOG_TYPE_I, 'Adaptation started')
        self._set_adapted(self._adapt(*p_args))
        return self.get_adapted()


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Please redefine and implement your specific adaptation algorithm. Furthermore please describe 
        the type and purpose of all parameters needed by your implementation. This method will be 
        called by public method adapt() if adaptivity is switched on. 

        Parameters:
            p_args[0]           ...
            p_args[1]           ...
            ...

        Returns:
            True, if something has been adapted. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        if self._buffer is not None: self._buffer.clear()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdaptiveFunction (Model, Function):
    """
    Model class for an adaptive bi-multivariate mathematical function.
    """

    C_TYPE          = 'Model Fct'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_input_space:MSpace, p_output_space:MSpace, p_output_elem_cls=Element, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=True):
        """
        Parameters:
            p_input_space       Input space
            p_output_space      Output space
            p_output_elem_cls   Output element class (compatible to class Element)
            p_threshold         Threshold for the difference between a setpoint and a computed output. 
                                Computed outputs with a difference less than this threshold will be 
                                assessed as 'good' outputs.
            p_buffer_size       Initial size of internal data buffer (0=no buffering)
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        Model.__init__(self, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)
        Function.__init__(self, p_input_space=p_input_space, p_output_space=p_output_space, p_output_elem_cls=p_output_elem_cls)
        self._threshold         = p_threshold
        self._mappings_total    = 0             # Number of mappings since last adaptation
        self._mappings_good     = 0             # Number of 'good' mappings since last adaptation


## -------------------------------------------------------------------------------------------------
    def adapt(self, p_input:Element, p_output:Element) -> bool:
        """
        Parameters:
            p_input         Abscissa/input element object (type Element)
            p_output        Setpoint ordinate/output element (type Element)
        """

        if not self._adaptivity: return False
        self.log(self.C_LOG_TYPE_I, 'Adaptation started')

        # Quality check
        if self._output_space.distance(p_output, self.map(p_input)) <= self._threshold:
            # Quality of function ok. No need to adapt.
            self._mappings_total    += 1
            self._mappings_good     += 1

        else:
            # Quality of function not ok. Adapation is to be triggered.
            self._set_adapted(self._adapt(p_input, p_output))
            if self.get_adapted():
                self._mappings_total    = 1
                self._mappings_good     = 1
            else:
                self._mappings_total    += 1

        return self.get_adapted()


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_input:Element, p_output:Element) -> bool:
        """
        Protected custom adaptation algorithm that is called by public adaptation method. Please redefine.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        """
        Returns the maturity of the adaptive function. The maturity is defined as the relation 
        between the number of successful mapped inputs and the total number of mappings since the 
        last adaptation.
        """

        if self._mappings_total == 0: return 0
        return self._mappings_good / self._mappings_total





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Mode (Log):
    """
    Property class that adds a mode and related methods to a child class.
    """

    C_MODE_INITIAL  = -1
    C_MODE_SIM      = 0
    C_MODE_REAL     = 1

    C_VALID_MODES   = [ C_MODE_SIM, C_MODE_REAL ]

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mode, p_logging=True):
        super().__init__(p_logging)
        self._mode = self.C_MODE_INITIAL
        self.set_mode(p_mode)


## -------------------------------------------------------------------------------------------------
    def get_mode(self):
        return self._mode


## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):
        if not p_mode in self.C_VALID_MODES: raise ParamError('Invalid mode')
        if self._mode == p_mode: return
        self._mode = p_mode
        self.log(self.C_LOG_TYPE_I, 'Operation mode set to', self._mode)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Scenario (Mode, LoadSave):
    """
    Template class for a common ML scenario with an adaptive model inside.
    """

    C_TYPE      = 'Scenario'
    C_NAME      = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mode=Mode.C_MODE_SIM, p_ada:bool=True, p_logging:bool=True):
        """
        Parameters:
            p_mode          Operation mode (see class Mode)
            p_ada           Boolean switch for adaptivity of internal model
            p_logging       Boolean switch for logging
        """

        self._model = self._setup(p_mode=self.C_MODE_SIM, p_ada=p_ada, p_logging=p_logging)
        super().__init__(p_mode, p_logging)


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_logging:bool) -> Model:
        """
        Setup the ML scenario by redefining this method in the child class. 

        Parameters:
            p_mode          Operation mode (see class Mode)
            p_ada           Boolean switch for adaptivity of internal model
            p_logging       Boolean switch for logging functionality

        Returns:
            Adaptive model inside the ML scenario
        """

        raise NotImplementedError


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
    def reset(self, p_seed=1):
        """
        Resets the scenario. Internal random generators shall be seed with the given value.

        Parameters:
            p_seed          Seed value for internal random generator
        """

        self.log(self.C_LOG_TYPE_I, 'Reset with seed', str(p_seed))
        self._reset(p_seed)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Custom method to reset the scenario and to set the given random seed value. See method reset().
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def run_step(self) -> bool:
        """
        Runs a single process step.

        Returns:
            True, if process step was successful. False otherwise.
        """

        self.log(self.C_LOG_TYPE_I, 'Start of process step')
        result = self._run_step()
        self.log(self.C_LOG_TYPE_I, 'End of process step')
        return result


## -------------------------------------------------------------------------------------------------
    def _run_step(self) -> bool:
        """
        Custom implementation of a single process step. To be redefined.

        Returns:
            True, if process step was successful. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def run(self):
        """
        Runs the scenario as a sequence of single process steps until there was a terminating event.
        """

        self.log(self.C_LOG_TYPE_I, 'Start of processing...')
        while self.run_step(): pass
        self.log(self.C_LOG_TYPE_I, 'End of processing...')





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TrainingResults (Saveable):
    """
    Results of a training (see class Training).
    """

    def __init__(self, p_scenario:Scenario):
        self.scenario       = p_scenario
        self.ts_start       = None
        self.ts_end         = None
        self.ts_duration    = 0
        self.score          = None
        self.trained_model  = None





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamTuner (Log):
    """
    Template class for hyperparameter tuning (HPT).
    """

    C_TYPE      = 'HyperParam Tuner'
    C_NAME      = '????'

## -------------------------------------------------------------------------------------------------
    def maximize(self, p_ofct, p_model:Model, p_num_trials) -> TrainingResults:
        """

        Parameters:
            p_ofct          Objective function to be maximized
            p_model         Model object to be tuned
            p_num_trials    Number of trials

        Returns:
            Training results of the best tuned model (see class TrainingResults)
        """

        self._ofct          = p_ofct
        self._model         = p_model
        self._num_trials    = p_num_trials
        return self._maximize()


## -------------------------------------------------------------------------------------------------
    def _maximize(self) -> TrainingResults:
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Training (Log):
    """
    Template class for a ML training and hyperparameter tuning.
    """

    C_TYPE      = 'Training'
    C_NAME      = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scenario:Scenario, p_hpt:HyperParamTuner=None, p_hpt_trials=0, p_logging=False):
        """
        Parameters:
            p_scenario          ML scenario (see class Scenario)
            p_hpt               Optional hyperparameter tuner (see class HyperParamTuner)
            p_hpt_trials        Optional number of hyperparameter tuning trials
            p_logging           Boolean switch for logging       
        """

        super().__init__(p_logging=p_logging)

        self._scenario      = p_scenario
        self._hpt           = p_hpt
        self._hpt_trials    = p_hpt_trials
        self._results       = None

        if self._hpt is not None: 
            raise NotImplementedError('Hyperparameter Tuning not yet implemented')

        if ( self._hpt is not None ) and ( p_hpt_trials <= 0 ):
            raise ParamError('Please check number of trials for hyperparameter tuning')


## -------------------------------------------------------------------------------------------------
    def get_scenario(self) -> Scenario:
        return self._scenario


## -------------------------------------------------------------------------------------------------
    def run(self) -> TrainingResults:
        """
        Runs a training and returns the results of the best trained/tuned agent.
        """

        if self._hpt is None:
            # 1 Training without hyperparameter tuning
            self.log(self.C_LOG_TYPE_I, 'Training started (without hyperparameter tuning)')
            self._results = self._run()

        else:
            # 2 Training 
            self.log(self.C_LOG_TYPE_I, 'Training started (with hyperparameter tuning)')
            self._results = self._hpt.maximize(p_ofct=self._run, p_model=self._scenario.get_model(), p_num_trials=self._hpt_trials)

        self.log(self.C_LOG_TYPE_I, 'Training completed')
        return self.get_results()


## -------------------------------------------------------------------------------------------------
    def _run(self) -> TrainingResults:
        """
        Training process to be implemented.

        Returns:
            Results of the best trained model.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_results(self) -> TrainingResults:
        return self._results