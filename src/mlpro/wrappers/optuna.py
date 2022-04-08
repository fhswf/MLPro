## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : optuna.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-03-24  0.0.0     SY       Creation 
## -- 2022-03-24  1.0.0     SY       Release of first version
## -- 2022-03-25  1.0.1     SY       Change methods names: _ofct_optuna and get_parameters
## -- 2022-04-05  1.0.2     SY       Add tuning recap visualization: class _plot_results
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-04-05)

This module provides a wrapper class for hyperparameter tuning by reusing Optuna framework
"""


import optuna
from mlpro.bf.ml import *
from mlpro.bf.math import *
from mlpro.bf.various import *
from mlpro.rl.models import *
from mlpro.gt.models import *
import os




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrHPTOptuna(HyperParamTuner, ScientificObject):
    """
    This class is a ready to use wrapper class for Optuna framework. 
    Objects of this type can be treated as a hyperparameter tuner object.
    
    Parameters
    ----------
    p_logging: Log
        Log level (see constants for log levels)
    p_ids : list of str, optional
        List of hyperparameter ids to be tuned, otherwise all hyperparameters, default: None
    p_visualization : boolean
        enable visualization at the end of the tuning, default: False
        
    Attributes
    ----------
    C_NAME: str
        Name of the class.
    """
    
    C_NAME              = 'Optuna'
        
    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_PROCEEDINGS
    C_SCIREF_AUTHOR     = "Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori"
    C_SCIREF_TITLE      = "Optuna: A Next-Generation Hyperparameter Optimization Framework"
    C_SCIREF_YEAR       = "2019"
    C_SCIREF_ISBN       = "9781450362016"
    C_SCIREF_PUBLISHER  = "Association for Computing Machinery"
    C_SCIREF_CITY       = "New York"
    C_SCIREF_COUNTRY    = "USA"
    C_SCIREF_URL        = "https://doi.org/10.1145/3292500.3330701"
    C_SCIREF_DOI        = "10.1145/3292500.3330701"
    C_SCIREF_PAGES      = "2623–2631"
    C_SCIREF_BOOKTITLE  = "Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining"
    
    C_LOG_SEPARATOR = '------------------------------------------------------------------------------'
    

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, p_ids=None, p_visualization=False):
        super().__init__(p_logging=p_logging)

        self._ids = p_ids
        self.num_trials = 0
        self.visualize = p_visualization
        
        self.log(self.C_LOG_TYPE_I, 'Optuna configuration is successful')


## -------------------------------------------------------------------------------------------------
    def _maximize(self) -> TrainingResults:
        """
        This method is a place to setup a hp tuner based on hp structure of the model
        and run the hp tuner. Create a study object and execute the optimization.

        Returns
        -------
        best_result : float
            The best result after a number of evaluations.

        """
        
        if self._training_cls is None:
            raise ParamError('Mandatory parameter self._training_cls is not supplied')
        
        if self._num_trials <= 0:
            raise ParamError('Parameter self._num_trials must be greater than 0')
        
        if self._root_path is None:
            raise ParamError('Mandatory parameter self._root_path is not supplied')
        
        if self._training_param is None:
            raise ParamError('Mandatory parameter self._training_param is not supplied')
        
        # change root path in training param
        self._training_param['p_training_param']['p_path'] = self._root_path+os.sep+'HyperparameterTuning'+os.sep+'Base (Preparation)'
        if not os.path.exists(self._training_param['p_training_param']['p_path']):
            os.mkdir(self._root_path+os.sep+'HyperparameterTuning')
            os.mkdir(self._training_param['p_training_param']['p_path'])
        
        # ignore collecting data during tuning to save tuning time and memory
        self._training_param['p_training_param']['p_collect_states'] = False
        self._training_param['p_training_param']['p_collect_actions'] = False
        self._training_param['p_training_param']['p_collect_rewards'] = False
        self._training_param['p_training_param']['p_logging'] = Log.C_LOG_NOTHING
        self._training_param['p_training_param']['p_visualize'] = False
        self._training_param['p_training_param']['p_collect_eval'] = True
        
        # instantiate a scenario class and define the model in a variable
        training_cls = self._training_cls(**self._training_param['p_training_param'])
        self._model = training_cls._scenario._model
        
        # prepare a data storing class
        for x, _id in enumerate(self._model._hyperparam_tuple.get_dim_ids()):
            hp_object = self._model._hyperparam_tuple.get_related_set().get_dim(_id)
            self.variables.append(hp_object.get_name_short()+'_'+str(x))
        self.HPDataStoring = DataStoring(self.variables)
        self.HPDataStoring.add_frame('HP_0')
    
        # run the trials and gain the highest score
        study = optuna.create_study(direction="maximize")
        study.optimize(self._ofct_optuna, n_trials=self._num_trials)
        best_trial = study.best_trial
        best_result = best_trial.value
        best_param = study.best_params
        self.save(best_param, best_result, 'best_parameters.csv')
        
        if self.visualize:
            self._plot_results(study)
        
        return best_result


## -------------------------------------------------------------------------------------------------
    def _ofct_optuna(self, trial):
        """
        Wrap model training with an objective function and return the output score.

        Parameters
        ----------
        trial : object
            Suggest hyperparameters using a trial object.

        Returns
        -------
        result.highscore : float
            final score of a trial.

        """
        self.log(self.C_LOG_TYPE_I, 'Trial number '+str(self.num_trials)+' has started')
        self.log(self.C_LOG_TYPE_I, self.C_LOG_SEPARATOR, '\n')

        # change root path in training param
        self._training_param['p_training_param']['p_path'] =  self._root_path+os.sep+'HyperparameterTuning'+os.sep+'Trial_'+str(self.num_trials)
        if not os.path.exists(self._training_param['p_training_param']['p_path']):
            os.mkdir(self._training_param['p_training_param']['p_path'])

        # instantiate a scenario class
        training_cls = self._training_cls(**self._training_param['p_training_param'])
        self._model = training_cls._scenario._model
        
        # setup parameters that compatible to optuna spaces
        p_params = self.get_parameters(trial)
        for x, _id in enumerate(self._model._hyperparam_tuple.get_dim_ids()):
            self._model._hyperparam_tuple.set_value(_id, p_params[x])

        # run the scenario and retrieve the high score
        result = training_cls.run()

        # store trial parameters and the trial's result
        self.HPDataStoring.memorize('Trial', 'HP_0', self.num_trials)
        self.HPDataStoring.memorize('Highscore', 'HP_0', result.highscore)
        for x, _id in enumerate(self._model._hyperparam_tuple.get_dim_ids()):
            hp_name_short = self._model._hyperparam_tuple.get_related_set().get_dim(_id).get_name_short()
            self.HPDataStoring.memorize(hp_name_short+'_'+str(x), 'HP_0', p_params[x])

        self.num_trials += 1

        self.log(self.C_LOG_TYPE_I, 'Trial number '+str(self.num_trials)+' has finished')
        self.log(self.C_LOG_TYPE_I, self.C_LOG_SEPARATOR, '\n')
        
        return result.highscore
        

## -------------------------------------------------------------------------------------------------
    def get_parameters(self, trial):
        """
        This method is used to get parameters within boundaries.
        The hyperparameter should be bounded both above and below.
        For different parameter expressions, please redefined this method and check https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html!
        For big data handling, please redifined this method!

        Parameters
        ----------
        trial : object
            Suggest hyperparameters using a trial object.
        
        Returns
        -------
        parameters : list
            List of parameter expressions.

        """
        
        if self._model._hyperparam_tuple is None:
            self._model._init_hyperparam()
        
        parameters = []
        for x, _id in enumerate(self._model._hyperparam_tuple.get_dim_ids()):
            hp_object = self._model._hyperparam_tuple.get_related_set().get_dim(_id)
            hp_boundaries = hp_object.get_boundaries()
            if not hp_boundaries:
                raise ImplementationError('Missing boundary of a hyperparameter!')
            else:
                hp_low = hp_boundaries[0]
                hp_high = hp_boundaries[1]
                if hp_object._base_set == Dimension.C_BASE_SET_N or hp_object._base_set == Dimension.C_BASE_SET_Z:
                    parameters.append(trial.suggest_int(hp_object.get_name_short()+'_'+str(x),hp_low,hp_high))
                elif hp_object._base_set == Dimension.C_BASE_SET_R:
                    parameters.append(trial.suggest_uniform(hp_object.get_name_short()+'_'+str(x),hp_low,hp_high))
                else:
                    raise ImplementationError('Missing a short name of a hyperparameter!')
        
        self.log(self.C_LOG_TYPE_I, 'New parameters for optuna tuner is ready')
        self.log(self.C_LOG_TYPE_I, self.C_LOG_SEPARATOR, '\n')
        
        return parameters


## -------------------------------------------------------------------------------------------------
    def _plot_results(self, p_study):
        """
        Visualize the tuning recap.

        Parameters
        ----------
        p_study : Study
            Optuna Study object.

        """
        self.log(self.C_LOG_TYPE_I, 'Plotting of tuning recap is started')
        
        fig = optuna.visualization.plot_slice(p_study)
        fig.show()
        
        fig = optuna.visualization.plot_param_importances(p_study)
        fig.show()
        
        fig = optuna.visualization.plot_parallel_coordinate(p_study)
        fig.show()
        
        self.log(self.C_LOG_TYPE_I, 'Plotting of tuning recap is succesful')
