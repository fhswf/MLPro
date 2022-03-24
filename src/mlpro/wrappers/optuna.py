## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : optuna.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-03-24  0.0.0     SY       Creation 
## -- 2022-03-24  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-03-24)

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
    p_logging:
        Log level (see constants for log levels)
    p_ids : list of str, optional
        List of hyperparameter ids to be tuned, otherwise all hyperparameters, default: None
        
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
    def __init__(self, p_logging=Log.C_LOG_ALL, p_ids=None):
        super().__init__(p_logging=p_logging)

        self._ids = p_ids
        self.num_trials = 0
        self.main_path = None
        
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
        return best_result


## -------------------------------------------------------------------------------------------------
    def objective(trial):
        """
        Wrap model training with an objective function and return the output score.

        Parameters
        ----------
        trial : object
            Suggest hyperparameters using a trial object.

        Returns
        -------
        score : TYPE
            DESCRIPTION.

        """
        
        return score
        

## -------------------------------------------------------------------------------------------------
    def SetupSpaces(self):
        """
        This method is used to setup the hyperparameter spaces, including the tuning boundaries and a suitable discrete value.
        The hyperparameter should be bounded both above and below.
        We are using the "quantized" continuous distributions for natural and integer numbers.
        Meanwhile the real numbers are not quantized.
        For different parameter expressions, please redefined this method and check https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html!
        For big data handling, please redifined this method!
        
        Returns
        -------
        spaces : list
            List of parameter expressions.

        """
        
        # if self._model._hyperparam_tuple is None:
        #     self._model._init_hyperparam()
        
        # spaces = []
        # for x, _id in enumerate(self._model._hyperparam_tuple.get_dim_ids()):
        #     hp_object = self._model._hyperparam_tuple.get_related_set().get_dim(_id)
        #     hp_boundaries = hp_object.get_boundaries()
        #     if not hp_boundaries:
        #         raise ImplementationError('Missing boundary of a hyperparameter!')
        #     else:
        #         hp_low = hp_boundaries[0]
        #         hp_high = hp_boundaries[1]
        #         if hp_object._base_set == Dimension.C_BASE_SET_N or hp_object._base_set == Dimension.C_BASE_SET_Z:
        #             spaces.append(hp.quniform(hp_object.get_name_short()+'_'+str(x),hp_low,hp_high,1))
        #         elif hp_object._base_set == Dimension.C_BASE_SET_R:
        #             spaces.append(hp.uniform(hp_object.get_name_short()+'_'+str(x),hp_low,hp_high))
        #         else:
        #             raise ImplementationError('Missing a short name of a hyperparameter!')
        #     self.variables.append(hp_object.get_name_short()+'_'+str(x))
        
        # self.log(self.C_LOG_TYPE_I, 'Spaces for hyperopt is ready')
        # self.log(self.C_LOG_TYPE_I, self.C_LOG_SEPARATOR, '\n')
        
        return spaces
