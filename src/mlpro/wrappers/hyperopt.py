## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : hyperopt.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-07  0.0.0     SY       Creation 
## -- 2021-12-08  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-12-08)
This module provides a wrapper class for hyperparameter tuning by reusinng Hyperopt framework
"""


from hyperopt import *
from mlpro.bf.ml import *
from mlpro.bf.math import *
from mlpro.bf.various import *
from mlpro.rl.models import *
from mlpro.gt.models import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrHPTHyperopt(HyperParamTuner, ScientificObject):
    """
    This class is a ready to use wrapper class for Hyperopt framework. 
    Objects of this type can be treated as a hyperparameter tuner object.
    
    Parameters
    ----------
    p_arg1 : TYPE
        explanation of the first parameter.
    p_arg2 : TYPE, optional
        explanation of the second parameter. The default is True.
        
    Attributes
    ----------
    C_NAME: str
        Name of the class.
    C_ALGO_TPE: str
        Refer to Tree of Parzen Estimators (TPE) algorithm.
    C_ALGO_RAND: str
        Refer to Random Grid Search algorithm.
    """
    
    C_NAME              = 'Hyperopt'
    
    C_ALGO_TPE          = 'TPE'
    C_ALGO_RAND         = 'RND'
        
    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_PROCEEDINGS
    C_SCIREF_AUTHOR     = "James Bergstra, Dan Yamins, David D. Cox"
    C_SCIREF_TITLE      = "Hyperopt: A Python Library for Otimizing the Hyperparameters of Machine Learning Algorithms"
    C_SCIREF_CONFERENCE = "Proceedings of the 12th Python in Science Conference"
    C_SCIREF_YEAR       = "2013"
    C_SCIREF_PAGES      = "13-19"
    C_SCIREF_DOI        = "10.25080/Majora-8b375195-003"
    C_SCIREF_EDITOR     = "Stefan van der Walt, Jarrod Millman, Katy Huff"
    

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, p_algo=C_ALGO_RAND, p_ids=None):
        """
        Parameters:
            p_logging     Log level (see constants for log levels)
        p_algo : str, optional    
            Selection of a hyperparameter tuning algorithm, default: C_ALGO_RAND
        p_ids : list of str, optional
            List of hyperparameter ids to be tuned, otherwise all hyperparameters, default: None
        """
        super().__init__(p_logging=p_logging)

        self._algo          = p_algo
        self._ids           = p_ids
    

## -------------------------------------------------------------------------------------------------
    def _maximize(self) -> TrainingResults:
        """
        This method is a place to setup a hp tuner based on hp structure of the model
        and run the hp tuner.

        Returns
        -------
        best_result : float
            The best result after a number of evaluations.

        """
        spaces              = self.SetupSpaces()
        if self._algo == 'TPE':
            self.algo       = tpe.suggest
        elif self._algo == 'RND':
            self.algo       = rand.suggest
            
        best_result         = fmin(self._ofct_hyperopt, spaces, self.algo, self._num_trials, trials=Trials())
        return -best_result

## -------------------------------------------------------------------------------------------------
    def _ofct_hyperopt(self, p_params):
        """
        This method is a place to run the evaluations by getting next set of hps from the tuner,
        inducting hps to the model, and running the the objective function.

        Returns
        -------
        result : float
            The result of an evaluations.
            
        """
        param_id                = 0
        for x in range(len(self.hp_tupel)):
            if not self._ids:
                _id             = self.hp_tupel[x].get_dim_ids()
            else:
                _id             = self_ids
                
            for i in range(len(_id)):
                if isinstance(self._model, MultiAgent) or isinstance(self._model, MultiPlayer):
                    for x in range(len(self._model.get_agents())):
                        self._model.get_agents()[x][0]._policy._hyperparam_tupel.set_value(_id[i], p_params[param_id])
                elif isinstance(self._model, Agent) or isinstance(self._model, Player):
                    self._model._policy._hyperparam_tupel.set_value(_id[i], p_params[param_id])
                else:
                    try:
                        self._model._hyperparam_tupel.set_value(_id[i], p_params[param_id])
                    except:
                        raise NotImplementedError
        
        result                  = self._ofct()
        return -(result.highscore)

## -------------------------------------------------------------------------------------------------
    def SetupSpaces(self):
        """
        This method is used to setup the hyperparameter spaces, including the tuning boundaries and a suitable discrete value.
        The hyperparameter should be bounded both above and below.
        We are using the "quantized" continuous distributions for natural and integer numbers.
        Meanwhile the real numbers are not quantized.
        For different parameter expressions, please redefined this method and check http://hyperopt.github.io/hyperopt/getting-started/search_spaces/!
        For big data handling, please redifined this method!
        
        Returns
        -------
        spaces : list
            List of parameter expressions.

        """
        self.hp_tupel            = []
        if isinstance(self._model, MultiAgent) or isinstance(self._model, MultiPlayer):
            for x in range(len(self._model.get_agents())):
                self.hp_tupel.append(self._model.get_agents()[x][0]._policy._hyperparam_tupel)
        elif isinstance(self._model, Agent) or isinstance(self._model, Player):
            self.hp_tupel.append(self._model._policy._hyperparam_tupel)
        else:
            try:
                self.hp_tupel.append(self._model._hyperparam_tupel)
            except:
                raise NotImplementedError
        
        spaces                  = []
        for x in range(len(self.hp_tupel)):
            if not self._ids:
                _id             = self.hp_tupel[x].get_dim_ids()
            else:
                _id             = self_ids
            for i in range(len(_id)):
                hp_id           = _id[i]
                hp_set          = self.hp_tupel[x].get_related_set().get_dim(hp_id)
                hp_base_set     = hp_set._base_set
                hp_boundaries   = hp_set.get_boundaries()
                hp_name_short   = hp_set.get_name_short()
                if not hp_boundaries:
                    raise ImplementationError('Missing boundary of a hyperparameter!')
                else:
                    hp_low      = hp_boundaries[0]
                    hp_high     = hp_boundaries[1]
                    if hp_base_set == Dimension.C_BASE_SET_N or hp_base_set == Dimension.C_BASE_SET_Z:
                        spaces.append(hp.quniform(hp_name_short+'_'+str(x),hp_low,hp_high,1))
                    elif hp_base_set == Dimension.C_BASE_SET_R:
                        spaces.append(hp.uniform(hp_name_short+'_'+str(x),hp_low,hp_high))
                    else:
                        raise NotImplementedError
        return spaces
