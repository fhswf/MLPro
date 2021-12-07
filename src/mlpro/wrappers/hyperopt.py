## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : hyperopt.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-07  0.0.0     SY       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2021-12-07)
This module provides a wrapper class for hyperparameter tuning by reusinng Hyperopt framework
"""


from hyperopt import *
from mlpro.bf.ml import *
from mlpro.bf.math import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrHPTHyperopt(HyperParamTuner):
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
    
    C_NAME          = 'Hyperopt'
    
    C_ALGO_TPE      = 'TPE'
    C_ALGO_RAND     = 'RND'

## -------------------------------------------------------------------------------------------------
    # def maximize(self, p_ofct, p_model:Model, p_num_trials) -> TrainingResults:
    #     """
    #     ...

    #     Parameters
    #     ----------
    #     p_ofct 
    #         Objective function to be maximized.
    #     p_model : Model
    #         Model object to be tuned.
    #     p_num_trials : int    
    #         Number of trials

    #     Returns
    #     -------
    #     TrainingResults
    #         Training results of the best tuned model (see class TrainingResults).

    #     """

    #     self._ofct          = p_ofct
    #     self._model         = p_model
    #     self._num_trials    = p_num_trials
    #     return self._maximize()

## -------------------------------------------------------------------------------------------------
    def maximize(self, p_ofct, p_model:Model, p_num_trials, p_algo=C_ALGO_RAND, p_ids=None) -> TrainingResults:
        """
        ...

        Parameters
        ----------
        p_ofct 
            Objective function to be maximized.
        p_model : Model
            Model object to be tuned.
        p_num_trials : int    
            Number of trials
        p_algo : str, optional    
            Selection of a hyperparameter tuning algorithm, default: C_ALGO_RAND
        p_algo : list of str, optional
            List of hyperparameter ids to be tuned, otherwise all hyperparameters, default: None

        Returns
        -------
        TrainingResults
            Training results of the best tuned model (see class TrainingResults).

        """
        super().maximize(p_ofct, p_model, p_num_trials)

        self._algo          = p_algo
        self._ids           = p_ids
        return self._maximize()


## -------------------------------------------------------------------------------------------------
    def _maximize(self) -> TrainingResults:
        spaces              = self.SetupSpaces()
        if self._algo == 'TPE':
            self.algo       = tpe.suggest()
        elif self._algo == 'RND':
            self.algo       = rand.suggest()
            
        best                = fmin(self.Objective, spaces, self.algo, self._num_trials, trials=Trials())
        raise NotImplementedError

## -------------------------------------------------------------------------------------------------
    def Objective(self, p_params):
        # bglp_algorithm = "GlobalInterpolation"
        # ExplorationHalf, LR_MARGIN, LR_DEMAND, LR_ENERGY = args
        # sum_potential = BGLP_Run(ExplorationHalf, LR_MARGIN, LR_DEMAND, LR_ENERGY, bglp_algorithm)
        # return sum_potential
        raise NotImplementedError

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
        hp_tupel            = self._model._hyperparam_tupel
        if not self._ids:
            self._ids       = hp_tupel.get_dim_ids()
        spaces              = []
        for i in range(len(self._ids)):
            hp_id           = self._ids[i]
            hp_set          = hp_tupel.get_related_set().get_dim(hp_id)
            hp_base_set     = hp_set._base_set
            hp_boundaries   = hp_set.get_boundaries()
            hp_name_short   = hp_set.get_name_short()
            if not hp_boundaries:
                raise NameError('Missing boundaries of a hyperparameter!')
            else:
                hp_low      = hp_boundaries[0]
                hp_high     = hp_boundaries[1]
                if hp_boundaries == Dimension.C_BASE_SET_N or hp_boundaries == Dimension.C_BASE_SET_Z:
                    spaces.append(hp.quniform(hp_name_short,hp_low,hp_high,1))
                elif hp_boundaries == Dimension.C_BASE_SET_Z:
                    spaces.append(hp.uniform(hp_name_short,hp_low,hp_high))
                else:
                    raise NotImplementedError
        return spaces
