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
    def maximize(self, p_ofct, p_model:Model, p_num_trials, p_algo=C_ALGO_RAND) -> TrainingResults:
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
        p_algo : str    
            Selection of a hyperparameter tuning algorithm

        Returns
        -------
        TrainingResults
            Training results of the best tuned model (see class TrainingResults).

        """
        super().maximize(p_ofct, p_model, p_num_trials)

        self._algo          = p_algo
        return self._maximize()


## -------------------------------------------------------------------------------------------------
    def _maximize(self) -> TrainingResults:
        space       = self.SetupSpaces()
        if self._algo == 'TPE':
            self.algo = tpe.suggest()
        elif self._algo == 'RND':
            self.algo = rand.suggest()
            
        best        = fmin(self.Objective, space, self.algo, self._num_trials, trials=Trials())
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
        # 1. setup boundaries
        
        # 2. setup algo
        raise NotImplementedError
