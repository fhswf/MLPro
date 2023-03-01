## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.feedforwardNN
## -- Module  : mlp.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-01  0.0.0     SY       Creation 
## -- 2023-03-01  0.1.0     SY       Initial design of MLP for MLPro v1.0.0
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2023-03-01)

This module provides model classes of multilayer perceptron. 
"""


from mlpro.sl import *
from mlpro.sl.feedforwardNN import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MLP(FeedforwardNN):
    """
    This class provides the base class of multilayer perceptron.
    """

    C_TYPE = 'Multilayer Perceptron'


## -------------------------------------------------------------------------------------------------
    def _hyperparameters_check(self) -> bool:
        """
        A method to check the hyperparameters related to the MLP model.

        Hyperparameters
        ----------
        num_hidden_layers :
            number of hidden layers.
        hidden_size :
            number of hidden neurons.
        activation_fct :
            activation function.
        output_activation_fct :
            extra activation function for the output layer.
        optimizer :
            optimizer.
        loss_fct :
            loss function.
        
        Returns
        ----------
        dict
            A dictionary includes the name of the hyperparameters and their values.
        """

        _param = {}

        hp = self.get_hyperparam()
        for idx in hp.get_dim_ids():
            par_name = hp.get_related_set().get_dim(idx).get_name_short()
            par_val  = hp.get_value(idx)
            _param[par_name] = par_val

        if (' input_size' not in _param ) or ( 'output_size' not in _param ):
            raise ParamError('Input size and/or output size of the network are not defined.')
        
        if 'update_rate' not in _param:
            _param['update_rate'] = 1
        elif _param.get('update_rate') < 1:
            raise ParamError("update_rate must be equal or higher than 1.")

        if 'num_hidden_layers' not in _param:
            raise ParamError("num_hidden_layers is not defined.")
        
        if 'hidden_size' not in _param:
            raise ParamError("hidden_size is not defined.") 
        
        if 'activation_fct' not in _param:
            raise ParamError("activation_fct is not defined.")

        if 'output_activation_fct' not in _param:
            _param['output_activation_fct'] = None
        
        if 'optimizer' not in _param:
            raise ParamError("optimizer is not defined.")
        
        if 'loss_fct' not in _param:
            raise ParamError("loss_fct is not defined.")
        
        return _param
