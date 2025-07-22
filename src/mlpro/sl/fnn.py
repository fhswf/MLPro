## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.sl
## -- Module  : fnn.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-01  0.0.0     SY       Creation 
## -- 2023-03-01  0.1.0     SY       Initial design of FNN for MLPro v1.0.0
## -- 2023-03-07  1.0.0     SY       Release first version 
## -- 2023-03-10  1.1.0     SY       Combining _hyperparameters_check and _init_hyperparam
## -- 2025-07-07  1.2.0     DA       Refactoring of method FNN._map()
## -- 2025-07-18  1.3.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2025-07-18)

This module provides model classes of feedforward neural networks for supervised learning tasks. 
"""


from mlpro.bf import ParamError
from mlpro.bf.math import Element
from mlpro.bf.ml import HyperParam, HyperParamTuple

from mlpro.sl.basics import SLAdaptiveFunction



# Export list for public API
__all__ = [ 'FNN', 
            'MLP' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FNN (SLAdaptiveFunction):
    """
    This class provides the base class of feedforward neural networks.
    """

    C_TYPE = 'Feedforward NN'


## -------------------------------------------------------------------------------------------------
    def forward(self, p_input:Element) -> Element:
        """
        Custom forward propagation in neural networks to generate some output that can be called by
        an external method. Please redefine.

        Parameters
        ----------
        p_input : Element
            Input data

        Returns
        ----------
        output : Element
            Output data
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _map(self, p_input: Element, p_output: Element = None, p_dim = None):
        """
        Maps a multivariate abscissa/input element to a multivariate ordinate/output element. 

        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)
        """
        
        output = self.forward(input)
        p_output.set_values(output)
        return p_output


## -------------------------------------------------------------------------------------------------
    def _optimize(self):
        """
        This method provides provide a funtionality to call the optimizer of the feedforward network.
        """
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _calc_loss(self, p_act_output:Element, p_pred_output:Element):
        """
        This method provides provide a funtionality to call the loss function of the feedforward
        network.

        Parameters
        ----------
        p_act_output : Element
            Actual output from the buffer.
        p_pred_output : Element
            Predicted output by the SL model.
        """
        
        ids_ = self.get_hyperparam().get_dim_ids()
        return self.get_hyperparam().get_value(ids_[8])(p_act_output.get_values(), p_pred_output.get_values())





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MLP (FNN):
    """
    This class provides the base class of multilayer perceptron.
    """

    C_TYPE = 'Multilayer Perceptron'


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par): 
        """
        A method to deal with the hyperparameters related to the MLP model.

        Hyperparameters
        ----------
        p_update_rate :
            update rate.
        p_num_hidden_layers :
            number of hidden layers.
        p_hidden_size :
            number of hidden neurons.
        p_activation_fct :
            activation function.
        p_output_activation_fct :
            extra activation function for the output layer.
        p_optimizer :
            optimizer.
        p_loss_fct :
            loss function.
        """
        
        try:
            p_input_size = self._input_space.get_num_dim()
            p_output_size = self._output_space.get_num_dim()
        except:
            raise ParamError('Input size and/or output size of the network are not defined.')
        
        if 'p_update_rate' not in p_par:
            p_par['p_update_rate'] = 1
        elif p_par.get('p_update_rate') < 1:
            raise ParamError("p_update_rate must be equal or higher than 1.")
    
        if 'p_num_hidden_layers' not in p_par:
            raise ParamError("p_num_hidden_layers is not defined.")
        
        if 'p_hidden_size' not in p_par:
            raise ParamError("p_hidden_size is not defined.") 
        
        if 'p_activation_fct' not in p_par:
            raise ParamError("p_activation_fct is not defined.")
    
        if 'p_output_activation_fct' not in p_par:
            p_par['p_output_activation_fct'] = None
        
        if 'p_optimizer' not in p_par:
            raise ParamError("p_optimizer is not defined.")
        
        if 'p_loss_fct' not in p_par:
            raise ParamError("p_loss_fct is not defined.")
        
        self._hyperparam_space.add_dim(HyperParam('p_input_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_output_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_update_rate','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_num_hidden_layers','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_hidden_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_activation_fct'))
        self._hyperparam_space.add_dim(HyperParam('p_output_activation_fct'))
        self._hyperparam_space.add_dim(HyperParam('p_optimizer'))
        self._hyperparam_space.add_dim(HyperParam('p_loss_fct'))
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        ids_ = self.get_hyperparam().get_dim_ids()
        self.get_hyperparam().set_value(ids_[0], p_input_size)
        self.get_hyperparam().set_value(ids_[1], p_output_size)
        self.get_hyperparam().set_value(ids_[2], p_par['p_update_rate'])
        self.get_hyperparam().set_value(ids_[3], p_par['p_num_hidden_layers'])
        self.get_hyperparam().set_value(ids_[4], p_par['p_hidden_size'])
        self.get_hyperparam().set_value(ids_[5], p_par['p_activation_fct'])
        self.get_hyperparam().set_value(ids_[6], p_par['p_output_activation_fct'])
        self.get_hyperparam().set_value(ids_[7], p_par['p_optimizer'])
        self.get_hyperparam().set_value(ids_[8], p_par['p_loss_fct'])