## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.feedforwardNN
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-01  0.0.0     SY       Creation 
## -- 2023-03-01  0.1.0     SY       Initial design of FNN for MLPro v1.0.0
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2023-03-01)

This module provides model classes of feedforward neural networks for supervised learning tasks. 
"""


from mlpro.sl import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FeedforwardNN(SLAdaptiveFunction):
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
    def _map(self, p_input: Element, p_output: Element):
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
        
        return self._parameters['loss_fct'](p_act_output.get_values(), p_pred_output.get_values())






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MLP(FeedforwardNN):
    """
    This class provides the base class of multilayer perceptron.
    """

    C_TYPE = 'Multilayer Perceptron'


## -------------------------------------------------------------------------------------------------
    def _hyperparameters_check(self) -> dict:
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
