## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.pool.pytorch
## -- Module  : mlp.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-23  0.0.0     SY       Creation
## -- 2023-02-23  1.0.0     SY       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-02-23)

This module provides a template ready-to-use MLP model using PyTorch. 
"""


from mlpro.sl.pytorch import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MLP(PyTorchSLNetwork):
    """
    This class provides the base class of a supervised learning network using PyTorch.

    Parameters
    ----------
    p_input_size : int
        Input size of the network. Default = None
    p_output_size : int
        Output size of the network. Default = None
    p_hyperparameter : HyperParamTuple
        Related hyperparameter tuple of the network. Default = None
    p_kwargs : Dict
        Further model specific parameters.
     """

    C_TYPE = 'MLP using PyTorch'
    C_NAME = '????'


## -------------------------------------------------------------------------------------------------
    def _setup_model(self):

        # parameters checking

        if self._parameters.get('p_num_hidden_layers') not in self._parameters:
            raise ParamError("p_num_hidden_layers is not defined.")
        
        if self._parameters.get('p_hidden_size') not in self._parameters:
            raise ParamError("p_hidden_size is not defined.")
        try:
            if len(self._parameters['p_hidden_size']) != self._parameters['p_num_hidden_layers']:
                raise ParamError("length of p_hidden_size list must be equal to p_num_hidden_layers or an integer.")
        except:
            if not isinstance(self._parameters['p_hidden_size'], int):
                raise ParamError("length of p_hidden_size list must be equal to p_num_hidden_layers or an integer.")
            else:
                self._parameters['p_hidden_size'] = [self._parameters['p_hidden_size']] * self._parameters['p_num_hidden_layers']
        
        if self._parameters.get('p_activation_fct') not in self._parameters:
            raise ParamError("p_activation_fct is not defined.")
        try:
            if len(self._parameters['p_activation_fct']) != self._parameters['p_num_hidden_layers']:
                raise ParamError("length of p_activation_fct list must be equal to p_num_hidden_layers or a single activation function.")
        except:
            if isinstance(self._parameters['p_activation_fct'], list):
                raise ParamError("length of p_activation_fct list must be equal to p_num_hidden_layers or a single activation function.")
            else:
                self._parameters['p_activation_fct'] = [self._parameters['p_activation_fct']] * self._parameters['p_num_hidden_layers']
        
        if self._parameters.get('p_custom_layers') in self._parameters and self._parameters['p_custom_layers'] is not None:
            try:
                if len(self._parameters['p_custom_layers']) != self._parameters['p_num_hidden_layers']+1:
                    raise ParamError("length of p_custom_layers list must be equal to p_num_hidden_layers or a single custom layer.")
            except:
                if isinstance(self._parameters['p_custom_layers'], list):
                    raise ParamError("length of p_custom_layers list must be equal to p_num_hidden_layers or a single custom layer.")
                else:
                    self._parameters['p_custom_layers'] = [self._parameters['p_custom_layers']] * (self._parameters['p_num_hidden_layers']+1)
        else:
            self._parameters['p_custom_layers'] = None
        
        if self._parameters.get('p_output_activation_fct') not in self._parameters:
            raise ParamError("p_output_activation_fct is not defined.")
        
        if self._parameters.get('p_optimizer') not in self._parameters:
            raise ParamError("p_optimizer is not defined.")
        else:
            self._optimizer = self._parameters['p_optimizer']
        
        if self._parameters.get('p_loss_fct') not in self._parameters:
            raise ParamError("p_loss_fct is not defined.")
        else:
            self._loss_function = self._parameters['p_loss_fct']
        
        if self._parameters.get('p_weight_init') not in self._parameters:
            self._parameters['p_weight_init'] = torch.nn.init.orthogonal_
        
        if self._parameters.get('p_bias_init') not in self._parameters:
            self._parameters['p_bias_init'] = lambda x: torch.nn.init.constant_(x, 0)
        
        if self._parameters.get('p_gain_init') not in self._parameters:
            self._parameters['p_gain_init'] = np.sqrt(2)
        
        # setting up model

        init_ = lambda mod: self._default_weight_bias_init(mod,
                                                           self._parameters['p_weight_init'],
                                                           self._parameters['p_bias_init'],
                                                           self._parameters['p_gain_init'])

        modules = []
        for hd in range(self._parameters['p_num_hidden_layers']+1):
            if hd == 0:
                act_input_size = self._input_size
                output_size = self._parameters['p_hidden_size'][hd]
                act_fct = self._parameters['p_activation_fct'][hd]
            elif hd == self._parameters['p_num_hidden_layers']:
                act_input_size = self._parameters['p_hidden_size'][hd-1]
                output_size = self._output_size
                act_fct = self._parameters['p_output_activation_fct']
            else:
                act_input_size = self._parameters['p_hidden_size'][hd-1]
                output_size = self._parameters['p_hidden_size'][hd]
                act_fct = self._parameters['p_activation_fct'][hd]
            
            if self._parameters['p_custom_layers'] is None:
                modules.append(init_(torch.nn.Linear(act_input_size, output_size)))
            else:
                modules.append(init_(self._parameters['p_custom_layers'](act_input_size, output_size)))
            modules.append(act_fct)

        model = torch.nn.Sequential(*modules)

        # add process to the model

        try:
            model = self._add_init(model)
        except:
            pass 

        return model


## -------------------------------------------------------------------------------------------------
    def _add_init(self, model):
        """
        This method is optional and is intended for additional initialization process of the
        _setup_model.

        Parameters
        ----------
        model :
            model network

        Returns
        ----------
            updated model network
        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _default_weight_bias_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module


## -------------------------------------------------------------------------------------------------
    def forward(self, p_input:torch.Tensor) -> torch.Tensor:
        BatchSize   = p_input.shape[0]
        output      = self._sl_model(p_input)   
        output      = output.reshape(BatchSize, self._output_size)
        return output