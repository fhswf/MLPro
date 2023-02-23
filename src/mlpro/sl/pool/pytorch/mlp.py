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
        
        if self._parameters.get('p_activation_fct') not in self._parameters:
            raise ParamError("p_activation_fct is not defined.")
        try:
            if len(self._parameters['p_activation_fct']) != self._parameters['p_num_hidden_layers']:
                raise ParamError("length of p_activation_fct list must be equal to p_num_hidden_layers or a single activation function.")
        except:
            if not isinstance(self._parameters['p_activation_fct'], int):
                raise ParamError("length of p_activation_fct list must be equal to p_num_hidden_layers or a single activation function.")
        
        if self._parameters.get('p_optimizer') not in self._parameters:
            raise ParamError("p_optimizer is not defined.")
        
        if self._parameters.get('p_loss_fct') not in self._parameters:
            raise ParamError("p_loss_fct is not defined.")

        model = None

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
    def _default_weight_bias_init(module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RobotMLPModel(torch.nn.Module):

## -------------------------------------------------------------------------------------------------
    def __init__(self, n_joint, timeStep):
        super(RobotMLPModel, self).__init__()
        self.n_joint = n_joint
        self.timeStep = timeStep
        self.hidden = 128

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.model1 = torch.nn.Sequential(
            init_(torch.nn.Linear(self.n_joint,self.hidden)),
            torch.nn.Tanh(),
            init_(torch.nn.Linear(self.hidden,self.hidden)),
            torch.nn.Tanh(),
            init_(torch.nn.Linear(self.hidden,self.hidden)),
            torch.nn.Tanh(),
            init_(torch.nn.Linear(self.hidden,7*(self.n_joint+1))),
            torch.nn.Tanh()
            )

## -------------------------------------------------------------------------------------------------
    def forward(self, I):
        BatchSize=I.shape[0]
        newI = I.reshape(BatchSize,2,self.n_joint) * torch.cat([torch.Tensor([self.timeStep]).repeat(1,self.n_joint), torch.ones(1,self.n_joint)])
        newI = torch.sum(newI,dim=1)
        out2 = self.model1(newI)   
        out2 = out2.reshape(BatchSize,self.n_joint+1,7)
        return out2