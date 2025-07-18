## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.sl.pool.FNN.pytorch
## -- Module  : mlp.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-23  0.0.0     SY       Creation
## -- 2023-03-07  1.0.0     SY       Released first version
## -- 2023-03-10  1.1.0     SY       Combining _hyperparameters_check and _init_hyperparam
## -- 2023-03-28  1.2.0     SY       - Add _complete_state, _reduce_state due to new class Persistent
## --                                - Update _map
## -- 2023-05-03  1.2.1     SY       Updating _adapt_offline method
## -- 2023-06-21  1.2.2     LSB      Updating _adapt_offline method
## -- 2023-07-04  1.2.3     LSB      Refactoring _complete_state for path conflict
## -- 2023-07-14  1.2.4     LSB      Refactoring for afct fct parameter, so it is provided after instanciating
## -- 2025-07-07  1.3.0     DA       Refactoring of method PyTorchMLP._map()
## -- 2025-07-18  1.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2025-07-18)

This module provides a template ready-to-use MLP model using PyTorch. 
"""

import numpy as np
import torch

from mlpro.bf import Log, ParamError
from mlpro.bf.math import Element, MSpace
from mlpro.bf.ml import HyperParam, HyperParamTuple

from mlpro.sl.pool.afct.pytorch import *
from mlpro.sl.fnn import MLP



# Export list for public API
__all__ = [ 'PyTorchMLP' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchMLP (MLP, PyTorchHelperFunctions):
    """
    Template class for an adaptive bi-multivariate mathematical function that adapts by
    supervised learning using PyTorch-based MLP.

    Parameters
    ----------
    p_input_space : MSpace
        Input space of function
    p_output_space : MSpace
        Output space of function
    p_output_elem_cls 
        Output element class (compatible to/inherited from class Element)
    p_threshold : float
        Threshold for the difference between a setpoint and a computed output. Computed outputs with 
        a difference less than this threshold will be assessed as 'good' outputs. Default = 0.
    p_buffer_size : int
        Initial size of internal data buffer. Default = 0 (no buffering).
    p_ada : bool
        Boolean switch for adaptivity. Default = True.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : Dict
        Further model specific parameters (to be specified in child class).
    """

    C_TYPE          = "PyTorch-based Adaptive Function using MLP"
    C_BUFFER_CLS    = PyTorchBuffer


## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_input_space: MSpace,
                  p_output_space:MSpace,
                  p_output_elem_cls=Element,
                  p_threshold=0,
                  p_buffer_size=0,
                  p_ada:bool=True,
                  p_visualize:bool=False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        super().__init__( p_input_space=p_input_space,
                          p_output_space=p_output_space,
                          p_output_elem_cls=p_output_elem_cls,
                          p_threshold=p_threshold,
                          p_buffer_size=p_buffer_size,
                          p_ada=p_ada,
                          p_visualize=p_visualize,
                          p_logging=p_logging,
                          **p_kwargs )
        
        if p_buffer_size > 0:
            ids_            = self.get_hyperparam().get_dim_ids()
            self._buffer    = self.C_BUFFER_CLS(p_size=p_buffer_size,
                                                p_test_data=self.get_hyperparam().get_value(ids_[9]),
                                                p_batch_size=self.get_hyperparam().get_value(ids_[10]),
                                                p_seed=self.get_hyperparam().get_value(ids_[11]))
        else:
            self._buffer = None


## -------------------------------------------------------------------------------------------------
    def _setup_model(self) -> torch.nn.Sequential:
        """
        A method to set up a supervised learning network.
        
        Returns
        ----------
            A set up supervised learning model
        """

        # setting up model
        ids_ = self.get_hyperparam().get_dim_ids()
        if self.get_hyperparam().get_value(ids_[13]):
            init_ = lambda mod: self._default_weight_bias_init(mod,
                                                               self.get_hyperparam().get_value(ids_[14]),
                                                               self.get_hyperparam().get_value(ids_[15]),
                                                               self.get_hyperparam().get_value(ids_[16]))

        modules = []
        for hd in range(int(self.get_hyperparam().get_value(ids_[3]))+1):
            if hd == 0:
                act_input_size  = self.get_hyperparam().get_value(ids_[0])
                output_size     = self.get_hyperparam().get_value(ids_[4])[hd]
                act_fct         = self.get_hyperparam().get_value(ids_[5])[hd]
            elif hd == self.get_hyperparam().get_value(ids_[3]):
                act_input_size  = self.get_hyperparam().get_value(ids_[4])[hd-1]
                output_size     = self.get_hyperparam().get_value(ids_[1])
                act_fct         = self.get_hyperparam().get_value(ids_[6])
            else:
                act_input_size  = self.get_hyperparam().get_value(ids_[4])[hd-1]
                output_size     = self.get_hyperparam().get_value(ids_[4])[hd]
                act_fct         = self.get_hyperparam().get_value(ids_[5])[hd]
            
            if self.get_hyperparam().get_value(ids_[13]):
                modules.append(init_(torch.nn.Linear(int(act_input_size), int(output_size))))
            else:
                modules.append(torch.nn.Linear(int(act_input_size), int(output_size)))
            modules.append(act_fct)

        model = torch.nn.Sequential(*modules)
        
        # add process to the model
        try:
            model = self._add_init(model)
        except:
            pass 
        
        self._loss_fct      = self.get_hyperparam().get_value(ids_[8])()
        self._optimizer     = self.get_hyperparam().get_value(ids_[7])(model.parameters(), lr=self.get_hyperparam().get_value(ids_[12]))
        self._sampling_seed = self.get_hyperparam().get_value(ids_[11])
        
        return model
    

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
        p_output_activation_fct :
            extra activation function for the output layer.
        p_optimizer :
            optimizer.
        p_loss_fct :
            loss function.
        p_test_data : float
            the proportion of test data during the sampling process. Default = 0.3.
        p_batch_size : int
            batch size of the buffer. Default = 100.
        p_seed_buffer : int
            seeding of the buffer. Default = 1.
        p_learning_rate : float
            learning rate of the optimizer. Default = 3e-4.
        p_hidden_size : int or list
            number of hidden neurons. There are two possibilities to set up the hidden size:
            1) if hidden_size is an integer, then the number of neurons in all hidden layers are
            exactly the same.
            2) if hidden_size is in a list, then the user can define the number of neurons in
            each hidden layer, but make sure that the length of the list must be equal to the
            number of hidden layers.
        p_activation_fct : torch.nn or list
            activation function. There are two possibilities to set up the activation function:
            1) if activation_fct is a single activation function, then the activation function after all
            hidden layers are exactly the same.
            2) if activation_fct is in a list, then the user can define the activation function after
            each hidden layer, but make sure that the length of the list must be equal to the
            number of hidden layers.
        p_weight_bias_init : bool, optional
            weight and bias initialization. Default : True
        p_weight_init : torch.nn, optional
            weight initialization function. Default : torch.nn.init.orthogonal_
        p_bias_init : torch.nn, optional
            bias initilization function. Default : lambda x: torch.nn.init.constant_(x, 0)
        p_gain_init : int, optional
                gain parameter of the weight and bias initialization. Default : np.sqrt(2)
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
    
        if 'p_output_activation_fct' not in p_par:
            p_par['p_output_activation_fct'] = None
        
        if 'p_optimizer' not in p_par:
            raise ParamError("p_optimizer is not defined.")
        
        if 'p_loss_fct' not in p_par:
            raise ParamError("p_loss_fct is not defined.")

        if 'p_test_data' not in p_par:
            p_par['p_test_data'] = 0.3

        if 'p_batch_size' not in p_par:
            p_par['p_batch_size'] = 100

        if 'p_seed_buffer' not in p_par:
            p_par['p_seed_buffer'] = 1

        if 'p_learning_rate' not in p_par:
            p_par['p_learning_rate'] = 3e-4
        
        if 'p_hidden_size' not in p_par:
            raise ParamError("p_hidden_size is not defined.")
        try:
            if len(p_par['p_hidden_size']) != p_par['p_num_hidden_layers']:
                raise ParamError("length of p_hidden_size list must be equal to p_num_hidden_layers or an integer.")
        except:
            p_par['p_hidden_size'] = [int(p_par['p_hidden_size'])] * int(p_par['p_num_hidden_layers'])
        
        if 'p_activation_fct' not in p_par:
            raise ParamError("p_activation_fct is not defined.")
        try:
            if len(p_par['p_activation_fct']) != p_par['p_num_hidden_layers']:
                raise ParamError("length of p_activation_fct list must be equal to p_num_hidden_layers or a single activation function.")
        except:
            if isinstance(p_par['p_activation_fct'], list):
                raise ParamError("length of p_activation_fct list must be equal to p_num_hidden_layers or a single activation function.")
            else:
                p_par['p_activation_fct'] = [p_par['p_activation_fct']] * int(p_par['p_num_hidden_layers'])
        
        if 'p_weight_bias_init' not in p_par:
            p_par['p_weight_bias_init'] = True
        
        if p_par['p_weight_bias_init']:
            if 'p_weight_init' not in p_par:
                p_par['p_weight_init'] = torch.nn.init.orthogonal_
            
            if 'p_bias_init' not in p_par:
                p_par['p_bias_init'] = lambda x: torch.nn.init.constant_(x, 0)
            
            if 'p_gain_init' not in p_par:
                p_par['p_gain_init'] = np.sqrt(2)
        
        self._hyperparam_space.add_dim(HyperParam('p_input_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_output_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_update_rate','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_num_hidden_layers','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_hidden_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('p_activation_fct'))
        self._hyperparam_space.add_dim(HyperParam('p_output_activation_fct'))
        self._hyperparam_space.add_dim(HyperParam('p_optimizer'))
        self._hyperparam_space.add_dim(HyperParam('p_loss_fct'))
        self._hyperparam_space.add_dim(HyperParam('p_test_data'))
        self._hyperparam_space.add_dim(HyperParam('p_batch_size'))
        self._hyperparam_space.add_dim(HyperParam('p_seed_buffer'))
        self._hyperparam_space.add_dim(HyperParam('p_learning_rate'))
        self._hyperparam_space.add_dim(HyperParam('p_weight_bias_init'))
        self._hyperparam_space.add_dim(HyperParam('p_weight_init'))
        self._hyperparam_space.add_dim(HyperParam('p_bias_init'))
        self._hyperparam_space.add_dim(HyperParam('p_gain_init'))
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
        self.get_hyperparam().set_value(ids_[9], p_par['p_test_data'])
        self.get_hyperparam().set_value(ids_[10], p_par['p_batch_size'])
        self.get_hyperparam().set_value(ids_[11], p_par['p_seed_buffer'])
        self.get_hyperparam().set_value(ids_[12], p_par['p_learning_rate'])
        self.get_hyperparam().set_value(ids_[13], p_par['p_weight_bias_init'])
        self.get_hyperparam().set_value(ids_[14], p_par['p_weight_init'])
        self.get_hyperparam().set_value(ids_[15], p_par['p_bias_init'])
        self.get_hyperparam().set_value(ids_[16], p_par['p_gain_init'])


## -------------------------------------------------------------------------------------------------
    def _add_init(self, p_model):
        """
        This method is optional and is intended for additional initialization process of the
        _setup_model.

        Parameters
        ----------
        p_model :
            model network

        Returns
        ----------
            updated model network
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _map(self, p_input:Element, p_output:Element, p_dim=None):
        """
        Maps a multivariate abscissa/input element to a multivariate ordinate/output element. 

        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)
        """
        
        self._sl_model.eval()

        # Input pre processing
        input = self.input_preproc(p_input)

        # Make prediction
        output = self.forward(input)

        # Output post processing
        output = self.output_postproc(output)

        # Set list to Element
        p_output.set_values(output)

        return p_output


## -------------------------------------------------------------------------------------------------
    def _add_buffer(self, p_buffer_element:PyTorchIOElement):
        """
        This method has a functionality to add data into the buffer.

        Parameters
        ----------
        p_buffer_element : PyTorchIOElement
            An element of PyTorchBuffer.
        """

        self._buffer.add_element(p_buffer_element)


## -------------------------------------------------------------------------------------------------
    def _calc_loss(self, p_act_output:torch.Tensor, p_pred_output:torch.Tensor) -> float:
        """
        This method has a functionality to evaluate the adapted SL model. 

        Parameters
        ----------
        p_act_output : torch.Tensor
            Actual output from the buffer.
        p_pred_output : torch.Tensor
            Predicted output by the SL model.
        """

        return self._loss_fct(p_act_output, p_pred_output)


## -------------------------------------------------------------------------------------------------
    def _optimize(self, p_loss):
        """
        This method provides provide a funtionality to call the optimizer of the feedforward network.
        """
        
        self._optimizer.zero_grad()
        p_loss.backward()
        self._optimizer.step()


## -------------------------------------------------------------------------------------------------
    def _adapt_online(self, p_input: Element, p_output: Element) -> bool:
        """
        Adaptation mechanism for PyTorch based model for online learning.

        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)

        Returns
        ----------
            bool
        """

        model_input   = self.input_preproc(p_input)
        model_output  = self.output_preproc(p_output)
        ids_          = self._hyperparam_tuple.get_dim_ids()

        self._add_buffer(PyTorchIOElement(model_input, model_output))
        
        if ( not self._buffer.is_full() ) or ( self._buffer.get_internal_counter()%self.get_hyperparam().get_value(ids_[2]) != 0 ):
            return False

        trainer, _  = self._buffer.sampling()
        
        # Future work: form trainer into DataSet format, once DataSet has been standardized.
        self._adapt_offline(trainer)
        
        return True


## -------------------------------------------------------------------------------------------------
    def _adapt_offline(self, p_dataset:dict) -> bool:
        """
        Adaptation mechanism for PyTorch based model for offline learning.

        Parameters
        ----------
        p_dataset : dict
            a dictionary that consists of a set of data, which are splitted to 2 keys such as input
            and output. The value of each key is a torch.Tensor of the sampled data.

        Returns
        ----------
            bool
        """

        self._sl_model.train()
        
        for input, target in p_dataset:

            try:
                input = torch.tensor(input.get_values(), dtype=torch.float)
                target = torch.tensor(target.get_values(), dtype=torch.float)
            except:
                pass

            torch.manual_seed(self._sampling_seed)
            outputs = self.forward(torch.squeeze(input))
            
            torch.manual_seed(self._sampling_seed)
            self._loss    = self._calc_loss(outputs, torch.squeeze(target))

            self._optimize(self._loss)

            if isinstance(self._loss, torch.Tensor):
                self._loss = self._loss.item()

            self._sampling_seed += 1
            
        return True


## -------------------------------------------------------------------------------------------------
    def forward(self, p_input:torch.Tensor) -> torch.Tensor:
        """
        Forward propagation in neural networks to generate some output using PyTorch.

        Parameters
        ----------
        p_input : Element
            Input data

        Returns
        ----------
        output : Element
            Output data
        """

        BatchSize   = p_input.shape[0]
        output      = self._sl_model(p_input)   
        ids_        = self.get_hyperparam().get_dim_ids()
        # output      = output.reshape(BatchSize, int(self.get_hyperparam().get_value(ids_[1])))
        
        return output


## -------------------------------------------------------------------------------------------------
    def _reduce_state(self, p_state:dict, p_path:str, p_os_sep:str, p_filename_stub:str):
        
        torch.save(p_state['_sl_model'].state_dict(),
                   p_path + p_os_sep + p_filename_stub + '_model.pt')
        # print(p_state['_sl_model'].state_dict())

        torch.save(p_state['_optimizer'].state_dict(),
                   p_path + p_os_sep + p_filename_stub + '_optimizer.pt')
        
        del p_state['_sl_model']
        del p_state['_optimizer']


## -------------------------------------------------------------------------------------------------
    def _complete_state(self, p_path:str, p_os_sep:str, p_filename_stub:str):
        try:
            load_model = torch.load(p_path + p_os_sep + p_filename_stub + '_model.pt')
        except:
            load_model      = torch.load(p_path + p_os_sep + 'model' + p_os_sep + p_filename_stub + '_model.pt')
        # print(load_model)
        try:
            load_optim = torch.load(p_path + p_os_sep + p_filename_stub + '_optimizer.pt')
        except:
            load_optim      = torch.load(p_path + p_os_sep + 'model' + p_os_sep + p_filename_stub + '_optimizer.pt')
        self._sl_model  = self._setup_model()
        self._sl_model.load_state_dict(load_model)
        self._optimizer.load_state_dict(load_optim)
        