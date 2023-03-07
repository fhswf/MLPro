## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.pool.FNN.pytorch
## -- Module  : mlp.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-23  0.0.0     SY       Creation
## -- 2023-03-07  1.0.0     SY       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-07)

This module provides a template ready-to-use MLP model using PyTorch. 
"""


from mlpro.sl.pool.afct.pytorch import *
from mlpro.sl import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchMLP(MLP, PyTorchHelperFunctions):
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
        
        self._output_space.distance = self._calc_loss
        
        if p_buffer_size > 0:
            self._buffer = self.C_BUFFER_CLS(p_size=p_buffer_size,
                                             p_test_data=self._parameters['test_data'],
                                             p_batch_size=self._parameters['batch_size'],
                                             p_seed=self._parameters['seed_buffer'])
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

        # additional pytorch-related parameters checking
        self._add_hyperparameters_check()
        
        # setting up model
        if self._parameters['weight_bias_init']:
            init_ = lambda mod: self._default_weight_bias_init(mod,
                                                               self._parameters['weight_init'],
                                                               self._parameters['bias_init'],
                                                               self._parameters['gain_init'])

        modules = []
        for hd in range(int(self._parameters['num_hidden_layers'])+1):
            if hd == 0:
                act_input_size  = self._parameters['input_size']
                output_size     = self._parameters['hidden_size'][hd]
                act_fct         = self._parameters['activation_fct'][hd]()
            elif hd == self._parameters['num_hidden_layers']:
                act_input_size  = self._parameters['hidden_size'][hd-1]
                output_size     = self._parameters['output_size']
                act_fct         = self._parameters['output_activation_fct']()
            else:
                act_input_size  = self._parameters['hidden_size'][hd-1]
                output_size     = self._parameters['hidden_size'][hd]
                act_fct         = self._parameters['activation_fct'][hd]()
            
            if self._parameters['weight_bias_init']:
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
        
        self._parameters['loss_fct'] = self._parameters['loss_fct']()
        self.optimizer = self._parameters['optimizer'](model.parameters(), lr=self._parameters['learning_rate'])

        return model


## -------------------------------------------------------------------------------------------------
    def _add_hyperparameters_check(self) -> bool:
        """
        Additional hyperparameters related to the PyTorch-based MLP model checking.

        Hyperparameters
        ----------
        test_data : float
            the proportion of test data during the sampling process. Default = 0.3.
        batch_size : int
            batch size of the buffer. Default = 100.
        seed_buffer : int
            seeding of the buffer. Default = 1.
        learning_rate : float
            learning rate of the optimizer. Default = 3e-4.
        hidden_size : int or list
            number of hidden neurons. There are two possibilities to set up the hidden size:
            1) if hidden_size is an integer, then the number of neurons in all hidden layers are
            exactly the same.
            2) if hidden_size is in a list, then the user can define the number of neurons in
            each hidden layer, but make sure that the length of the list must be equal to the
            number of hidden layers.
        activation_fct : torch.nn or list
            activation function. There are two possibilities to set up the activation function:
            1) if activation_fct is a single activation function, then the activation function after all
            hidden layers are exactly the same.
            2) if activation_fct is in a list, then the user can define the activation function after
            each hidden layer, but make sure that the length of the list must be equal to the
            number of hidden layers.
        weight_bias_init : bool, optional
            weight and bias initialization. Default : True
        weight_init : torch.nn, optional
            weight initialization function. Default : torch.nn.init.orthogonal_
        bias_init : torch.nn, optional
            bias initilization function. Default : lambda x: torch.nn.init.constant_(x, 0)
        gain_init : int, optional
            gain parameter of the weight and bias initialization. Default : np.sqrt(2)
        
        Returns
        ----------
        bool
            True, if everything is succesful.
        """

        _param = {}
        hp = self.get_hyperparam()
        for idx in hp.get_dim_ids():
            par_name = hp.get_related_set().get_dim(idx).get_name_short()
            par_val  = hp.get_value(idx)
            _param[par_name] = par_val

        if 'test_data' not in _param:
            self._parameters['test_data'] = 0.3
        else:
            self._parameters['test_data'] = _param['test_data']

        if 'batch_size' not in _param:
            self._parameters['batch_size'] = 100
        else:
            self._parameters['batch_size'] = _param['batch_size']

        if 'seed_buffer' not in _param:
            self._parameters['seed_buffer'] = 1
        else:
            self._parameters['seed_buffer'] = _param['seed_buffer']

        if 'learning_rate' not in _param:
            self._parameters['learning_rate'] = 3e-4
        else:
            self._parameters['learning_rate'] = _param['learning_rate']
        
        if 'hidden_size' not in self._parameters:
            raise ParamError("hidden_size is not defined.")
        try:
            if len(self._parameters['hidden_size']) != self._parameters['num_hidden_layers']:
                raise ParamError("length of hidden_size list must be equal to num_hidden_layers or an integer.")
        except:
            self._parameters['hidden_size'] = [int(self._parameters['hidden_size'])] * int(self._parameters['num_hidden_layers'])
        
        if 'activation_fct' not in self._parameters:
            raise ParamError("activation_fct is not defined.")
        try:
            if len(self._parameters['activation_fct']) != self._parameters['num_hidden_layers']:
                raise ParamError("length of p_activation_fct list must be equal to p_num_hidden_layers or a single activation function.")
        except:
            if isinstance(self._parameters['activation_fct'], list):
                raise ParamError("length of activation_fct list must be equal to num_hidden_layers or a single activation function.")
            else:
                self._parameters['activation_fct'] = [self._parameters['activation_fct']] * int(self._parameters['num_hidden_layers'])
        
        if 'weight_bias_init' not in _param:
            self._parameters['weight_bias_init'] = True
        else:
            self._parameters['weight_bias_init'] = _param['weight_bias_init']
        
        if self._parameters['weight_bias_init']:
            if 'weight_init' not in _param:
                self._parameters['weight_init'] = torch.nn.init.orthogonal_
            else:
                self._parameters['weight_init'] = _param['weight_init']
            
            if 'bias_init' not in _param:
                self._parameters['bias_init'] = lambda x: torch.nn.init.constant_(x, 0)
            else:
                self._parameters['bias_init'] = _param['bias_init']
            
            if 'gain_init' not in _param:
                self._parameters['gain_init'] = np.sqrt(2)
            else:
                self._parameters['gain_init'] = _param['gain_init']
        
        return True


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
    def _map(self, p_input:Element, p_output:Element):
        """
        Maps a multivariate abscissa/input element to a multivariate ordinate/output element. 

        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)
        """

        # Input pre processing
        input = self.input_preproc(p_input)

        # Make prediction
        output = self.forward(input)

        # Output post processing
        output = self.output_postproc(output)

        # Set list to Element
        p_output.set_values(output)


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
    def _calc_loss(self, p_act_output:Element, p_pred_output:Element) -> float:
        """
        This method has a functionality to evaluate the adapted SL model. 

        Parameters
        ----------
        p_act_output : Element
            Actual output from the buffer.
        p_pred_output : Element
            Predicted output by the SL model.
        """

        model_act_output  = self.output_preproc(p_act_output)
        model_pred_output = self.output_preproc(p_pred_output)

        return self._parameters['loss_fct'](model_act_output, model_pred_output)


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

        self._add_buffer(PyTorchIOElement(model_input, model_output))

        if ( not self._buffer.is_full() ) and ( self._buffer.get_internal_counter()%self._parameters['update_rate'] != 0 ):
            return False

        trainer, _  = self._buffer.sampling()
        self._adapt_offline(trainer)
        return True


## -------------------------------------------------------------------------------------------------
    def _adapt_offline(self, p_dataset:torch.Tensor) -> bool:
        """
        Adaptation mechanism for PyTorch based model for offline learning.
        """

        self._sl_model.train()
        for i, (In, Label) in enumerate(p_dataset):
            outputs = self.forward(In)
            loss    = self._calc_loss(outputs, Label)
            self._optimize(loss)
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
        output      = output.reshape(BatchSize, int(self._parameters['output_size']))
        return output