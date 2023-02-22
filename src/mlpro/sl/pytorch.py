## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : pytorch.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-21  0.0.0     SY       Creation 
## -- 2023-02-22  0.1.0     SY       Pre-release version
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2023-02-22)

This module provides model classes for supervised learning tasks using PyTorch. 
"""


import torch
from torch.utils.data.sampler import SubsetRandomSampler
from mlpro.sl.models import *
from mlpro.bf.ml import *
from mlpro.bf.data import BufferElement





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchSLNetwork(SLNetwork):
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

    C_TYPE = 'PyTorchSLNetwork'
    C_NAME = '????'


## -------------------------------------------------------------------------------------------------
    def forward(self, p_input:torch.Tensor) -> torch.Tensor:
        """
        Custom forward propagation in neural networks to generate some output that can be called by
        an external method. Please redefine.

        Parameters
        ----------
        p_input : torch.Tensor
            Input data

        Returns
        ----------
        output : torch.Tensor
            Output data
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchIOElement(BufferElement):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_input:torch.Tensor, p_output:torch.Tensor):
        super().__init__({"input": p_input, "output": p_output})





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchBuffer(Buffer, torch.utils.data.Dataset):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_size:int=1, p_test_data:float=0.3, p_batch_size:int=100, p_seed:int=1):
        Buffer.__init__(self, p_size=p_size)
        self._internal_counter = 0
        self._testing_data     = p_test_data
        self._batch_size       = p_batch_size
        np.random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def add_element(self, p_elem:BufferElement):
        Buffer.add_element(self, p_elem)
        self._internal_counter += 1


## -------------------------------------------------------------------------------------------------
    def get_internal_counter(self):
        return self._internal_counter


## -------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        return self._data_buffer["input"][idx], self._data_buffer["output"][idx]


## -------------------------------------------------------------------------------------------------
    def sampling(self):
        dataset_size    = len(self._data_buffer)
        indices         = list(range(dataset_size))
        split           = int(np.floor(self._testing_data*dataset_size))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        train_sampler   = SubsetRandomSampler(train_indices)
        test_sampler    = SubsetRandomSampler(test_indices)
        trainer         = torch.utils.data.DataLoader(self._data_buffer,
                                                      batch_size=self._batch_size,
                                                      sampler=train_sampler)
        tester          = torch.utils.data.DataLoader(self._data_buffer,
                                                      batch_size=self._batch_size,
                                                      sampler=test_sampler)
        
        return trainer, tester
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchAFct(SLAdaptiveFunction):
    """
    Template class for an adaptive bi-multivariate mathematical function that adapts by supervised
    learning using PyTorch.

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

    C_NAME          = "PyTorch-based Adaptive Function"
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
        
        if p_kwargs.get('p_test_data') is None:
            raise ParamError("p_test_data is not defined.")
        
        if p_kwargs.get('p_batch_size') is None:
            raise ParamError("p_batch_size is not defined.")
        
        if p_kwargs.get('p_seed_buffer') is None:
            raise ParamError("p_seed_buffer is not defined.")
        
        if p_kwargs.get('p_update_rate') is None:
            raise ParamError("p_update_rate is not defined.")
        else:
            self._update_rate = p_kwargs['p_update_rate']
        
        try:
            self._net_model._optimizer
            self._net_model._loss_function
        except:
            raise ParamError("The optimizer and/or loss function of the model are not defined.")

        if p_buffer_size > 0:
            self._buffer = self.C_BUFFER_CLS(p_size=p_buffer_size,
                                             p_test_data=p_kwargs['p_test_data'],
                                             p_batch_size=p_kwargs['p_batch_size'],
                                             p_seed=p_kwargs['p_seed_buffer'])
        else:
            self._buffer = None

## -------------------------------------------------------------------------------------------------
    def input_preproc(self, p_input:Element) -> torch.Tensor:
        # Convert p_input from Element to Tensor
        input = torch.Tensor([p_input.get_values()])

        # Preprocessing Data if needed
        try:
            input = self._input_preproc(input)
        except:
            pass

        return input

## -------------------------------------------------------------------------------------------------
    def output_preproc(self, p_output:Element) -> torch.Tensor:
        # Convert p_input from Element to Tensor
        output = torch.Tensor([p_output.get_values()])

        # Preprocessing Data if needed
        try:
            output = self._output_preproc(output)
        except:
            pass

        return output


## -------------------------------------------------------------------------------------------------
    def output_postproc(self, p_output:torch.Tensor) -> list:
        # Output Post Processing if needed
        try:
            output = self._output_postproc(p_output)
        except:
            output = p_output

        # Convert output from Tensor to a list
        output = output.detach().flatten().tolist()

        return output


## -------------------------------------------------------------------------------------------------
    def _input_preproc(self, p_input:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _output_preproc(self, p_output:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _output_postproc(self, p_output:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _map(self, p_input:Element, p_output:Element):
        # Input pre processing
        input = self.input_preproc(p_input)

        # Make prediction
        output = self._net_model(input)

        # Output post processing
        output = self.output_postproc(output)

        # Set list to Element
        p_output.set_values(output)


## -------------------------------------------------------------------------------------------------
    def _add_buffer(self, p_buffer_element:PyTorchIOElement):
        self._buffer.add_element(p_buffer_element)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_input: Element, p_output: Element) -> bool:
        """
        Adaptation mechanism for PyTorch based model.

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

        model_input = self.input_preproc(p_input)
        model_output = self.output_preproc(p_output)

        self._add_buffer(PyTorchIOElement(model_input, model_output))

        if ( not self._buffer.is_full() ) and ( self._buffer.get_internal_counter()%self._update_rate != 0 ):
            return False

        trainer, _ = self._buffer.sampling()
        self._net_model.train()
        for i, (In, Label) in enumerate(trainer):
            outputs = self._net_model(In)
            loss = self._net_model._loss_function(outputs, Label)
            self._net_model._optimizer.zero_grad()
            loss.backward()
            self._net_model._optimizer.step()
