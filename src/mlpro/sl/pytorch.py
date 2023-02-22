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
    def __init__(self, p_size:int=1):
        Buffer.__init__(self, p_size=p_size)
        self._internal_counter = 0


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
## -------------------------------------------------------------------------------------------------
class PyTorchAFct(SLAdaptiveFunction):
    C_NAME = "PyTorch-based Adaptive Function"



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