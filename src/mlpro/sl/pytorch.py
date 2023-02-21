## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : pytorch.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-21  0.0.0     SY       Creation 
## -- 2023-02-21  0.1.0     SY       Introduction torch-related base classes for adaptive function
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2023-02-21)

This module provides model classes for supervised learning tasks using PyTorch. 
"""


import torch
from mlpro.sl.models import *
from mlpro.bf.ml import *
from mlpro.bf.data import BufferElement





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchSLNetwork (SLNetwork):
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
    def __init__(self, p_input: torch.Tensor, p_output: torch.Tensor):
        super().__init__({"input": p_input, "output": p_output})





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchBuffer(Buffer, torch.utils.data.Dataset):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_size=1):
        Buffer.__init__(self, p_size=p_size)
        self._internal_counter = 0


## -------------------------------------------------------------------------------------------------
    def add_element(self, p_elem: BufferElement):
        Buffer.add_element(self, p_elem)
        self._internal_counter += 1


## -------------------------------------------------------------------------------------------------
    def get_internal_counter(self):
        return self._internal_counter


## -------------------------------------------------------------------------------------------------
    def __getitem__(self,idx):
        return self._data_buffer["input"][idx], self._data_buffer["output"][idx]