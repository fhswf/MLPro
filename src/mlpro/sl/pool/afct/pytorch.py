## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.sl.pool.afct
## -- Module  : pytorch.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-17  0.0.0     MRD       Creation
## -- 2021-12-17  1.0.0     MRD       Released first version
## -- 2022-01-02  2.0.0     MRD       Re-released afct for pytorch
## -- 2022-05-22  2.0.1     MRD       Renamed Class to TorchAFct
## -- 2022-11-15  2.0.2     DA        Class TorchAFct: new parent class SLAdaptiveFunction
## -- 2023-02-22  3.0.0     SY        - Shifted from mlpro.sl.afct
## --                                 - Release the third version
## -- 2023-03-02  3.0.1     SY        Updating and shifting from mlpro.sl.models
## -- 2023-03-10  3.0.2     SY        Refactoring PyTorchBuffer
## -- 2023-04-09  3.0.3     SY        Refactoring
## -- 2023-05-03  3.0.4     SY        Updating sampling method
## -- 2023-06-20  3.0.5     LSB       Updating the sampling method
## -- 2023-07-02  3.0.6     LSB       Refactoring the postproc and preproc methods
## -- 2023-07-14  3.0.7     LSB       Bug Fix
## -- 2025-07-18  3.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 3.1.0 (2025-07-18)

This a helper module for supervised learning models using PyTorch. 
"""


import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from mlpro.bf.data import BufferElement, Buffer
from mlpro.bf.math import Element

from mlpro.sl import *



# Export list for public API
__all__ = [ 'PyTorchIOElement',
            'PyTorchBuffer',
            'PyTorchHelperFunctions' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchIOElement (BufferElement):
    """
    This class provides a buffer element for PyTorch based SLNetwork.

    Parameters
    ----------
    p_input : Element
        Abscissa/input element object (type Element)
    p_output : Element
        Setpoint ordinate/output element (type Element)
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_input:torch.Tensor, p_output:torch.Tensor):
        super().__init__({"input": p_input, "output": p_output})





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchBuffer (Buffer, torch.utils.data.Dataset):
    """
    This class provides buffer functionalities for PyTorch based SLNetwork and also using several
    built-in PyTorch functionalities.

    Parameters
    ----------
    p_size : int
        the buffer size. Default = 1.
    p_test_data : float
        the proportion of testing data within the sampled data. Default = 0.3.
    p_batch_size : int
        the batch size for a sample. Default = 100.
    p_seed : int
        the seeding for randomizer in the buffer, optional. Default = 1.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_size:int=1, p_test_data:float=0.3, p_batch_size:int=100, p_seed:int=1):
        Buffer.__init__(self, p_size=p_size)
        self._internal_counter = 0
        self._testing_data     = p_test_data
        self._batch_size       = p_batch_size
        np.random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def add_element(self, p_elem:BufferElement):
        """
        This method has a functionality to add an element to the buffer.

        Parameters
        ----------
        p_elem : BufferElement
            an element of the buffer
        """

        Buffer.add_element(self, p_elem)
        self._internal_counter += 1


## -------------------------------------------------------------------------------------------------
    def get_internal_counter(self) -> int:
        """
        This method has a functionality to get the number of elements being added to the buffer.
        """

        return self._internal_counter


## -------------------------------------------------------------------------------------------------
    def __getitem__(self, idx:int):
        """
        This method has a functionality to get an item from the buffer with a specific index.

        Parameters
        ----------
        idx : int
            an index of the buffer
        """

        return self._data_buffer["input"][idx], self._data_buffer["output"][idx]


## -------------------------------------------------------------------------------------------------
    def sampling(self):
        """
        This method has a functionality to sample from the buffer using built-in PyTorch
        functionalities.

        Returns
        ----------
        trainer : dict
            a dictionary that consists of sampled data for training, which are splitted to 2 keys
            such as input and output. The value of each key is a torch's DataLoader of the sampled
            data.
        tester : dict
            a dictionary that consists of sampled data for testing, which are splitted to 2 keys
            such as input and output. The value of each key is a torch's DataLoader of the sampled
            data.
        """

        dataset_size    = len(self._data_buffer["input"])
        indices         = list(range(dataset_size))
        split           = int(np.floor(self._testing_data*dataset_size))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        train_sampler   = SubsetRandomSampler(train_indices)
        test_sampler    = SubsetRandomSampler(test_indices)
        trainer         = {}
        tester          = {}
        
        trainer = torch.utils.data.DataLoader(self,
                                              batch_size=self._batch_size,
                                              sampler=train_sampler
                                              )
        # trainer["output"] = torch.utils.data.DataLoader(self._data_buffer["output"],
        #                                                 batch_size=self._batch_size,
        #                                                 sampler=train_sampler
        #                                                 )
        tester = torch.utils.data.DataLoader(self,
                                             batch_size=self._batch_size,
                                             sampler=test_sampler
                                             )
        # tester["output"] = torch.utils.data.DataLoader(self._data_buffer["output"],
        #                                                batch_size=self._batch_size,
        #                                                sampler=test_sampler
        #                                                )
        
        return trainer, tester
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PyTorchHelperFunctions():
    """
    PyTorch Helper Functions in MLPro-SL.
    """

## -------------------------------------------------------------------------------------------------
    def input_preproc(self, p_input:Element) -> torch.Tensor:
        """
        This method has a functionality to transform input data in the form of Element to
        torch.Tensor for pre-processing.

        Parameters
        ----------
        p_input : Element
            Input data in the form of Element.

        Returns
        ----------
        input : torch.Tensor
            Input data in the form of torch.Tensor.
        """

        # Convert p_input from Element to Tensor
        # input = torch.Tensor(np.array([p_input.get_values()]))
        input = torch.Tensor(np.array(p_input.get_values()))

        # Preprocessing Data if needed
        try:
            input = self._input_preproc(input)
        except:
            pass

        return input


## -------------------------------------------------------------------------------------------------
    def output_preproc(self, p_output:Element) -> torch.Tensor:
        """
        This method has a functionality to transform output data in the form of Element to
        torch.Tensor for pre-processing.

        Parameters
        ----------
        p_output : Element
            Output data in the form of Element.

        Returns
        ----------
        output : torch.Tensor
            Output data in the form of torch.Tensor.
        """

        # Convert p_output from Element to Tensor
        output = torch.Tensor(np.array([p_output.get_values()]))

        # Preprocessing Data if needed
        try:
            output = self._output_preproc(output)
        except:
            pass

        return output


## -------------------------------------------------------------------------------------------------
    def output_postproc(self, p_output:torch.Tensor) -> list:
        """
        This method has a functionality to transform output data in the form of torch.Tensor to
        a list for post-processing.

        Parameters
        ----------
        p_output : torch.Tensor
            Output data in the form of torch.Tensor.

        Returns
        ----------
        output : list
            Output data in the form of list.
        """

        # Output Post Processing if needed
        try:
            output = self._output_postproc(p_output)
        except:
            output = p_output

        # Convert output from Tensor to a list
        output = output.detach().tolist()

        return output


## -------------------------------------------------------------------------------------------------
    def _input_preproc(self, p_input:torch.Tensor) -> torch.Tensor:
        """
        Additional process of input_preproc. This is optional. Please redefine if you need it.

        Parameters
        ----------
        p_input : torch.Tensor
            Input data in the form of torch.Tensor.

        Returns
        ----------
        input : torch.Tensor
            Processed input data in the form of torch.Tensor.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _output_preproc(self, p_output:torch.Tensor) -> torch.Tensor:
        """
        Additional process of output_preproc. This is optional. Please redefine if you need it.

        Parameters
        ----------
        p_output : torch.Tensor
            Output data in the form of torch.Tensor.

        Returns
        ----------
        output : torch.Tensor
            Processed output data in the form of torch.Tensor.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _output_postproc(self, p_output:torch.Tensor) -> torch.Tensor:
        """
        Additional process of output_postproc. This is optional. Please redefine if you need it.

        Parameters
        ----------
        p_output : torch.Tensor
            Output data in the form of torch.Tensor.

        Returns
        ----------
        output : torch.Tensor
            Processed output data in the form of torch.Tensor.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _default_weight_bias_init(self, module, weight_init, bias_init, gain=1):
        """
        Weight and bias initialization method.
        """

        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module
