## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.afct
## -- Module  : afct_pytorch
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-17  0.0.0     MRD       Creation
## -- 2021-12-17  1.0.0     MRD       Released first version
## -- 2022-01-02  2.0.0     MRD       Re-released afct for pytorch
## -- 2022-02-18  2.0.1     MRD       Refactor as General Adaptive Function based on Pytorch
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.0 (2022-01-02)

This module provides Adaptive Functions with Neural Network based on Pytorch.
"""

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from mlpro.rl.models import *
from mlpro.sl.pool.afct.afct_base_nn import AFctBaseNN
from mlpro.sl.pool.afct.afct_base_nn import IOElement


class TorchBuffer(Buffer, torch.utils.data.Dataset):
    def __init__(self, p_size=1):
        Buffer.__init__(self, p_size=p_size)
        self._internal_counter = 0

    def add_element(self, p_elem: BufferElement):
        Buffer.add_element(self, p_elem)
        self._internal_counter += 1

    def get_internal_counter(self):
        return self._internal_counter

    def __getitem__(self, idx):
        return self._data_buffer["input"][idx], self._data_buffer["output"][idx]


class TorchAFct(AFctBaseNN):
    C_NAME = "Pytorch based Adaptive Function"
    C_BUFFER_CLS = TorchBuffer

    def __init__(
            self,
            p_input_space: MSpace,
            p_output_space: MSpace,
            p_output_elem_cls=Element,
            p_data_split=0.3,
            p_batch_size=100,
            p_threshold=0,
            p_buffer_size=0,
            p_ada=True,
            p_logging=Log.C_LOG_ALL,
            **p_par
    ):

        self.batch_size = p_batch_size
        self.data_split = p_data_split

        super().__init__(
            p_input_space=p_input_space,
            p_output_space=p_output_space,
            p_output_elem_cls=p_output_elem_cls,
            p_threshold=p_threshold,
            p_buffer_size=p_buffer_size,
            p_ada=p_ada,
            p_logging=p_logging,
            **p_par
        )

    def _adapt(self, p_input: Element, p_output: Element):
        self._buffer.add_element(IOElement(p_input, p_output))

        if self._buffer.get_internal_counter() % self._buffer._size != 0:
            return False

        self.net_model.train()

        dataset_size = len(self._buffer)
        indices = list(range(dataset_size))
        split = int(np.floor(self.data_split * dataset_size))
        np.random.seed(random.randint(1, 1000))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        trainer = torch.utils.data.DataLoader(self._buffer, batch_size=self.batch_size, sampler=train_sampler)
        tester = torch.utils.data.DataLoader(self._buffer, batch_size=self.batch_size, sampler=test_sampler)

        # Training
        self.net_model.train()

        for i, (In, Label) in enumerate(trainer):
            outputs = self.net_model(In)
            loss = self.loss_dyn(outputs, Label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return True

    def input_preproc(self, p_input: Element) -> torch.Tensor:
        # Convert p_input from Element to Tensor
        input = torch.Tensor([p_input.get_values()])

        # Preprocessing Data if needed
        input = self._input_preproc(input)

        return input

    def output_postproc(self, p_output: torch.Tensor) -> list:
        # Output Post Processing
        output = self._output_postproc(p_output)

        # Convert output from Tensor to List
        output = output.detach().flatten().tolist()

        return output

    def _input_preproc(self, p_input: torch.Tensor) -> torch.Tensor:
        return p_input

    def _output_postproc(self, p_output: torch.Tensor) -> torch.Tensor:
        return p_output
