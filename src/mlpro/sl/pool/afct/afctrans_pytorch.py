## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.afct
## -- Module  : afctrans_pytorch
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-17  0.0.0     MRD       Creation
## -- 2021-12-17  1.0.0     MRD       Released first version
## -- 2022-01-02  2.0.0     MRD       Re-released afct for pytorch
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.0 (2022-01-02)

This module provides Adaptive Functions for state transition with Neural Network based on Pytorch.
"""

import torch

from mlpro.rl.models import *

class TorchAFctTrans(AdaptiveFunction):
    C_NAME = "Pytorch based Adaptive Function"

    def __init__(
        self,
        p_input_space: MSpace,
        p_output_space: MSpace,
        p_output_elem_cls=Element,
        p_threshold=0,
        p_buffer_size=0,
        p_ada=True,
        p_logging=Log.C_LOG_ALL,
        **p_par
    ):

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

        self._setup_model()

        if self.net_model is None:
            raise ParamError("Please assign your network model to self.net_model")
        
    
    def _setup_model(self):
        """
        Setup Neural Network.
        
        Here, the user needs to implement the Neural Network structure and all the necessary variable
        to train the neural network, e.g, optimizer, loss function.

        Input space can be obtained from the following variable.
        self._input_space

        Output space can be obtained from the following variable.
        self._output_space

        Please return the neural network.
        """

        return None

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

    def _map(self, p_input: Element, p_output: Element):
        # Input pre processing
        input = self.input_preproc(p_input)

        # Make prediction
        output = self.net_model(input)

        # Output post processing
        output = self.output_postproc(output)

        p_output.set_values(output)