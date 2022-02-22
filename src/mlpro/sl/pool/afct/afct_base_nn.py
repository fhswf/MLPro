## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.afct
## -- Module  : afct_base
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-02-18  0.0.0     MRD       Creation
## -- 2022-02-18  1.0.0     MRD       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-02-18)

This module provides the base function for implementation of Adaptive Function based Neural Network.
"""

from mlpro.rl.models import *


class IOElement(BufferElement):
    def __init__(self, p_input, p_output):
        super().__init__({"input": p_input, "output": p_output})


class AFctBaseNN(AdaptiveFunction):
    C_NAME = "Neural Network based Adaptive Function"

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

        self.net_model = self._setup_model()

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

    def _map(self, p_input: Element, p_output: Element):
        # Input pre processing
        input = self.input_preproc(p_input)

        # Make prediction
        output = self.net_model(input)

        # Output post processing
        output = self.output_postproc(output)

        p_output.set_values(output)
