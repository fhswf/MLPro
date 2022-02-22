## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.afct
## -- Module  : afct_pytorch_mlp
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-17  0.0.0     MRD       Creation
## -- 2021-12-17  1.0.0     MRD       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-01-02)

This module provides Adaptive Functions with Neural Network based on Pytorch.
"""

import torch

from mlpro.rl.models import *
from mlpro.sl.pool.afct.afct_pytorch import TorchAFct


class TorchAFctMLP(TorchAFct):
    C_NAME = "Pytorch based Adaptive Function"

    """
    Template class for an adaptive bi-multivariate mathematical function that adapts by supervised
    learning.

    Parameters
    ----------
    p_input_size : int
        Input size of the neural network
    p_output_size : int
        Output size of the neural network
    p_net_arch : list
        List of hidden layer.
        eg. [10, 10], 2 hidden layers, 10 neurons each

    """

    def __init__(
            self,
            p_input_size: int,
            p_output_size: int,
            p_net_arch: list = [32, 32],
            p_activation_fn=torch.nn.Tanh,
            p_optimizer_class=torch.optim.Adam,
            p_learning_rate=3e-4,
            p_batch_size=100,
            p_threshold=0,
            p_buffer_size=0,
            p_ada=True,
            p_logging=Log.C_LOG_ALL,
            **p_par
    ):
        # Convert input size to input space
        input_space = ESpace()
        for idx in range(p_input_size):
            input_space.add_dim(
                Dimension(idx, "I%i" % (idx), p_boundaries=[-np.inf, np.inf])
            )

        # Convert output size to output space
        output_space = ESpace()
        for idx in range(p_output_size):
            output_space.add_dim(
                Dimension(idx, "O%i" % (idx), p_boundaries=[-np.inf, np.inf])
            )

        self.input_size = p_input_size
        self.net_arch = p_net_arch
        self.output_size = p_output_size
        self.activation_fn = p_activation_fn
        self.optimizer_class = p_optimizer_class
        self.learning_rate = p_learning_rate

        super().__init__(
            p_input_space=input_space,
            p_output_space=output_space,
            p_output_elem_cls=Element,
            p_batch_size=p_batch_size,
            p_threshold=p_threshold,
            p_buffer_size=p_buffer_size,
            p_ada=p_ada,
            p_logging=p_logging,
            **p_par
        )

    def _setup_model(self):
        # Construct Neural Network
        neural_net = []
        last_layer_dim = self.input_size
        for layer in self.net_arch:
            neural_net.append(torch.nn.Linear(last_layer_dim, layer))
            neural_net.append(self.activation_fn)
            last_layer_dim = layer

        self.net_model = torch.nn.Sequential(neural_net)
        self.optimizer = self.optimizer_class(self.net_model.parameters(), lr=self.learning_rate)

        return self.net_model

