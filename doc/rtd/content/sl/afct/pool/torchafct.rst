`Adaptive Function Pytorch <https://github.com/fhswf/MLPro/blob/main/src/mlpro/sl/pool/afct/afct_pytorch.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an adaptive function module based on Pytorch. This module provides functionality to map input based on MLPro.bf.math.Element 
to torch.Tensor. Then the input goes through a Pytorch Neural Network Module. After that, the output is being mapped again to 
MLPro.bf.math.Element. The pytorch neural network module needs to be defined by the user.

Example of implementing a multilayer perceptron (MLP) neural network with this module.

.. code-block:: python

    import torch
    from torch.utils.data.sampler import SubsetRandomSampler
    from mlpro.sl.pool.afct.afct_pytorch import TorchAFct, TorchBufferElement, TorchBuffer

    class MLPAFct(TorchAFct):
        C_NAME = "MLP Adaptive Function"

        # Special Buffer for using Pytorch
        C_BUFFER_CLS = TorchBuffer

        def _setup_model(self):
            self.net_model = torch.nn.Sequential(
                torch.nn.Linear(self._input_space.get_num_dim(), 100),
                torch.nn.Sigmoid(),
                torch.nn.Linear(100, 100),
                torch.nn.Sigmoid(),
                torch.nn.Linear(100, self._output_space.get_num_dim()),
            )

            self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=3e-4)
            self.loss_func = torch.nn.MSELoss()

        def _input_preproc(self, p_input: torch.Tensor) -> torch.Tensor:
            # Do something here for pre-processing input
            input = something
            return input

        def _output_postproc(self, p_output: torch.Tensor) -> torch.Tensor:
            # Do something here for post-processing output
            output = something
            return output

        def _adapt(self, p_input: Element, p_output: Element) -> bool:
            # Create your function how to train your network
            # For example:
            
            # Add input and output to the buffer
            # Special BufferElement for using Pytorch
            self._buffer.add_element(TorchBufferElement(p_input, p_output))

            # Wait until buffer has 100 data, otherwise keep collecting data
            if self._buffer.get_internal_counter() % 100 != 0:
                return False

            # Create Dataset with Pytorch Dataset Loader
            dataset_size = len(self._buffer)
            indices = list(range(dataset_size))
            train_sampler = SubsetRandomSampler(indices)
            trainer = torch.utils.data.DataLoader(self._buffer, batch_size=100, sampler=train_sampler)

            # Train Network once
            # The batch_size is equal to the number of data buffer
            # So the loop below will only run once
            for i, (In, Label) in enumerate(trainer):
                outputs = self.net_model(In)
                loss = self.loss_func(outputs, Label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Clear the buffer
            self._buffer.clear()

            return True
            
            