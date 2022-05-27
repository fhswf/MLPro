`Adaptive Function Pytorch <https://github.com/fhswf/MLPro/blob/main/src/mlpro/sl/pool/afct/afct_pytorch.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an adaptive function module based on Pytorch. This module provides functionality to map input based on MLPro.bf.math.Element 
to torch.Tensor. Then the input goes through a Pytorch Neural Network Module. After that, the output is being mapped again to 
MLPro.bf.math.Element. The pytorch neural network module needs to be defined by the user.

Example of implementing a multilayer perceptron (MLP) neural network with this module.