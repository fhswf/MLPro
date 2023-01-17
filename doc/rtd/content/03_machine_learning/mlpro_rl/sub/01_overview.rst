Overview
--------

MLPro-RL is the first ready-to-use subpackage in MLPro.
MLPro-RL provides complete base classes of the main reinforcement learning (RL) components, e.g. agent, environment, policy, multiagent, and training.
The training loop is developed based on the Markov Decision Process model.
MLPro-RL can handle a broad scope of RL training, including model-free or model-based RL, single-agent or multi-agent RL, and simulation or real hardware mode.
Hence, this subpackage can be a one-stop solution for students, educators, RL engineers or RL researchers to support their RL-related tasks.
The structure of MLPro-RL can be found in the following figure.

.. figure:: images/MLPro-RL_overview.png
  :width: 600
  
  This figure is taken from `MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

Additionally, you can find the more comprehensive explanations of MLPro-RL including a sample application on controlling a UR5 Robot in this paper:
`MLPro 1.0 - Standardized Reinforcement Learning and Game Theory in Python <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

If you are interested to utilize MLPro-RL, you can easily access the RL modules, as follows:

    .. code-block:: python

        from mlpro.rl.models import *

You can also check out numerous ready to run examples on our `how to files <https://mlpro.readthedocs.io/en/latest/content/append1/howto.rl.html>`_
or on our `MLPro GitHub <https://github.com/fhswf/MLPro/tree/main/src/mlpro/rl/examples>`_.
Moreover, a technical API documentation can be found in the appendix 2.