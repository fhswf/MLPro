.. _target_mpc_RL:
`MPC <https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/pool/actionplanner/mpc.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We introduce an MPC method as action planner in the model-based RL territory.
Monte Carlo MPC is a control algorithm that uses a Monte Carlo simulation-based approach to generate control actions for a dynamic system.
It is a type of MPC, which is a well-established control algorithm that predicts the future behavior of a system based on a mathematical model and uses this information to generate optimal control actions.

In this Monte Carlo MPC, instead of using a single action prediction of the future system behavior, a large number of simulations are run, each with different random actions variations in the model parameters.
Based on these trials, Monte Carlo MPC generates control actions that minimize a defined cost function, taking into account the control objectives.
The control actions are updated at each time step based on new measurements of the system state.

Monte Carlo MPC is particularly useful in situations where the system is uncertain, unpredictable, or subject to significant external disturbances, as it allows for a probabilistic treatment of these uncertainties.
It has found applications in a variety of fields, including autonomous systems, robotics, and process control.
    
This MPC policy can be imported via:

.. code-block:: python

    import mlpro.rl.pool.actionplanner.mpc


Prerequisites
=============

    - `NumPy <https://pypi.org/project/numpy/>`_



Additional Information
======================

Multiprocessing has also been incorporated into MPC, which allows parallel computations.
Depending on the number of planning horizon, but we believe that this reduces the training time massively.
 
  
Change Log
==========

+--------------------+---------------------------------------------+
| Version            | Changes                                     |
+====================+=============================================+
| 1.1.1              | Initial version release                     |
+--------------------+---------------------------------------------+


Cross Reference
===============

- :ref:`Howto RL-MB-002: MBRL with MPC on Grid World Environment <Howto MB RL 002>`.

If you apply this environment in your research or work, please kindly cite the following related paper:

.. code-block:: bibtex

@INPROCEEDINGS{Williams2017,
  author={Williams, Grady and Wagener, Nolan and Goldfain, Brian and Drews, Paul and Rehg, James M. and Boots, Byron and Theodorou, Evangelos A.},
  booktitle={2017 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Information theoretic MPC for model-based reinforcement learning}, 
  year={2017},
  volume={},
  number={},
  pages={1714-1721},
  doi={10.1109/ICRA.2017.7989202}
  }

