.. _Howto MB RL 002:
Howto RL-MB-002: MBRL with MPC on Grid World Environment
====================================================================================================

.. automodule:: mlpro.rl.examples.howto_rl_mb_002_grid_world_environment



**Prerequisites**

Please install the following packages to run this examples properly:
    - `PyTorch <https://pypi.org/project/torch/>`_
    - `NumPy <https://pypi.org/project/numpy/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/rl/examples/howto_rl_mb_002_grid_world_environment.py
	:language: python



**Results**


After the environment is initiated, the training will run for the specified amount of limits. The expected initial console output can be seen below.

.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment Grid World: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment Grid World: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment Grid World: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment Grid World: Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: -- Training run 0 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: ------------------------------------------------------------------------------ 
     
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: -- Evaluation period 0 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: ------------------------------------------------------------------------------ 
     
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: -- Evaluation episode 0 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training RL: ------------------------------------------------------------------------------ 
    ...
    
After termination the local result folder contains the training result files:
    - agent_actions.csv
    - env_rewards.csv
    - env_states.csv
    - evaluation.csv
    - summary.csv
    - trained model.pkl
    
.. image:: images/howto23.png

The image above shows that the agent in most cases can reach the goal.


**Cross Reference**

    + :ref:`API Reference - RL Agent <target_api_rl_agents>`
    + :ref:`API Reference - RL Environments <target_api_rl_env>`
    + :ref:`API Reference - Environment Model <target_api_rl_env_ada>`
    + :ref:`API Reference - RL Scenario and Training <target_api_rl_run_train>`