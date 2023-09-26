.. _Howto MB RL 003:
Howto RL-MB-003: MBRL on RobotHTM Environment
=============================================

**Prerequisites**

Please install the following packages to run this examples properly:

    - `Pytorch <https://pypi.org/project/torch/>`_
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/rl/howto_rl_mb_003_robothtm_environment.py
	:language: python



**Results**

After the environment is initiated, the training will run for the specified amount of limits.
The expected initial console output can be seen below.


.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Training Actual: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment RobotHTM: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment RobotHTM: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment RobotHTM: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment RobotHTM: Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  SB3 Policy ????: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  SB3 Policy ????: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent Smith1: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent Smith1: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  SB3 Policy ????: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  RL-Scenario Matrix1: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  RL-Scenario Matrix1: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Training Actual: Training started (without hyperparameter tuning) 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Results  RL: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training Actual: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training Actual: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training Actual: -- Training run 0 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training Actual: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training Actual: ------------------------------------------------------------------------------ 
     
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  RL-Scenario Matrix1: Process time 0:00:00 : Scenario reset with seed 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment RobotHTM: Reset 
    ...



**Cross Reference**

    - :ref:`API Reference - RL Agent <target_api_rl_agents>`
    - :ref:`API Reference - RL Environments <target_api_rl_env>`
    - :ref:`API Reference - Environment Model <target_api_rl_env_ada>`
    - :ref:`API Reference - RL Scenario and Training <target_api_rl_run_train>`