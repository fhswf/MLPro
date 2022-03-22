.. _Howto RL 16:
`Howto 16 - (RL) Model Based Reinforcement Learning <https://https://github.com/fhswf/MLPro/blob/main/examples/rl/Howto%2016%20-%20(RL)%20Model%20Based%20Reinforcement%20Learning.py>`_
================
Ver. 1.0.1 (2022-01-01)

This module demonstrates model-based reinforcement learning.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
    - `Pytorch <https://pypi.org/project/torch/>`_
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
  ..
    - `NumPy <https://pypi.org/project/numpy/>`_
  ..
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
  ..
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
  ..
    - `PettingZoo <https://pypi.org/project/PettingZoo/>`_
  ..
    - `Optuna <https://pypi.org/project/optuna/>`_
  ..
    - `Hyperopt <https://pypi.org/project/hyperopt/>`_
  ..
    - `ROS <http://wiki.ros.org/noetic/Installation>`_
    

Results
`````````````````

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
    
Example Code
`````````````````

.. literalinclude:: ../../../../../examples/rl/Howto 16 - (RL) Model Based Reinforcement Learning.py
    :language: python

