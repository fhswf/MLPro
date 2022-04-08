.. _Howto RL 22:
`Howto 22 - (RL) Train DoublePendulum with SB3 Wrapper <https://github.com/fhswf/MLPro/blob/main/examples/rl/Howto%2022%20-%20(RL)%20Train%20DoublePendulum%20with%20SB3%20wrapper.py>`_
================
Ver. 1.0.2 (2022-02-27)

This module shows how to use SB3 wrapper to train double pendulum

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
    - `PyTorch <https://pypi.org/project/torch/>`_
    - `NumPy <https://pypi.org/project/numpy/>`_
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
  ..
    - `NumPy <https://pypi.org/project/numpy/>`_
  ..
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
  ..
    - `PyTorch <https://pypi.org/project/torch/>`_
  ..
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
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
.. image:: images/DoublePendulum.png

The Double Pendulum environment window should appear. Afterwards, the training should run 
for a few episodes before terminating and printing the result. The training log
is also stored in the location specified. 

.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment DoublePendulum: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment DoublePendulum: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment DoublePendulum: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment DoublePendulum: Reset 
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
    
In the folder, there should be some files including:
    - agent_actions.csv
    - env_rewards.csv
    - env_states.csv
    - evaluation.csv
    - summary.csv
    - trained model.pkl

Example Code
`````````````````

.. literalinclude:: ../../../../../examples/rl/Howto 22 - (RL) Train DoublePendulum with SB3 Wrapper.py
    :language: python
