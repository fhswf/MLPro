.. _Howto RL 021:
Howto RL-021: Train a wrapped Stable Baselines 3 policy on MLPro's native DoublePendulum environment
====================================================================================================

.. automodule:: mlpro.rl.examples.howto_rl_021_train_wrapped_sb3_policy_on_doublependulum



Prerequisites
-------------

Please install the following packages to run this examples properly:
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
    - `PyTorch <https://pypi.org/project/torch/>`_
    - `NumPy <https://pypi.org/project/numpy/>`_
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_



Executable code
---------------
.. literalinclude:: ../../../../../src/mlpro/rl/examples/howto_rl_021_train_wrapped_sb3_policy_on_doublependulum.py
	:language: python



Results
-------

.. image:: images/DoublePendulum.png

The Double Pendulum environment window appears. Afterwards, the training should run 
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
    
After termination the local result folder contains the training result files:
    - agent_actions.csv
    - env_rewards.csv
    - env_states.csv
    - evaluation.csv
    - summary.csv
    - trained model.pkl