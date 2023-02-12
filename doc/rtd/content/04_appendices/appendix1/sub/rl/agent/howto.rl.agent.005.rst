.. _Howto Agent RL 005:
Howto RL-AGENT-005: Train and Reload Single Agent
==========================================================

.. automodule:: mlpro.rl.examples.howto_rl_agent_005_train_and_reload_single_agent



Prerequisites
-------------

Please install the following packages to run this examples properly:
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_



Executable code
---------------
.. literalinclude:: ../../../../../../../../src/mlpro/rl/examples/howto_rl_agent_005_train_and_reload_single_agent.py
	:language: python



Results
-------

The Gym Cartpole environment window appears. Afterwards, the training runs 
for a few episodes before terminating and printing the result. 

.. image:: images/Cartpole.png

.. code-block:: bash

    ...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Training Results of run 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Scenario          : RL-Scenario Matrix 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Model             : Agent Smith 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Start time stamp  : YYYY-MM-DD HH:MM:SS.SSSSSS  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- End time stamp    : YYYY-MM-DD HH:MM:SS.SSSSSS  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Duration          : HH:MM:SS.SSSSSS 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Start cycle id    : 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- End cycle id      : 249 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Training cycles   : 250 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Evaluation cycles : 189 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Adaptations       : 50 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- High score        : 9.6
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Results stored in : "C:\Users\%username%\YYYY-MM-DD  HH:MM:SS Training RL" 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Training Episodes : 15 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Evaluations       : 4 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    ...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Training Results of run 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Scenario          : RL-Scenario Matrix2
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Model             : Agent Smith 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Start time stamp  : YYYY-MM-DD HH:MM:SS.SSSSSS  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- End time stamp    : YYYY-MM-DD HH:MM:SS.SSSSSS  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Duration          : HH:MM:SS.SSSSSS 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Start cycle id    : 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- End cycle id      : 249 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Training cycles   : 250 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Evaluation cycles : 143 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Adaptations       : 50 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- High score        : 9.6
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Results stored in : "C:\Users\%username%\YYYY-MM-DD  HH:MM:SS Training RL" 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Training Episodes : 11 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Evaluations       : 3
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    
    
After termination the local result folders contain the training result files:
    - agent_actions.csv
    - env_rewards.csv
    - env_states.csv
    - evaluation.csv
    - summary.csv
    - trained model.pkl

Both training results are from the same agent.