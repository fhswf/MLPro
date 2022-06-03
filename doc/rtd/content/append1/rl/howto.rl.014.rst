.. _Howto RL 014:
Howto RL-014: Advanced training with stagnation detection
=========================================================

.. automodule:: mlpro.rl.examples.howto_rl_014_advanced_training_with_stagnation_detection



Prerequisites
-------------

Please install the following packages to run this examples properly:
    - `Pytorch <https://pypi.org/project/torch/>`_



Executable code
---------------
.. literalinclude:: ../../../../../src/mlpro/rl/examples/howto_rl_014_advanced_training_with_stagnation_detection.py
	:language: python



Results
-------

After the multiple environments are initialised, the training will run for the specified amount of limits.
When stagnation is detected, the training will be stopped. 


.. code-block:: bash

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
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- End cycle id      :  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Training cycles   :  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Evaluation cycles :  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Adaptations       :  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- High score        :  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Results stored in : "C:\Users\%username%\YYYY-MM-DD  HH:MM:SS Training RL" 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Training Episodes : 120 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: -- Evaluations       : 25 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  RL: ------------------------------------------------------------------------------ 
    
After termination the local result folder contains the training result files:
    - agent_actions.csv
    - env_rewards.csv
    - env_states.csv
    - evaluation.csv
    - summary.csv
    - Agent 0 Smith-1(0).pkl
    - Agent 1 Smith-2(1).pkl
