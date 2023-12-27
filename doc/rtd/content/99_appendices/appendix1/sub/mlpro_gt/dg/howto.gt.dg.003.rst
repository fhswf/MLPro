.. _Howto GT 003:
Howto GT-003: Train Multi-Player in Potential Games
===========================================================================

**Prerequisites**


Please install the following packages to run this examples properly:
    - `NumPy <https://pypi.org/project/numpy/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/gt/howto_gt_dg_003_train_own_multi_player_in_potential_games.py
	:language: python



**Results**

.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training run 0 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training episode 0 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  E  GT Game "Matrix": Process time 0:01:40 : Environment terminated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": Limit of 100 cycles per episode reached (Environment) 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training episode 0 finished after 100 cycles 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training cycles finished: 100 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 

    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training episode 1 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  E  GT Game "Matrix": Process time 0:01:40 : Environment terminated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": Limit of 100 cycles per episode reached (Environment) 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training episode 1 finished after 100 cycles 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training cycles finished: 200 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 

    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": Training cycle limit 200 reached 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training run 0 finished 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Results stored in : "C:\Users\%username%\YYYY-MM-DD  HH:MM:SS Training GT" 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Training Results of run 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Scenario          : GT Game Matrix 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Model             : GT Multi-Player BGLP Players with Random Policies 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Start time stamp  : YYYY-MM-DD  HH:MM:SS.SSSSSS  
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- End time stamp    : YYYY-MM-DD  HH:MM:SS.SSSSSS 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Duration          : 0:00:01.664187 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Start cycle id    : 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- End cycle id      : 199 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Training cycles   : 200 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Evaluation cycles : 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Adaptations       : 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- High score        : None 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Training Episodes : 2 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Evaluations       : 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": ------------------------------------------------------------------------------ 



**Cross Reference**

+ :ref:`API Reference: Game Theory in Dynamic Games <target_api_gt>`