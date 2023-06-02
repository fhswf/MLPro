.. _Howto GT 004:
Howto GT-004: Train Multi-Player in Stackelberg Games
===========================================================================

.. automodule:: mlpro.gt.examples.howto_gt_dg_004_train_own_multi_player_in_stackelberg_games


**Prerequisites**


Please install the following packages to run this examples properly:
    - `NumPy <https://pypi.org/project/numpy/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/gt/examples/howto_gt_dg_004_train_own_multi_player_in_stackelberg_games.py
	:language: python



**Results**

.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Training "GT Training": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Game "Matrix": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Multi-Player SG "BGLP Players with Random Policies": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 478c892f-f980-4f5a-b894-f6d57dd357f7": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "BELT_CONVEYOR_A (Leader)": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "BELT_CONVEYOR_A (Leader) 478c892f-f980-4f5a-b894-f6d57dd357f7": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 478c892f-f980-4f5a-b894-f6d57dd357f7": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "BELT_CONVEYOR_A (Leader) 478c892f-f980-4f5a-b894-f6d57dd357f7": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 478c892f-f980-4f5a-b894-f6d57dd357f7": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": GT Player SG BELT_CONVEYOR_A (Leader) 478c892f-f980-4f5a-b894-f6d57dd357f7 added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VACUUM_PUMP_B (Follower)": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VACUUM_PUMP_B (Follower) 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VACUUM_PUMP_B (Follower) 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": GT Player SG VACUUM_PUMP_B (Follower) 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VIBRATORY_CONVEYOR_B (Follower)": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VIBRATORY_CONVEYOR_B (Follower) 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VIBRATORY_CONVEYOR_B (Follower) 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": GT Player SG VIBRATORY_CONVEYOR_B (Follower) 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 2aef81a6-135c-4ce5-9b47-01576e635930": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VACUUM_PUMP_C (Follower)": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VACUUM_PUMP_C (Follower) 2aef81a6-135c-4ce5-9b47-01576e635930": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 2aef81a6-135c-4ce5-9b47-01576e635930": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VACUUM_PUMP_C (Follower) 2aef81a6-135c-4ce5-9b47-01576e635930": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 2aef81a6-135c-4ce5-9b47-01576e635930": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": GT Player SG VACUUM_PUMP_C (Follower) 2aef81a6-135c-4ce5-9b47-01576e635930 added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 88354d45-5265-4ed1-b51a-acd9f9f77f15": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "ROTARY_FEEDER_C (Leader)": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "ROTARY_FEEDER_C (Leader) 88354d45-5265-4ed1-b51a-acd9f9f77f15": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 88354d45-5265-4ed1-b51a-acd9f9f77f15": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "ROTARY_FEEDER_C (Leader) 88354d45-5265-4ed1-b51a-acd9f9f77f15": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 88354d45-5265-4ed1-b51a-acd9f9f77f15": Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": GT Player SG ROTARY_FEEDER_C (Leader) 88354d45-5265-4ed1-b51a-acd9f9f77f15 added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Training "GT Training": Training started (without hyperparameter tuning) 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Results  "RL": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training run 0 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Game "Matrix": Process time 0:00:00 : Scenario reset with seed 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": -- Training episode 0 started... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT Training": ------------------------------------------------------------------------------ 
    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Game "Matrix": Process time 0:00:00 : Start of cycle 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Game "Matrix": Process time 0:00:00 : Agent computes action... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": Start of action computation for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "BELT_CONVEYOR_A (Leader) 478c892f-f980-4f5a-b894-f6d57dd357f7": Action computation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "BELT_CONVEYOR_A (Leader) 478c892f-f980-4f5a-b894-f6d57dd357f7": Action computation finished 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "ROTARY_FEEDER_C (Leader) 88354d45-5265-4ed1-b51a-acd9f9f77f15": Action computation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "ROTARY_FEEDER_C (Leader) 88354d45-5265-4ed1-b51a-acd9f9f77f15": Action computation finished 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VACUUM_PUMP_B (Follower) 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Action computation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VACUUM_PUMP_B (Follower) 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Action computation finished 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VIBRATORY_CONVEYOR_B (Follower) 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Action computation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VIBRATORY_CONVEYOR_B (Follower) 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Action computation finished 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VACUUM_PUMP_C (Follower) 2aef81a6-135c-4ce5-9b47-01576e635930": Action computation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player SG "VACUUM_PUMP_C (Follower) 2aef81a6-135c-4ce5-9b47-01576e635930": Action computation finished 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": End of action computation for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Game "Matrix": Process time 0:00:00 : Env processes action... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Start processing action 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Actions of agent 478c892f-f980-4f5a-b894-f6d57dd357f7 = [0.84442185] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Actions of agent 88354d45-5265-4ed1-b51a-acd9f9f77f15 = [0.7579544] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Actions of agent 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e = [0.42057158] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Actions of agent 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef = [0.25891675] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Actions of agent 2aef81a6-135c-4ce5-9b47-01576e635930 = [0.51127472] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Assessment for success... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Assessment for breakdown... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Board "BGLP_GT": Action processing finished successfully 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Game "Matrix": Process time 0:00:01 : Agent adapts policy... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Multi-Player SG "BGLP Players with Random Policies": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": Start of adaptation for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": Start adaption for agent 478c892f-f980-4f5a-b894-f6d57dd357f7 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "BELT_CONVEYOR_A (Leader) 478c892f-f980-4f5a-b894-f6d57dd357f7": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 478c892f-f980-4f5a-b894-f6d57dd357f7": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy 478c892f-f980-4f5a-b894-f6d57dd357f7": Sorry, I am a stupid agent... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": Start adaption for agent 88354d45-5265-4ed1-b51a-acd9f9f77f15 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "ROTARY_FEEDER_C (Leader) 88354d45-5265-4ed1-b51a-acd9f9f77f15": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 88354d45-5265-4ed1-b51a-acd9f9f77f15": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy 88354d45-5265-4ed1-b51a-acd9f9f77f15": Sorry, I am a stupid agent... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": Start adaption for agent 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VACUUM_PUMP_B (Follower) 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy 61ceca2b-52fc-4f34-8fbb-36bff56d9e1e": Sorry, I am a stupid agent... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": Start adaption for agent 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VIBRATORY_CONVEYOR_B (Follower) 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy 6b4f1ea8-9b24-48b1-b4bc-f58662dc9cef": Sorry, I am a stupid agent... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": Start adaption for agent 2aef81a6-135c-4ce5-9b47-01576e635930 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Player SG "VACUUM_PUMP_C (Follower) 2aef81a6-135c-4ce5-9b47-01576e635930": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  Policy "MyPolicy 2aef81a6-135c-4ce5-9b47-01576e635930": Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy "MyPolicy 2aef81a6-135c-4ce5-9b47-01576e635930": Sorry, I am a stupid agent... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Multi-Player SG "BGLP Players with Random Policies": End of adaptation for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  S  GT Game "Matrix": Process time 0:00:01 : End of cycle 0 

    ....

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Training Results of run 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": ------------------------------------------------------------------------------ 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Scenario          : GT Game Matrix 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Model             : GT Multi-Player SG BGLP Players with Random Policies 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Start time stamp  : YYYY-MM-DD  HH:MM:SS.SSSSSS 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- End time stamp    : YYYY-MM-DD  HH:MM:SS.SSSSSS 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "RL": -- Duration          : 0:00:02.156790 
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
    
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Training "GT Training": Training completed 



**Cross Reference**

+ :ref:`API Reference: Game Theory in Dynamic Games <target_api_gt>`