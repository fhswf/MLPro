.. _Howto GT 003:
Howto GT-DG-003: Train Multi-Player with State-based Potential Games on the BGLP
================================================================================

**Prerequisites**


Please install the following packages to run this examples properly:
    - `NumPy <https://pypi.org/project/numpy/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/gt/howto_gt_dg_003_train_multi_player_with_SbPG_on_BGLP.py
	:language: python



**Results**

.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": -- Training run 0 started...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------ 

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": -- Training episode 0 started...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------ 

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": Limit of 10000 cycles per episode reached (Training)
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": -- Training episode 0 finished after 10000 cycles
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": -- Training cycles finished: 10000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------ 


    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": -- Training episode 1 started...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------ 

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": Limit of 10000 cycles per episode reached (Training)
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": -- Training episode 1 finished after 10000 cycles
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": -- Training cycles finished: 20000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------ 


    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": -- Training episode 2 started...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------ 

    .....

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Training "GT": ------------------------------------------------------------------------------ 

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Results stored in : "xxx"
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Training Results of run 0
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Scenario          : Game SbPG_Scenario
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Model             : Multi-Player SbPG Players
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Start time stamp  : YYYY-MM-DD  HH:MM:SS.SSSSSS
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- End time stamp    : 2025-04-10 13:41:52.739906
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Duration          : 0:14:09.071832
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Start cycle id    : 0
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- End cycle id      : 99999
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Training cycles   : 100000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Evaluation cycles : 0
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Adaptations       : 100000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- High score        : None
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Training Episodes : 10
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": -- Evaluations       : 0
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": ------------------------------------------------------------------------------
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Results  "GT": ------------------------------------------------------------------------------ 



**Cross Reference**

+ :ref:`API Reference: Game Theory in Dynamic Games <target_api_gt>`