.. _Howto GTN 002:
Howto GT-Native-002: 3P Prisoners' Dilemma
===========================================================================


**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/gt/howto_gt_native_002_prisoners_dilemma_3p.py
	:language: python


  
**Results**

.. code-block:: bash
 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Training "Native GT Training": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Game "PrisonersDilemma3PGame": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Solver "RandomSolver": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Solver "MinGreedyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player of Prisoner 1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player of Prisoner 1": Player 1 is switching to solver MinGreedyPolicy 1 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Coalition "Coalition of Prisoner 1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Coalition "Coalition of Prisoner 1": Player of Prisoner 1 added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Solver "RandomSolver": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Solver "MinGreedyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player of Prisoner 2": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player of Prisoner 2": Player 2 is switching to solver RandomSolver 2 
    ... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Training "Native GT Training": Training completed 



**Cross Reference**

+ :ref:`API Reference: Native Game Theory <target_api_gt_native>`