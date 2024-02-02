.. _Howto GTN 005:
Howto GT-Native-005: 3P Routing Problems
===========================================================================

**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/gt/howto_gt_native_005_routing_problems_3p.py
	:language: python
  
**Results**

.. code-block:: bash
 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Training "Native GT Training": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Game "Routing_3P": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Solver "MinGreedyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player 1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player 1": Player 1 is keeping the same solver 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Coalition "Coalition of Player 1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Coalition "Coalition of Player 1": Player 1 added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Solver "MinGreedyPolicy": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Solver "RandomSolver": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player 2": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player 2": Player 1 is switching to solver MinGreedyPolicy 1 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Coalition "Coalition of Player 2": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Coalition "Coalition of Player 2": Player 2 added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Solver "RandomSolver": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player 3": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Player "Player 3": Player 3 is keeping the same solver 2 
    ... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  GT Training "Native GT Training": Training completed 



**Cross Reference**

+ :ref:`API Reference: Native Game Theory <target_api_gt_native>`