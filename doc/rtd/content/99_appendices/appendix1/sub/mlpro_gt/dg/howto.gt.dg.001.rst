.. _Howto GT 001:
Howto GT-001: Run Multi-Player with Own Policy
===========================================================================

**Prerequisites**

Please install the following packages to run this examples properly:
    - `NumPy <https://pypi.org/project/numpy/>`_
    - `OpenAI Gym <https://pypi.org/project/gym/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/gt/howto_gt_dg_001_run_multi_player_with_own_policy_on_multicartpole_game_board.py
	:language: python


  
**Results**

.. image:: images/howto.gt.001/Cartpole.png

Three Gym Cartpole game board windows should appear and the following output should be expected in the console.

.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Potential Game Board MultiCartPole(PGT): Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Potential Game Board MultiCartPole(PGT): Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Potential Game Board MultiCartPole(PGT): Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1": Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Potential Game Board MultiCartPole(PGT): Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1" (0): Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1" (1): Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  OpenAI Gym Env Env "CartPole-v1" (2): Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player Neo: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player Neo: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player Neo: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Player 0 Neo added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player Trinity: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player Trinity: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player Trinity: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Player 1 Trinity added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Process time 0:00:00 Start of processing 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Process time 0:00:00 : Start of cycle 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Process time 0:00:00 : Agent computes action... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Start of action computation for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player 0 Neo: Action computation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player 0 Neo: Action computation finished 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player 1 Trinity: Action computation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player 1 Trinity: Action computation finished 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: End of action computation for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Process time 0:00:00 : Env processes action... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Potential Game Board MultiCartPole(PGT): Start processing action 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Potential Game Board MultiCartPole(PGT): Actions of agent 0 = [0.02821633] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Potential Game Board MultiCartPole(PGT): Actions of agent 1 = [0.5796828  0.73351315] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Potential Game Board MultiCartPole(PGT): Action processing finished successfully 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Process time 0:00:01 : Agent adapts policy... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Start of adaptation for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Start adaption for agent 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player 0 Neo: Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Policy MyPolicy: Sorry, I am a stupid agent... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Start adaption for agent 1 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Player 1 Trinity: Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptation started 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Policy MyPolicy: Sorry, I am a stupid agent... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: End of adaptation for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Process time 0:00:01 : End of cycle 0 
    ... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Player Human Beings: Start vizualization for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Game Matrix: Process time 0:00:12 End of processing 



**Cross Reference**

+ :ref:`API Reference: Game Theory in Dynamic Games <target_api_gt>`