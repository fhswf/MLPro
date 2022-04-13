.. _Howto RL 8:
`Howto 08 - (RL) Run own agents with petting zoo environment <https://github.com/fhswf/MLPro/blob/main/examples/rl/Howto%2008%20-%20(RL)%20Run%20own%20agents%20with%20petting%20zoo%20environment.py>`_
================
Ver. 1.1.7 (2022-20-25)

This module shows how to run an own policy inside the standard agent model with a Petting Zoo environment using 
the mlpro framework.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
    - `PettingZoo <https://pypi.org/project/PettingZoo/>`_
  ..
    - `NumPy <https://pypi.org/project/numpy/>`_
  ..
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
  ..
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
  ..
    - `Pytorch <https://pypi.org/project/torch/>`_
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

By running the example code, there should be a similar line printed in the terminal output.

.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Petting Zoo Env Env "connect_four_v3": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Petting Zoo Env Env "connect_four_v3": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Petting Zoo Env Env "connect_four_v3": Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Agent Connect4_Agents: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Agent Connect4_Agents: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy DiscRandPolicy: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy DiscRandPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent Agent_0: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent Agent_0: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy DiscRandPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent Agent_0: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy DiscRandPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Agent Connect4_Agents: Agent 0 Agent_0 added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy DiscRandPolicy: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy DiscRandPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent Agent_1: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent Agent_1: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy DiscRandPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent Agent_1: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy DiscRandPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Agent Connect4_Agents: Agent 1 Agent_1 added. 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  RL-Scenario Connect Four V3: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  RL-Scenario Connect Four V3: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  RL-Scenario Connect Four V3: Process time 0:00:00 : Scenario reset with seed 1 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Agent Connect4_Agents: Init vizualization for all agents... 
    ....
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Multi-Agent Connect4_Agents: Start vizualization for all agents... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  RL-Scenario Connect Four V3: Process time 0:00:12 End of processing 


Example Code
`````````````````

.. literalinclude:: ../../../../../examples/rl/Howto 08 - (RL) Run own agents with petting zoo environment.py
    :language: python

We use Connect Four V3 as default testing environment in this example.
However, you can also change the environment into Pistonball V5 by uncommenting Line 160-166 and commenting Line 168-174.

.. code-block:: python
    
    myscenario  = PBScenario(
            p_mode=Mode.C_MODE_SIM,
            p_ada=True,
            p_cycle_limit=100,
            p_visualize=visualize,
            p_logging=logging
    )

    # myscenario  = C4Scenario(
    #         p_mode=Mode.C_MODE_SIM,
    #         p_ada=True,
    #         p_cycle_limit=100,
    #         p_visualize=visualize,
    #         p_logging=logging
    # )
