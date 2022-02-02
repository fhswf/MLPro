.. _Howto RL 1:
`Howto 01 - (RL) Types of reward <https://github.com/fhswf/MLPro/blob/main/examples/rl/Howto%2001%20-%20(RL)%20Types%20of%20reward.py>`_
================
Ver. 1.0.0 (2021-09-11)

This module shows how to create and interprete reward objects in own projects.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
  ..
    - `NumPy <https://pypi.org/project/numpy/>`_
  ..
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
  ..
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
  ..
    - `PyTorch <https://pypi.org/project/torch/>`_
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

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Example for reward type C_TYPE_OVERALL: 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward is just a scalar... 4.77 
     
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Example for reward type C_TYPE_EVERY_AGENT 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward is a list with entries for each agent... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward for agent 1 added: 4.77 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward for agent 2 added: 5.19 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward for agent 3 added: 0.23 
     
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Example for reward type C_TYPE_EVERY_ACTION 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward is a list with entries for each agent and its action components... 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward for agent 1, action 1 added: 1.23 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward for agent 1, action 2 added: 0.47 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward for agent 1, action 3 added: 1.63 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Reward Demo : Reward for agent 2, action 4 added: 4.23 


Example Code
`````````````````

.. literalinclude:: ../../../../../examples/rl/Howto 01 - (RL) Types of reward.py
    :language: python



