.. _Howto RL 11:
`Howto 11 - (RL) Wrap mlpro Environment class to gym environment <https://github.com/fhswf/MLPro/blob/main/examples/rl/Howto%2011%20-%20(RL)%20Wrap%20mlpro%20Environment%20class%20to%20gym%20environment.py>`_
================
Ver. 1.0.1 (2021-10-04)

This module shows how to wrap mlpro's Environment class to gym compatible.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
    - `OpenAI Gym <https://pypi.org/project/gym/0.19.0/>`_
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
  ..
    - `NumPy <https://pypi.org/project/numpy/>`_
  ..
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
  ..
    - `Pytorch <https://pypi.org/project/torch/>`_
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
The GridWorld environment will be wrapped to a gym environment. By making use of gym's environment
checker, we could confirm the success of the environment wrapping.


.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Start processing action 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Actions of agent 0 = [3.415721893310547, -7.9934492111206055] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Action processing finished successfully 
    ...
    
There will be several more lines of action processing logs due to the nature of the environment checker.
When there is no detected failure, the environment is successfully wrapped.

Example Code
`````````````````

.. literalinclude:: ../../../../../examples/rl/Howto 11 - (RL) Wrap mlpro Environment class to gym environment.py
    :language: python


