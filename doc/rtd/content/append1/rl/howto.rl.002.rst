.. _Howto RL 2:
`Howto 02 - (RL) Run agent with own policy with gym environment <https://github.com/fhswf/MLPro/blob/main/examples/rl/Howto%2002%20-%20(RL)%20Run%20agent%20with%20own%20policy%20with%20gym%20environment.py>`_
================
Ver. 1.2.2 (2021-12-03)

This module shows how to run an own policy inside the standard agent model with an OpenAI Gym environment using 
the fhswf_at_ml framework.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
  ..
    - `NumPy <https://pypi.org/project/numpy/>`_
  ..
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
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
    

Example Code
`````````````````

.. literalinclude:: ../../../../../examples/rl/Howto 02 - (RL) Run agent with own policy with gym environment.py
    :language: python

Results
`````````````````
The Gym cartpole window appears and shows a randowm behavior while the demo run is logged on the console. Deno should terminate after 61 cycles because of the fixed random seed.

