.. _Howto RL 12:
`Howto 12 - (RL) Wrap mlpro Environment class to petting zoo environment <https://github.com/fhswf/MLPro/blob/main/examples/rl/Howto%2012%20-%20(RL)%20Wrap%20mlpro%20Environment%20class%20to%20petting%20zoo%20environment.py>`_
================
Ver. 1.0.3 (2021-12-03)

This module shows how to wrap mlpro's Environment class to petting zoo compatible.

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
    - `Optuna <https://pypi.org/project/optuna/>`_
  ..
    - `Hyperopt <https://pypi.org/project/hyperopt/>`_
  ..
    - `ROS <http://wiki.ros.org/noetic/Installation>`_
    

Results
`````````````````
The Bulk Good Laboratory Plant (BGLP) environment will be wrapped to a petting zoo compliant environment. 


.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Reset 
    Starting API test
    ...
    Passed API test
    test completed
    
There are several lines of action processing logs due to the API tests-
When there is no detected failure, the completed test message will be printed.




Example Code
`````````````````

.. literalinclude:: ../../../../../examples/rl/Howto 12 - (RL) Wrap mlpro Environment class to petting zoo environment.py
    :language: python

