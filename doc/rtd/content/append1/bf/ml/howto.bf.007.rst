.. _Howto BF 7:
`Howto 07 - (ML) Hyperparameter Tuning using Hyperopt <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2007%20-%20(ML)%20Hyperparameter%20Tuning%20using%20Hyperopt.py>`_
================
Ver. 1.0.3 (2022-02-25)

This module demonstrates how to utilize wrapper class for Hyperopt in RL context.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
  
    - `Hyperopt <https://pypi.org/project/hyperopt/>`_
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
    - `ROS <http://wiki.ros.org/noetic/Installation>`_
    

Results
`````````````````
.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  HyperParam Tuner Hyperopt: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  HyperParam Tuner Hyperopt: Hyperopt configuration is successful 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Reset 
    ...
    100%|██████████| 10/10 [00:24<00:00,  2.46s/trial, best loss: -25.432588752666504]


Example Code
`````````````````

.. literalinclude:: ../../../../../../examples/bf/Howto 07 - (ML) Hyperparameter Tuning using Hyperopt.py
    :language: python
