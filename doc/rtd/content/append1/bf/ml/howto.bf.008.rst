.. _Howto BF 8:
`Howto 08 - (ML) Hyperparameter Tuning using Optuna <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2008%20-%20(ML)%20Hyperparameter%20Tuning%20using%20Optuna.py>`_
================
Ver. 1.0.0 (2022-03-24)

This module demonstrates how to utilize wrapper class for Optuna in RL context.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
  
    - `Optuna <https://pypi.org/project/optuna/>`_
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
    - `Hyperopt <https://pypi.org/project/hyperopt/>`_
  ..
    - `ROS <http://wiki.ros.org/noetic/Installation>`_
    

Results
`````````````````
.. code-block:: bash

    [I YYYY-MM-DD  HH:MM:SS.SSSSSS] A new study created in memory with name: no-name-398ac198-7424-450e-adbe-5624060a8e55
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  HyperParam Tuner Optuna: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  HyperParam Tuner Optuna: Optuna configuration is successful 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent BELT_CONVEYOR_A: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent BELT_CONVEYOR_A: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent BELT_CONVEYOR_A: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VACUUM_PUMP_B: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VACUUM_PUMP_B: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VACUUM_PUMP_B: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VIBRATORY_CONVEYOR_B: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VIBRATORY_CONVEYOR_B: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VIBRATORY_CONVEYOR_B: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VACUUM_PUMP_C: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VACUUM_PUMP_C: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent VACUUM_PUMP_C: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent ROTARY_FEEDER_C: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent ROTARY_FEEDER_C: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Agent ROTARY_FEEDER_C: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Policy MyPolicy: Adaptivity switched on 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  HyperParam Tuner Optuna: Trial number 0 has started
    ...
    [I YYYY-MM-DD  HH:MM:SS.SSSSSS] Trial 0 finished with value: 24.64927354148731 and parameters: {'num_states_0': 12, 'smoothing_1': 0.354494285942355, 'lr_rate_2': 0.04680120891085934, 'buffer_size_3': 65780, 'update_rate_4': 19, 'sampling_size_5': 82, 'num_states_6': 18, 'smoothing_7': 0.19081023337497865, 'lr_rate_8': 0.0919439019482214, 'buffer_size_9': 45391, 'update_rate_10': 12, 'sampling_size_11': 160, 'num_states_12': 55, 'smoothing_13': 0.395358303704589, 'lr_rate_14': 0.07028853082420614, 'buffer_size_15': 14435, 'update_rate_16': 16, 'sampling_size_17': 171, 'num_states_18': 85, 'smoothing_19': 0.1751593882011415, 'lr_rate_20': 0.0722610656948767, 'buffer_size_21': 13727, 'update_rate_22': 8, 'sampling_size_23': 108, 'num_states_24': 56, 'smoothing_25': 0.18578907083856855, 'lr_rate_26': 0.05625367463777988, 'buffer_size_27': 29712, 'update_rate_28': 6, 'sampling_size_29': 172}. Best is trial 0 with value: 24.64927354148731.
    ...

Example Code
`````````````````

.. literalinclude:: ../../../../../../examples/bf/Howto 07 - (ML) Hyperparameter Tuning using Hyperopt.py
    :language: python
