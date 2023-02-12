.. _Howto Env RL 003:
Howto RL-ENV-003: SB3 Policy on MultiGeo Environment
====================================================================================================

.. automodule:: mlpro.rl.examples.howto_rl_env_003_train_agent_with_sb3_policy_on_multigeo_environment



**Prerequisites**


Please install the following packages to run this examples properly:
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
    - :ref:`RL Environment Multi Geometry Robot <multigeorobot>`
    - `NumPy <https://pypi.org/project/numpy/>`_
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
    - `Pytorch <https://pypi.org/project/torch/>`_
    


**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/rl/examples/howto_rl_env_003_train_agent_with_sb3_policy_on_multigeo_environment.py
	:language: python



**Results**


.. image:: images/multigeosim.gif

The Gazebo GUI should be the first thing that shows up. 
The Multi Geometry robot will move depending on the given action and the training is run. 
When the training is done, the logged rewards will be plotted using the matplotlib library.

The plotted figure is is not reproducible due to the simulator's nature of simulating real
world scenario. Although seeds can be set for the random generator, the sampling cannot be 
done at the exact same time during different runs.


**Cross Reference**

+ API References: :ref:`RL Agent <target_api_rl_agents>`, :ref:`RL Environments <target_api_rl_env>`