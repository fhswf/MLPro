.. _Howto Env RL 001:
Howto RL-ENV-001: SB3 Policy on UR5 Environment
========================================================================================

.. automodule:: mlpro.rl.examples.howto_rl_env_001_train_agent_with_sb3_policy_on_ur5_environment



**Prerequisites**


Please install the following packages to run this examples properly:
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
    - :ref:`RL Environment UR5 Joint Control <ur5jointcontrol>`
    - `NumPy <https://pypi.org/project/numpy/>`_
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
    - `Pytorch <https://pypi.org/project/torch/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/rl/examples/howto_rl_env_001_train_agent_with_sb3_policy_on_ur5_environment.py
	:language: python



**Results**


.. image:: images/ur5simulation.gif

The Gazebo GUI should be the first thing that shows up. 
The UR5 robot will move depending on the given action and the training is run. 
When the training is done, the logged rewards will be plotted using the matplotlib library.

The plotted figure is not reproducible due to the simulator's nature of simulating real
world scenario. Although seeds can be set for the random generator, the sampling cannot be 
done at the exact same time during different runs. For a more reproducible results, 
:ref:`Howto RL-ENV-002 <Howto Env RL 002>` is more appropriate.


**Cross Reference**


+ API References: :ref:`RL Agent <target_api_rl_agents>`, :ref:`RL Environments <target_api_rl_env>`, :ref:`RL Scenario and Training` <target_api_rl_run_train>
