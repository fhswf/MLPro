.. _Howto Env RL 003:
Howto RL-ENV-003: Run Agent with random Policy in Double Pendulum MuJoCo Environment
====================================================================================

**Prerequisites**

Please install the following packages to run this examples properly:

    - `NumPy <https://pypi.org/project/numpy/>`_
    - `MuJoCo <https://pypi.org/project/mujoco/>`_


**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/rl/howto_rl_env_003_run_agent_with_random_policy_on_double_pendulum_mujoco_environment.py
	:language: python


**Results**

Running this howto opens a new MuJoCo window that shows the double pendulum environment
controlled by a random policy. This, in turn, causes chaotic behavior.

.. image:: images/howto_rl_env_003.gif
    :scale: 50 %


**Cross Reference**

    - :ref:`API Reference - RL Agent <target_api_rl_agents>`
    - :ref:`API Reference - RL Environments <target_api_rl_env>`
    - :ref:`API Reference - RL Scenario and Training <target_api_rl_run_train>`