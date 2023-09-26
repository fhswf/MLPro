.. _Howto Agent RL 001:
Howto RL-AGENT-001: Run an Agent with Own Policy
======================================================================

**Prerequisites**

Please install the following packages to run this examples properly:

    - `OpenAI Gym <https://pypi.org/project/gym/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/rl/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.py
	:language: python



**Results**

The OpenAI Gym cartpole window appears and shows a random behavior while the demo run is logged on the console. Demo should terminate after 61 cycles because of the fixed random seed.

.. image:: images/Cartpole.png



**Cross Reference**

    - :ref:`API Reference - RL Agent <target_api_rl_agents>`
    - :ref:`API Reference - RL Environments <target_api_rl_env>`
    - :ref:`API Reference - RL Scenario and Training <target_api_rl_run_train>`