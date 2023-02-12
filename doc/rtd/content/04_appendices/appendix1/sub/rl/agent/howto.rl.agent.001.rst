.. _Howto Agent RL 001:
Howto RL-AGENT-001: Run an Agent with Own Policy
======================================================================

.. automodule:: mlpro.rl.examples.howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment



**Prerequisites**


Please install the following packages to run this examples properly:
    - `OpenAI Gym <https://pypi.org/project/gym/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/rl/examples/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.py
	:language: python



**Results**

The OpenAI Gym cartpole window appears and shows a randowm behavior while the demo run is logged on the console. Demo should terminate after 61 cycles because of the fixed random seed.

.. image:: images/Cartpole.png

**Cross Reference**

+ API References: :ref:`RL Agent <target_api_rl_agents>`, :ref:`RL Environments <target_api_rl_env>`