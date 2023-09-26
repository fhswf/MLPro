.. _Howto Agent RL 004:
Howto RL-AGENT-004: Train Multi-Agent with Own Policy
======================================================================================

**Prerequisites**

Please install the following packages to run this examples properly:

    - `OpenAI Gym <https://pypi.org/project/gym/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/rl/howto_rl_agent_004_train_multiagent_with_own_policy_on_multicartpole_environment.py
	:language: python



**Results**

Similar output as in :ref:`Howto RL-AGENT-002 <Howto Agent RL 002>` is displayed. However, three cartpole windows will be opened while the training
log runs through. In addition, the training result folder will contain one more pkl file for the second agent.



**Cross Reference**

    - :ref:`API Reference - RL Agent <target_api_rl_agents>`
    - :ref:`API Reference - RL Environments <target_api_rl_env>`
    - :ref:`API Reference - RL Scenario and Training <target_api_rl_run_train>`