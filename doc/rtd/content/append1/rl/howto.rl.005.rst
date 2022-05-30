.. _Howto RL 005:
Howto RL-005: Train a multi-agent with an own policy on the Multi-Cartpole environment
======================================================================================

.. automodule:: mlpro.rl.examples.howto_rl_005_train_multi_agent_with_own_policy_on_multicartpole_nvironment



Prerequisites
-------------

Please install the following packages to run this examples properly:
    - `OpenAI Gym <https://pypi.org/project/gym/>`_



Executable code
---------------
.. literalinclude:: ../../../../../src/mlpro/rl/examples/howto_rl_005_train_multi_agent_with_own_policy_on_multicartpole_nvironment.py
	:language: python



Results
-------
Similar output as in :ref:`Howto RL-003 <Howto RL 003>` is displayed. However, three cartpole windows will be opened while the training
log runs through. In addition, the training result folder will contain one more pkl file for the second agent.