.. _Howto RL 002:
Howto RL-002: Run an agent with an own policy with an OpenAI GYM environment
======================================================================

.. automodule:: mlpro.rl.examples.howto_rl_002_run_agent_with_own_policy_with_gym_environment



Prerequisites
-------------

Please install the following packages to run this examples properly:
    - `OpenAI Gym <https://pypi.org/project/gym/>`_



Executable code
---------------
.. literalinclude:: ../../../../../src/mlpro/rl/examples/howto_rl_002_run_agent_with_own_policy_with_gym_environment.py
	:language: python



Results
-------
The OpenAI Gym cartpole window appears and shows a randowm behavior while the demo run is logged on the console. Demo should terminate after 61 cycles because of the fixed random seed.

.. image:: images/Cartpole.png