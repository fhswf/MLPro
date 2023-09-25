.. _Howto WP RL 005:
Howto RL-WP-005: Validation SB3 Wrapper (On-Policy) 
===================================================

Prerequisites
^^^^^^^^^^^^^

Please install the following packages to run this examples properly:

    - `Pytorch <https://pypi.org/project/torch/>`_
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
    - `Panda <https://pypi.org/project/panda/>`_



Executable code
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../../../../../src/mlpro/rl/examples/howto_rl_wp_005_validation_wrapped_sb3_on_policy.py
	:language: python



Results
^^^^^^^

The result plot shows that MLPro's wrapper for Stable Baselines 3 behaves neutrally.

.. image:: images/howto016.png



Cross Reference
^^^^^^^^^^^^^^^

    - :ref:`API Reference - RL Agent <target_api_rl_agents>`
    - :ref:`API Reference - RL Environments <target_api_rl_env>`
    - :ref:`API Reference - Wrapper SB3 <Wrapper SB3>`