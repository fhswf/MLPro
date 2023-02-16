.. _Howto WP RL 001:
Howto RL-WP-001: MLPro to OpenAI Gym
===========================================================================

.. automodule:: mlpro.rl.examples.howto_rl_wp_001_mlpro_environment_to_gym_environment



**Prerequisites**


Please install the following packages to run this examples properly:
    - `OpenAI Gym <https://pypi.org/project/gym/0.19.0/>`_
    


**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/rl/examples/howto_rl_wp_001_mlpro_environment_to_gym_environment.py
	:language: python



**Results**


The native MLPro GridWorld environment will be wrapped to a OpenAI Gym environment. By making use of Gym's environment
checker, we could confirm the success of the environment wrapping.


.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Reset 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Start processing action 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Actions of agent 0 = [3.415721893310547, -7.9934492111206055] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment GridWorld: Action processing finished successfully 
    ...
    
There will be several more lines of action processing logs due to the nature of the environment checker.
When there is no detected failure, the environment is successfully wrapped.

**Cross Reference**

+ API References: :ref:`RL Agent <target_api_rl_agents>`, :ref:`RL Environments <target_api_rl_env>`, :ref:`Wrapper OpenAI Gym <Wrapper OpenAI Gym>