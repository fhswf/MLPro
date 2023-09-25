.. _Howto WP RL 002:
Howto RL-WP-002: MLPro to PettingZoo
====================================

**Prerequisites**

Please install the following packages to run this examples properly:

    - `PettingZoo <https://pypi.org/project/PettingZoo/>`_

    

**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/rl/howto_rl_wp_002_mlpro_environment_to_petting_zoo_environment.py
	:language: python



**Results**

The Bulk Good Laboratory Plant (BGLP) environment will be wrapped to a PettingZoo compliant environment. 


.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Operation mode set to 0 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Environment BGLP: Reset 
    Starting API test
    ...
    Passed API test
    test completed
    
There are several lines of action processing logs due to the API tests. When there is no detected failure, the environment is successfully wrapped.



**Cross Reference**

    - :ref:`API Reference - RL Agent <target_api_rl_agents>`
    - :ref:`API Reference - RL Environments <target_api_rl_env>`
    - :ref:`API Reference - Wrapper PettingZoo <Wrapper PettingZoo>`
