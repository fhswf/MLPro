.. _Howto RL 009:
Howto RL-009: Wrap native MLPro environment class to PettingZoo environment
===========================================================================

.. automodule:: mlpro.rl.examples.howto_rl_009_wrap_mlpro_environment_to_pettingzoo_environment



Prerequisites
-------------

Please install the following packages to run this examples properly:
    - `PettingZoo <https://pypi.org/project/PettingZoo/>`_

    

Executable code
---------------
.. literalinclude:: ../../../../../src/mlpro/rl/examples/howto_rl_009_wrap_mlpro_environment_to_pettingzoo_environment.py
	:language: python



Results
-------

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