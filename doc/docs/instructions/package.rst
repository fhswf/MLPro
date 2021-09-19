Third Party Packages
========
`MLPro <https://github.com/fhswf/MLPro.git>`_ allows you to reuse widely-used packages and
integrate them to MLPro interface by calling wrapper classes.

At the moment, a wrapper class for OpenAI Gym Environments has been tested and is ready-to-use.
However, it has not been very stable yet and some minor improvements might be needed.

In the near future, we are going to add wrapper classes for PettingZoo and Ray RLlib.

Soure code of available wrappers: https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/wrappers.py


OpenAI Gym Environments
-----------------------------------

Our wrapper class for gym environment is pretty straightforward. You can just simply apply
a command to setup a gym-based environment, while creating a scenario.

.. code-block:: bash

    self._env = WrEnvGym([gym environment object], p_state_space:MSpace=None, p_action_space:MSpace=None, p_logging=True)

For more information, please check our how to files :ref:`here<target-howto-rl>`.


PettingZoo Environments
-----------------------------------

Under construction. The wrapper will be available soon.

.. code-block:: bash

    self._env = WrEnvPZoo([zoo environment object], p_state_space:MSpace=None, p_action_space:MSpace=None, p_logging=True)

Ray RLlib
-----------------------------------

Under construction. The wrapper will be available soon.

.. code-block:: bash

    wrPolicyRay(...)