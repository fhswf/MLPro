.. _target-package:

3rd Party Support
================

Add text here!

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

.. code-block:: python

    from mlpro.rl.wrappers import WrEnvGym
    
    self._env = WrEnvGym([gym environment object], p_state_space:MSpace=None, p_action_space:MSpace=None, p_logging=True)

For more information, please check our how to files :ref:`here<target-howto-rl>`.


PettingZoo Environments
-----------------------------------

Under construction. The wrapper will be available soon.

.. code-block:: python

    from mlpro.rl.wrappers import WrEnvPZoo
    
    self._env = WrEnvPZoo([zoo environment object], p_state_space:MSpace=None, p_action_space:MSpace=None, p_logging=True)

Stable-Baselines3
-----------------------------------

The stable-baselines3 can be used also in MLPro interface. The wrapper provides both the On-Policy and Off-Policy from stable-baselines3.

.. code-block:: python

    from stable_baselines3 import PPO
    from mlpro.rl.wrappers import WrPolicySB32MLPro

    class MyScenario(Scenario):

        C_NAME      = 'Matrix'

        def _setup(self, p_mode, p_ada, p_logging):
            gym_env     = gym.make('CartPole-v1')
            self._env   = WrEnvGYM2MLPro(gym_env, p_logging=False)

            policy_sb3 = PPO(
                policy="MlpPolicy",
                n_steps=5, 
                env=None,
                _init_setup_model=False)

            policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3, 
                p_observation_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space())

            self._agent = Agent(
                p_policy=self.policy_wrapped,   
                p_envmodel=None,
                p_name='Smith'
            )

For more information, please check our how to files `here <https://github.com/fhswf/MLPro/blob/main/examples/rl/Howto%2010%20-%20(RL)%20Train%20using%20SB3%20Wrapper.py>`_.


Ray RLlib
-----------------------------------

Under construction. The wrapper will be available soon.

.. code-block:: python

    from mlpro.rl.wrappers import wrPolicyRay

    wrPolicyRay(...)