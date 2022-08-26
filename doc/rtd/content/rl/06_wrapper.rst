.. _target-package:

4.6 3rd Party Support
=========================

`MLPro <https://github.com/fhswf/MLPro.git>`_ allows the user to reuse widely-used 3rd-party packages and
integrate them to the MLPro interface and also the other way around via wrapper classes.
Therefore, the user is free to select an environment and/or a policy from the 3rd-party packages and the native MLPro-RL.
It is also possible to combine an environment and a policy from different packages.

At the moment, we have five ready-to-use wrapper classes related to RL from 3rd-party packages to MLPro and two wrapper classes from MLPro to 3rd-party packages, such as:

(1) RL Environment: OpenAI Gym to MLPro

(2) RL Environment: MLPro to OpenAI Gym

(3) RL Environment: PettingZoo to MLPro

(4) RL Environment: MLPro to PettingZoo

(5) RL Policy: StableBaselines3 to MLPro

Moreover, wrapper classes for hyperparameter tuning by `Optuna <https://mlpro.readthedocs.io/en/latest/content/append1/bf/howto.bf.008.html>`_ and `Hyperopt <https://mlpro.readthedocs.io/en/latest/content/append1/bf/howto.bf.007.html>`_ can also be incorporated to your RL training.


RL Environment: OpenAI Gym to MLPro
------------------------------------------

Here is the wrapper class to convert RL Environment from OpenAI Gym to MLPro.
The implementation is pretty simple and straightforward.
The user can call the wrapper class while setting up an environment, as follows:

.. code-block:: python

    from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
    import gym
    
    p_gym_env = gym.make('CartPole-v1', new_step_api=True, render_mode=None)
    self._env = WrEnvGYM2MLPro(p_gym_env, p_logging=True)

For more information, please check the `how-to file <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.002.html>`_.


RL Environment: MLPro to OpenAI Gym
------------------------------------------

Here is the wrapper class to convert RL Environment from MLPro to OpenAI Gym.
The implementation is pretty simple and straightforward.
The user can call the wrapper class while setting up an environment, as follows:

.. code-block:: python

    from mlpro.wrappers.openai_gym import WrEnvMLPro2GYM
    from mlpro.rl.pool.envs.gridworld import GridWorld
    
    mlpro_env = GridWorld(p_logging=Log.C_LOG_ALL)
    env = WrEnvMLPro2GYM(mlpro_env, p_state_space=None, p_action_space=None, p_new_step_api=True)

For more information, please check the `how-to file <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.008.html>`_.


RL Environment: PettingZoo to MLPro
------------------------------------------

Here is the wrapper class to convert RL Environment from PettingZoo to MLPro.
The implementation is pretty simple and straightforward.
The user can call the wrapper class while setting up an environment, as follows:

.. code-block:: python

    from pettingzoo.butterfly import pistonball_v6
    from mlpro.wrappers.pettingzoo import WrEnvPZOO2MLPro
    
    p_zoo_env = pistonball_v6.env()
    self._env = WrEnvPZOO2MLPro(p_zoo_env, p_logging=True)

For more information, please check the `how-to file <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.006.html>`_.


RL Environment: MLPro to PettingZoo
------------------------------------------

Here is the wrapper class to convert RL Environment from MLPro to PettingZoo.
The implementation is pretty simple and straightforward.
The user can call the wrapper class while setting up an environment, as follows:

.. code-block:: python

    from mlpro.wrappers.pettingzoo import WrEnvMLPro2PZoo
    from mlpro.rl.pool.envs.bglp import BGLP
    
    mlpro_env = BGLP(p_logging=Mode.C_LOG_ALL)
    env = WrEnvMLPro2PZoo(mlpro_env, p_num_agents=5, p_state_space=None, p_action_space=None).pzoo_env

For more information, please check the `how-to file <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.009.html>`_.


RL Policy: StableBaselines3 to MLPro
------------------------------------------

Here is the wrapper class to convert RL Environment from StableBaselines3 to MLPro.
The wrapper provides both the On-Policy and Off-Policy from StableBaselines3.
The implementation is pretty simple and straightforward.
The user can call the wrapper class while setting up an environment, as follows:

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
                _init_setup_model=False,
                device="cpu")

            policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3,
                p_cycle_limit=self._cycle_limit,
                p_observation_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_ada=p_ada,
                p_logging=p_logging)

            return Agent(
                p_policy=policy_wrapped,
                p_envmodel=None,
                p_name='Smith',
                p_ada=p_ada,
                p_logging=p_logging
            )

For more information, please check the `how-to file <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.007.html>`_.


