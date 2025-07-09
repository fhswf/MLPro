.. _target_env_RL:

Environments
------------

In RL, the environment represents the system—whether physical, virtual, or abstract—in which an agent interacts and learns.
It provides the stimuli that the agent perceives and serves as the arena where actions are taken.

An environment is defined by a set of states, actions, and transition dynamics:

   - State Space: The collection of all possible states that the agent can observe.

   - Action Space: The set of all possible actions the agent can take.

   - Transition Dynamics: The rules that dictate how the environment changes in response to the agent's actions.

Additionally, the environment supplies a reward signal to indicate the agent's performance in achieving its objectives.
The reward function maps states and actions to real-valued scalars, quantifying the desirability of each state-action pair.

The agent interacts with the environment over sequential time steps:

   (1) The agent observes the current state of the environment.

   (2) It selects an action.

   (3) The environment transitions to a new state and returns a reward signal.

Ultimately, the environment enables the agent to learn a policy that maps states to actions, with the goal of maximizing cumulative rewards.
Environments can be real-world or simulated, and they may be represented using mathematical models or treated as black boxes.

**MLPro-RL Environment Classes**

MLPro-RL provides two primary base classes to support both model-free and model-based RL:

   - Environment: A template for designing environments in either RL approach.

   - EnvModel: An adaptive class used in model-based RL.

Both classes inherit from the common base class **EnvBase**, which provides fundamental properties such as:

   - State and action space definitions.

   - Resetting the environment.

   - State transition methods.

**Setting Up an Environment in MLPro-RL**

There are two main ways to set up an environment in MLPro-RL:

   .. toctree::
      :maxdepth: 1
      
      env/customenv
      env/pool

**Third-Party Environment Wrappers**

Alternatively, you can also reuse environments from third-party packages via :ref:`wrapper classes <target_extension_hub>`.
   
MLPro-RL includes wrapper technology to integrate external environments with MLPro's framework.
Additionally, wrappers are available for converting MLPro environments into third-party formats.
Currently, MLPro-RL offers two ready-to-use wrapper classes:

   - Gymnasium Wrapper

   - PettingZoo Wrapper

Step-by-step guides for using these wrappers are available:

   (1) `Gymnasium to MLPro <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_wp_002_gymnasium_environment_to_mlpro_environment.html>`_,

   (2) `MLPro to Gymnasium <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_wp_001_mlpro_environment_to_gymnasium_environment.html>`_,

   (3) `PettingZoo to MLPro <https://mlpro-int-pettingzoo.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_wp_002_run_multiagent_with_own_policy_on_petting_zoo_environment.html>`_, and

   (4) `MLPro to PettingZoo <https://mlpro-int-pettingzoo.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_wp_001_mlpro_environment_to_petting_zoo_environment.html>`_.

By leveraging MLPro-RL's environment capabilities, you can seamlessly integrate, design, and deploy RL environments suited to your needs.