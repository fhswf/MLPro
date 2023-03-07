.. _target_env_RL:
Environments
------------

In RL, the environment refers to the physical, virtual, or abstract system in which the agent interacts and learns.
The environment is the source of stimuli that the agent perceives and the arena in which it takes actions.

The environment is defined by a set of states, actions, and transition dynamics.
The state space is the set of all possible states that the agent can observe, and the action space is the set of all possible actions that the agent can take.
The transition dynamics describe how the environment changes in response to the agent's actions.

The environment also provides the agent with a reward signal that indicates how well it is doing in terms of achieving its goals.
The reward function is a mapping from states and actions to real-valued scalars that quantifies the desirability of each state-action pair.

The agent interacts with the environment over a sequence of time steps.
At each time step, the agent observes the current state of the environment and selects an action.
The environment then transitions to a new state and returns a reward signal to the agent.

Overall, the environment in RL provides the agent with the necessary information to learn a policy that maps states to actions and maximizes the cumulative reward signal.
The environment can be real-world or simulated, and can be described by a mathematical model or a black box.

MLPro-RL supplies two main classes for an environment to support model-free and model-based RL.
The first base class is Environment, which has a role as a template for designing environments for both approaches.
The second base class is EnvModel, which is adaptive and utilized in model-based RL.
Both Environment and EnvModel classes inherit a common base class EnvBase and its fundamental properties, e.g.
state and action space definition, reset the corresponding environment method, state transition method, etc.

There are two main possibilities to set up an environment in MLPro, such as,

.. toctree::
   :maxdepth: 1
   
   env/customenv
   env/pool

Alternatively, you can also :ref:`reuse available environments from 3rd-party packages via wrapper classes <target-package-third>` (currently available: OpenAI Gym or PettingZoo).
   
For reusing the 3rd packages, we develop a wrapper technology to transform the environment from the 3rd-party package to the MLPro-compatible environment.
Additionally, we also provide the wrapper for the other way around, which is from MLPro Environment to the 3rd-party package.
At the moment, there are two ready-to-use wrapper classes. The first wrapper class is intended for OpenAI Gym and the second wrapper is intended for PettingZoo.
The guide to using the wrapper classes is step-by-step explained in our how-to files, as follows:

(1) :ref:`OpenAI Gym to MLPro <Howto WP RL 004>`,

(2) :ref:`MLPro to OpenAI Gym <Howto WP RL 001>`,

(3) :ref:`PettingZoo to MLPro <Howto WP RL 003>`, and

(4) :ref:`MLPro to PettingZoo <Howto WP RL 002>`.
