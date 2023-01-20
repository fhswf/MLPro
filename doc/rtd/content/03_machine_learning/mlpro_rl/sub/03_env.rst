.. _target_env_RL:
Environments
------------

MLPro-RL supplies two main classes for an environment to support model-free and model-based RL.
The first base class is Environment, which has a role as a template for designing environments for both approaches.
The second base class is EnvModel, which is adaptive and utilized in model-based RL.
Both Environment and EnvModel classes inherit a common base class EnvBase and its fundamental properties, e.g.
state and action space definition, reset the corresponding environment method, state transition method, etc.

There are four main possibilities to set up an environment in MLPro, such as,

(1) the user can develop a custom environment,

(2) the user can develop a custom environment model (specifically for model-based RL),

(3) the user can reuse the provided environments by accessing them from the pool of objects, and

(4) the user can reuse available environments from 3rd-party packages via wrapper classes (currently available: OpenAI Gym or PettingZoo).

.. toctree::
   :maxdepth: 1
   
   env/customenv
   env/customenvmodel
   env/pool
   
For reusing the 3rd packages, we develop a wrapper technology to transform the environment from the 3rd-party package to the MLPro-compatible environment.
Additionally, we also provide the wrapper for the other way around, which is from MLPro Environment to the 3rd-party package.
At the moment, there are two ready-to-use wrapper classes. The first wrapper class is intended for OpenAI Gym and the second wrapper is intended for PettingZoo.
The guide to using the wrapper classes is step-by-step explained in our how-to files, as follows:

(1) :ref:`OpenAI Gym to MLPro <Howto WP RL 004>`,

(2) :ref:`MLPro to OpenAI Gym <Howto WP RL 001>`,

(3) :ref:`PettingZoo to MLPro <Howto WP RL 003>`, and

(4) :ref:`MLPro to PettingZoo <Howto WP RL 002>`.
