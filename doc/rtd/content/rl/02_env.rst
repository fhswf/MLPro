4.2 Environments
================

MLPro-RL supplies two main classes for environment to support model-free and model-based RL.
The first base class is Environment, which has a role as a template for designing environemnts for both approaches.
The second base class is EnvModel, which is adaptive and utilized in model-based RL.
Both Environment and EnvModel classes inhereit a common base class EnvBase and its fundamental properties, e.g.
state and action space definition, reset the corresponding environment method, state transition method, etc.

There are four main possibilities to set up a in MLPro, such as,

(1) the user can develop a custom environment,

(2) the user can develop a custom environment model (specifically for model-based RL),

(3) the user can reuse the provided environments by accessing them from the pool of objects, and

(4) the user can reuse available environments from 3rd-party packages via wrapper classes (Available currently: OpenAI Gym or PettingZoo).

.. toctree::
   :maxdepth: 1
   
   env/customenv
   env/customenvmodel
   env/pool
   
For reusing the 3rd packages, we develop a wrapper technology to transform the environment from the 3rd-party package to MLPro-compatible environment.
Additionally, we also provide the wrapper for the other way around, which is from MLPro Environment to the 3rd-party package.
At the moment, there are two ready-to-use wrapper classes. The first wrapper class is intended for OpenAI Gym and the second wrapper is intended for PettingZoo.
The guide of using the wrapper classes are step-by-step explained in our how-to files, as follows:
(1) `OpenAI Gym to MLPro, <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.002.html>`_
(2) `MLPro to OpenAI Gym, <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.008.html>`_
(3) `PettingZoo to MLPro, <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.006.html>`_ and
(4) `MLPro to PettingZoo. <https://mlpro.readthedocs.io/en/latest/content/append1/rl/howto.rl.009.html>`_
