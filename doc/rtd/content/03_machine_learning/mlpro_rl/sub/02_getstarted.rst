.. _target_getstarted_RL:
Getting Started
---------------

Here is a concise series to introduce all users (from the first-timer to the advanced MLPro user) to the MLPro-RL in a practical way.

If you are a first-timer, then you can begin with :ref:`understanding MLPro <target_getstarted_RL_1>`.
If you have understood MLPro but not reinforcement learning, then you can jump to :ref:`understanding reinforcement learning <target_getstarted_RL_2>`.
If you have experienced in both MLPro and reinforcement learning, then you can directly start with :ref:`understanding MLPro-RL <target_getstarted_RL_3>`.
After following this step-by-step guideline, we expect the user understands the MLPro-RL in practice and starts using MLPro-RL.

.. _'target_getstarted_RL_1':
Understanding MLPro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you are the first-time user of MLPro, you might wonder what is actually MLPro.
Therefore, we recommend to initially start with understanding MLPro by checking out the following steps:

(1) :ref:`MLPro: An Introduction <target_mlpro_introduction>`,

(2) `introduction video of MLPro <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc1.mp4>`_,

(3) :ref:`installing and getting started with MLPro <target_mlpro_getstarted>`, and

(4) optionally `MLPro paper in Software Impact journal <https://doi.org/10.1016/j.simpa.2022.100421>`_.

.. _target_getstarted_RL_2:
Understanding Reinforcement Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you have not dealt with reinforcement learning, we recommend to start understanding at least the basic concept of reinforcement learning.
There are plenty of references, articles, papers, books, or videos in the internet that explains reinforcement learning.
But, for deep understanding, we recommend you to read the book from Sutton and Barto, which is `Reinforcement Learning: An Introduction <https://dl.acm.org/doi/10.5555/3312046>`_.

.. _target_getstarted_RL_3:
Understanding MLPro-RL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We expect that you have a basic knowledge of MLPro and reinforcement learning.
Therefore, you need to understand the overview MLPro-RL by following the steps below:

(1) :ref:`MLPro-RL introduction page <target_overview_RL>`, and

(2) `Section 4 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

Understanding Environment in MLPro-RL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First of all, it is important to understand the structure of an environment in MLPro, which can be found in :ref:`this page <target_env_RL>`.

Then, you can start following some of our howto files related to environment in MLPro-RL, as follows:

(1) :ref:`Howto RL-001: Reward <Howto RL 001>`, and

(2) :ref:`Howto RL-ENV-004: A Random Agent on Double Pendulum Environment <Howto Env RL 004>`.

Understanding Agent in MLPro-RL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In reinforcement learning, we have two types of agent, such as a single-agent RL or a multi-agent RL. Both of the types are covered by MLPro-RL.
To understand different possibilities of an agent in MLPro, you can visit :ref:`this page <target_agents_RL>`.

Then, you need to understand how to set up a single-agent and a multi-agent RL in MLPro-RL by following these examples:

(1) :ref:`Howto RL-AGENT-001: Run an Agent with Own Policy <Howto Agent RL 001>`, and

(2) :ref:`Howto RL-AGENT-003: Run Multi-Agent with Own Policy <Howto Agent RL 003>`.

Choosing between Model-Free and Model-Based RL 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this section, you need to select your direction of the RL training, whether it is a model-free RL or a model-based RL.
However, firstly, you can pay attention to these two pages, which are :ref:`RL scenario <target_scenario_RL>` and :ref:`training <target_training_RL>`, before selecting either of the paths below.

.. toctree::
   :maxdepth: 1
   
   getstarted/mf_rl
   getstarted/mb_rl

Additional Guidance 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After following the previous steps, we hope that you could practice MLPro-RL and start using this subpackage for your RL-related activities.
For more advanced features, we highly recommend you to check out the following howto files:

(1) :ref:`Howto RL-AGENT-005: Train and Reload Single Agent <Howto Agent RL 005>`,

(2) :ref:`Howto RL-HT-001: Hyperopt <Howto HT RL 001>`,

(3) :ref:`Howto RL-HT-002: Optuna <Howto HT RL 002>`,

(4) :ref:`Howto RL-ATT-001: Stagnation Detection <Howto RL ATT 001>`, and

(5) :ref:`Howto RL-ATT-002: SB3 Policy with Stagnation Detection <Howto RL ATT 002>`.
