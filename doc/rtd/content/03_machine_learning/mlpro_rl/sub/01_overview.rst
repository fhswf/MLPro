.. _target_overview_RL:
Overview
--------

MLPro-RL is the first ready-to-use subpackage in MLPro, designed specifically for reinforcement learning (RL)-related activities.
It provides comprehensive base classes for core RL components, including agents, environments, policies, multi-agent systems, and training frameworks.
The training loop follows the Markov Decision Process (MDP) model, as illustrated in the diagram below.

.. figure:: images/MDP.png
  :width: 600
  
  This figure is adapted from `Sutton and Barto, licensed by CC BY-NC-ND 2.0 <https://dl.acm.org/doi/10.5555/3312046>`_.


**Markov Decision Process (MDP)**

An MDP consists of two primary components: the environment and the agent.

- **Agent**: The decision-maker that selects actions based on its policy, considering the current state of the environment.

- **Environment**: The surrounding system in which the agent operates and interacts. The environment’s condition is represented by states.

MDP models the interaction between the agent and the environment.
The agent chooses an action and submits it to the environment, which then reacts by altering its state.
The environment provides feedback in the form of a new state and a reward, indicating the consequences of the action.
Through continuous interactions, the agent refines its policy to achieve optimal performance.

**Why Choose MLPro-RL?**

MLPro-RL supports a wide range of RL training configurations, including:

- Model-free and model-based RL

- Single-agent and multi-agent systems

- Simulated and real hardware implementations

This versatility makes MLPro-RL a valuable tool for students, educators, RL engineers, and researchers looking for a standardized and flexible RL framework.
The structure of MLPro-RL is depicted in the following figure:

.. figure:: images/MLPro-RL_overview.png
  :width: 600
  
  This figure is sourced from `MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

To begin using MLPro-RL, you can easily import the RL modules with the following command:

    .. code-block:: python

        from mlpro.rl import *

For a more comprehensive explanation of MLPro-RL, including a sample application on controlling a UR5 robot, refer to the paper:
`MLPro 1.0 - Standardized Reinforcement Learning and Game Theory in Python <https://doi.org/10.1016/j.mlwa.2022.100341>`_.


**Learn more**

  - :ref:`Getting started with MLPro-RL <target_getstarted_RL>`


**Cross reference**

  - :ref:`Related howtos <target_appendix1_RL>`
  - :ref:`API reference: MLPro-RL <target_api_rl>`
  - :ref:`API reference: MLPro-RL Pool of Objects <target_api_pool_rl>`
  - `MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_
