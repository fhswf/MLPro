.. _target_overview_SL:
Overview
--------

MLPro provides a subtopic package for supervised learning, namely MLPro-SL.
At the moment, the implementation is still limited but we are working on it and improving it to bring you full supervised learning functionalities in the near future.
MLPro-SL is designed to handle online and offline supervised learning, which means that the model can be used for different purposes, e.g. model-based reinforcement learning, online adaptivity, and more.

The current implementation covers:

 - A base class of an adaptive function for supervised learning
 - A base class of an adaptive function for feedforward neural networks, including MLP
 - Ready-to-use PyTorch-based MLP networks in the pool of objects

**Learn more**

  - :ref:`Getting started with MLPro-SL <target_getstarted_SL>`


**Cross Reference**
    - :ref:`Howto RL-MB-001: Train and Reload Model Based Agent (Gym) <Howto MB RL 001>`
    - :ref:`Howto RL-MB-002: MBRL with MPC on Grid World Environment <Howto MB RL 002>`
    - :ref:`Howto RL-MB-003: MBRL on RobotHTM Environment <Howto MB RL 003>`
    - :ref:`API Reference: MLPro-SL <target_api_sl>`
    - :ref:`API Reference: MLPro-SL Pool of Objects <target_api_pool_sl>`