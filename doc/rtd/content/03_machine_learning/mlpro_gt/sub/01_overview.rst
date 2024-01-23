.. _target_overview_GT:
Overview
--------

Game theory (GT) is a branch of mathematics and economics that studies strategic interactions among rational decision-makers.
It has applications in various fields, including economics, engineering, biology, political science, and computer science.
In essence, GT provides a framework for analyzing and understanding the behavior of individuals or entities when their decisions affect each other.

In the native GT, at its core, GT models decision-making scenarios where the outcome of an individual's choice depends not only on their actions but also on the actions of others.
These scenarios, known as "games," involve players, strategies, and payoffs.
Players make decisions based on their understanding of the strategic environment, aiming to maximize their utility or payoff.

MLPro-GT is designed to empower researchers, analysts, and developers with a comprehensive set of tools for game theory analysis.
Python's simplicity, readability, and extensive libraries make it an ideal language for implementing and experimenting with GT models.
Our package distinguishes itself by offering two powerful sub-frameworks:

  **(1) MLPro-GT-Native**: This framework focuses on modeling and analyzing traditional static or native games.
  Native games capture strategic interactions where players make decisions simultaneously.
  Our Python package simplifies the representation and analysis of native games, providing a user-friendly interface to explore solution concepts such as Nash equilibrium.

  **(2) MLPro-GT-DG**: DG stands for dynamic games.
  For scenarios involving sequential decision-making or repeated interactions, our dynamic games framework comes into play.
  Dynamic GT extends the analysis to capture evolving strategies over time, allowing users to model and simulate complex, dynamic strategic environments.


MLPro-GT-DG is developed as a sub-framework for the cooperative GT approach to solving multi-agent problems by inheriting a handful of main functionalities of MLPro-RL,
such as the environment model, the agent model, and the environment-agent interaction model. This sub-framework focuses on the cooperative GT approach to Markov games.
A Markov game contains a group of independent players that make decisions simultaneously, see the figure below for an overview.

.. figure:: images/MLPro_GT_Game.png
  :width: 400
  
  This figure is taken from `MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

If you are interested to utilize MLPro-GT-Native and MLPro-GT-DG, you can easily access the GT modules, as follows:

    .. code-block:: python

        from mlpro.gt.native import *
        from mlpro.gt.dynamicgames import *


Additionally, you can find more comprehensive explanations of MLPro-GT-DG including a sample application and difference with a native RL approach in this paper:
`MLPro 1.0 - Standardized Reinforcement Learning and Game Theory in Python <https://doi.org/10.1016/j.mlwa.2022.100341>`_.


**Learn more**

  - :ref:`Getting started with MLPro-GT <target_getstarted_GT>`


**Cross Reference**

  - :ref:`Related Howtos <target_appendix1_GT>`
  - :ref:`API Reference: MLPro-GT <target_api_gt>`
  - :ref:`API Reference: MLPro-GT Pool of Objects <target_api_pool_gt>`
  - `MLPro 1.0 Paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_