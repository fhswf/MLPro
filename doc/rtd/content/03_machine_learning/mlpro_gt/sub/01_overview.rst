.. _target_overview_GT:
Overview
--------

Game Theory (GT) is a well-known branch of economic studies as a theoretical approach to modelling the strategic
interaction between multiple individuals or players in a specific circumstance. GT can also be adopted in the scientific and engineering area, for instance,
to optimize decision-making processes in a strategic setting. Moreover, GT has successfully solved Multi-Agent Reinforcement Learning (MARL) problems.
If you would like to know more about the corporation between GT and MARL, you can have a look at these papers:

  (1) `Self-optimization using a State-based Potential Game approach <https://www.researchgate.net/publication/341980093_Distributed_Self-Optimization_of_Modular_Production_Units_A_State-Based_Potential_Game_Approach>`_ and
  
  (2) `Potential game-based distributed optimization of a production unit <https://www.researchgate.net/publication/332868950_Potential_Game_based_Distributed_Optimization_of_Modular_Production_Units>`_.

MLPro-GT is developed as a sub-framework for the cooperative GT approach to solving MARL problems by inheriting a handful of main functionalities of MLPro-RL,
such as the environment model, the agent model, and the environment-agent interaction model. This sub-framework focuses on the cooperative GT approach to Markov games.
A Markov game contains a group of independent players that make decisions simultaneously, see the figure below for an overview.

.. figure:: images/MLPro_GT_Game.png
  :width: 600
  
  This figure is taken from `MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

Additionally, you can find more comprehensive explanations of MLPro-GT including a sample application and difference with a native RL approach in this paper:
`MLPro 1.0 - Standardized Reinforcement Learning and Game Theory in Python <https://doi.org/10.1016/j.mlwa.2022.100341>`_.
The simplified diagram below shows the architecture of MLPro-GT, where MLPro-GT serves as a child package and MLPro-RL is its parent package.

.. figure:: images/MLPro-GT_class_diagram.png
  :width: 600
  
  This figure is taken from `MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

If you are interested to utilize MLPro-GT, you can easily access the GT modules, as follows:

    .. code-block:: python

        from mlpro.gt import *


MLPro team recommends visiting our :ref:`getting started <target_getstarted_GT>` page to directly learn MLPro-GT in a practical way.
You can also check out numerous ready-to-run examples on our :ref:`how to files <target_appendix1>`.
Moreover, technical API documentation can be found in :ref:`appendix 2 <target_appendix2>`.