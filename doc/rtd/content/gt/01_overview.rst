5.1 Overview
================

Game Theory (GT) is well-known in economic studies as a theoretical approach to model the strategic
interaction between multiple individuals or players in a specific situation. Game Theory
approach can also be adopted in the science area to optimize decision-making processes in a
strategic setting and has been successfully solved Multi-Agent RL (MARL) problems.
If you would like to know more about the corporation between GT and MARL, you can have a look at these papers:
`(1) self-optimization using a State-based Potential Game approach <https://www.researchgate.net/publication/341980093_Distributed_Self-Optimization_of_Modular_Production_Units_A_State-Based_Potential_Game_Approach>`_ and
`(2) potential game-based distributed optimization of a production unit <https://www.researchgate.net/publication/332868950_Potential_Game_based_Distributed_Optimization_of_Modular_Production_Units>`_.

MLPro-GT is developed as a sub-framework for the cooperative GT approach to solving MARL problems by inheriting a handful of main functionalities of MLPro-RL,
such as the environment model, the agent model, and the environment-agent interaction model.
MLPro-GT serves as a child package and MLPro-RL is its parent package, which is explained in the simplified diagram below:

.. image:: images/MLPro-GT_class_diagram.png

This sub-framework foucuses on the cooperative GT approach on Markov games. A Markov game contains a group of independent players that make decisions simultaneuosly,
see the figure below for overview.

.. image:: images/MLPro_GT_Game.png

You can easily access the GT module, as follows:

    .. code-block:: python

        from mlpro.gt.models import *

You can check out some of the examples on our `how to files <https://mlpro.readthedocs.io/en/latest/content/append1/howto.gt.html>`_
or `here <https://github.com/fhswf/MLPro/tree/main/examples/gt>`_.