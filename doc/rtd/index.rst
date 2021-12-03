.. MLPro Documentations documentation master file, created by
   sphinx-quickstart on Wed Sep 15 12:06:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLPro - Machine Learning Professional
================================================

`MLPro <https://github.com/fhswf/MLPro.git>`_ is a synoptic framework for standardized machine learning tasks in Python!

MLPro was developed in 2021 by `Automation Technology and Learning Systems team at Fachhochschule Südwestfalen <https://www.fh-swf.de/de/forschung___transfer_4/labore_3/labs/labor_fuer_automatisierungstechnik__soest_1/standardseite_57.php>`_.

MLPro provides complete, standardized, and reusable functionalities to support your scientific research, educational task, or industrial project in machine learning.

In the first version of MLPro, we provide a standardized Python package for reinforcement learning (RL) and game
theoretical (GT) approaches, including environments, algorithms, multi-agent RL (MARL), model-based RL (MBRL) and many more.
Additionally, we incorporate the available third party packages by developing wrapper classes
to enable our users to reuse the third party packages in MLPro.

Github repository: https://github.com/fhswf/MLPro.git


Main Contributions
--------------

- Test-driven development (CI/CD concept)
- Clean code and constructed through Object-Oriented Programming
- Ready-to-use functionalities
- Usability in scientific, industrial and educational contexts
- Extensible, maintainable, understandable
- Attractive UI support (available soon)
- Reuse of available state-of-the-art implementations
- Clear documentations


Instructions for use
--------------

.. toctree::
   :maxdepth: 2
   :caption: 1 Introduction
   
   content/intro/overview
   content/intro/getstarted
   content/intro/architecture
   content/intro/dependencies
   
.. toctree::
   :maxdepth: 2
   :caption: 2 MLPro-BF – Basic Functions
   
   content/bf/elementary
   content/bf/math
   content/bf/ml

.. toctree::
   :maxdepth: 3
   :caption: 3 MLPro-RL – Reinforcement Learning
   
   content/rl/overview
   content/rl/env
   content/rl/agents
   content/rl/scenario
   content/rl/train
   content/rl/wrapper

.. toctree::
   :maxdepth: 2
   :caption: 4 MLPro-GT – Game Theory
   
   content/gt/overview
   content/gt/players
   content/gt/gameboard

.. toctree::
   :maxdepth: 2
   :caption: 5 MLPro-UI – Interactive ML
   
   content/ui/sciui

.. toctree::
   :maxdepth: 2
   :caption: Appendix 1: List of Examples
   
   content/append1/test

.. toctree::
   :maxdepth: 4
   :caption: Appendix 2: API Reference
   
   content/append2/mlpro.core
   content/append2/mlpro.wrappers
   content/append2/mlpro.pool
   content/append2/mlpro.template

.. toctree::
   :maxdepth: 2
   :caption: Appendix 3: Project MLPro
   
   content/append3/versions
   content/append3/paper
   content/append3/cont
   
Citing MLPro
------------------------
To cite this project in publications:

.. code-block:: bibtex

    @misc{...
    }

Contact Data
------------------------
Mail: mlpro@listen.fh-swf.de
