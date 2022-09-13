.. MLPro Documentations documentation master file, created by
   sphinx-quickstart on Wed Sep 15 12:06:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLPro - Machine Learning Professional
=====================================

Welcome to MLPro - the integrative middleware-framework for standardized machine learning in Python!

MLPro is developed by scientists to enable
   - real world ML projects at a high quality level
   - comparable and reproducible results in publications
   - the exchange and reuse of standardized ML code

For this purpose, MLPro provides advanced models and templates at a scientific level for a constantly growing number of sub-areas of machine learning.
These are embedded in standard processes for training and real operations. But of course we have not reinvented existing wheels. An integral part of MLPro's philosophy is to seamlessly integrate proven functionalities of relevant 3rd party packages instead of developing them again.
The scope is rounded off by numerous executable example programs that make it easier to get started in the world of MLPro.

MLPro is also present on...
   - `ResearchGate <https://www.researchgate.net/project/MLPro-A-Synoptic-Framework-for-Standardized-Machine-Learning-Tasks-in-Python>`_
   - `Python Package Index (PyPI) <https://pypi.org/project/mlpro/>`_
   - `Anaconda <https://anaconda.org/mlpro/mlpro/>`_
   - `GitHub <https://github.com/fhswf/MLPro>`_


Notes on the current version:
   - MLPro already provides two stable sub-frameworks MLPro-RL for reinforcement learning and MLPro-GT for game theory.
   - The documentation is not quite complete yet, but we are working hard on it and the numerous sample programs in :ref:`Appendix 1 <target_appendix1>` and the API specification in :ref:`Appendix 2 <target_appendix2>` should help in the meantime. 
   - Next sub-framework in progress: MLPro-OA for online adaptive systems...





Table of Content
================

.. toctree::
   :maxdepth: 1
   :caption: 1 Introduction
   
   content/intro/overview
   content/intro/getstarted
   content/intro/architecture



.. toctree::
   :maxdepth: 2
   :caption: 2 MLPro-BF – Basic Functions
   :glob:
   
   content/bf/*



.. toctree::
   :maxdepth: 3
   :caption: 3 MLPro-SL - Supervised Learning
   :glob:
   
   content/sl/*



.. toctree::
   :maxdepth: 3
   :caption: 4 MLPro-RL - Reinforcement Learning
   :glob:

   content/rl/*



.. toctree::
   :maxdepth: 2
   :caption: 5 MLPro-GT – Game Theory
   :glob:
   
   content/gt/*



.. toctree::
   :maxdepth: 1
   :caption: 6 MLPro-OA – Online Adaptivity
   :glob:

   content/oa/*


.. _target_appendix1:
.. toctree::
   :maxdepth: 2
   :caption: Appendix 1: Examples
   
   content/append1/howto.bf
   content/append1/howto.rl
   content/append1/howto.gt
   content/append1/howto.oa



.. _target_appendix2:
.. toctree::
   :maxdepth: 6
   :caption: Appendix 2: API Reference
   
   content/append2/mlpro.core
   content/append2/mlpro.wrappers
   content/append2/mlpro.pool
   content/append2/mlpro.template


.. _target_appendix3:
.. toctree::
   :maxdepth: 2
   :caption: Appendix 3: Project MLPro
   
   content/append3/versions
   content/append3/publications
   content/append3/cont
   content/append3/disclaimer
   


Contact Data
------------------------
Mail: mlpro@listen.fh-swf.de
