Release Notes
================


Version 0.9.1
---------------------

Release Highlights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Changes in gym wrapper due to new gym version of 0.25.0
- Gym wrapper can now use either the new step api or the old step api
- UR5 Environment (Simulation and Real)
- Wrapper for Scikit-learn, OpenML, River
- Extensions to the documentation



Version 0.9.0
---------------------

Release Highlights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- RL/GT: New environment model Homogeneous Matrix
- Revision of all example files (howtos)
- Extensions to the documentation
- Minor corrections
- Integration to Zenodo (doi management)



Version 0.8.6
---------------------

Release Highlights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Adding MANIFEST.in for including additional data package for UR5 ROS Environment



Version 0.8.5
---------------------

Release Highlights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- RL/GT: Improvements and fixes of training - especially stagnation detection
- RL/GT: Introduction of new environment Double Pendulum
- RL/GT: Improvement of installation procedure of UR5 environment
- Numerous corrections and improvements


.. New Features
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- BF: Loading/storing of objects uses Dill instead of Pickle
- RL/GT: minor corrections on MultiCartPole environment 
- RL/GT: performance optimization of SB3 wrapper (off-policy)
- RL/GT: Wrappers OpenAI Gym and PettingZoo: removed redundant cycle counting


Documentation Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Class diagrams resized
- Structure of environment documentation assigned to example documentation
- Appendix 1 (examples) restructured
- Disclaimer added


.. Others
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^



Version 0.8.1
---------------------

Release Highlights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

RL/GT: New environment Multi Geometry Robot


.. New Features
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Fixed Issues
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Documentation Changes
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Others
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^




Version 0.8.0
---------------------

Release Highlights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1 Sub-Package Reinforcement Learning

- Support of model-based agents (MBRL)
- Support of multi-agents (MARL)
- Training and hyperparameter tuning with progress detection
- Ready to use sample environments
- Full 3rd party support (OpenAI Gym, PettingZoo, Stable Baselines 3)

2 Sub-Package Game Theory

- Models for cooperative game theoretical approaches

3 Numerous ready to run examples

4 API Reference 


.. New Features
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Fixed Issues
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Documentation Changes
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Others
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^