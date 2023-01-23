.. _target_mlpro_introduction:
Introduction
============

MLPro is a synoptic and standardized Python package to produce a solution for standard Machine Learning (ML) tasks.
In the first version of MLPro, MLPro provides sub-packages for a subtopic of ML, namely Reinforcement Learning (RL),
which is developed under a uniform infrastructure of basic and cross-sectional functionalities.
MLPro supports simulation as well as real-hardware implementations. MLPro team has developed this framework by taking care of
several main features, such as CI/CD method, clean code, object-oriented programming, ready-to-use functionalities, and clear documentation.

Additionally, we use established and well-known scientific terminologies in the naming of the development objects.
Although MLPro is standardized and has a high complexity, we make the implementation of MLPro as easy as possible, understandable, and flexible at the same time.
One of the properties of being flexible is the possibility to incorporate the widely-used third party packages in MLPro via wrapper classes.
The comprehensive and clear documentation also helps the user to quickly understand MLPro.

One of the main advantages of MLPro is the complete structure of MLPro that is not limited to only environments or policy and is not restricted to any dependencies.
MLPro covers environment, agents, multi-agents, model-based RL, and many more in a sub-framework, including cooperative Game Theoretical approach to solve RL problems.

We are committed to continuously enhancing MLPro, thus it can have more features and be applicable in more ML tasks.




Key Features
------------
   - Numerous extensive sub-frameworks for relevant ML areas like reinforcement learning, game theory, online machine learning, etc.
   - Powerful substructure of overarching basic functionalities for mathematics, data management, training and tuning of ML models, and much more
   - Numerous wrapper classes to integrate 3rd party packages


Brainstorming

a) Development: Intro or Project?
- design first
- clean code
- test automation


Architecture
------------

MLPro consists of a continuously growing number of sub-frameworks covering different areas of machine learning.
These include one or more fundamental process models (e.g. the Markovian Decision Process in reinforcement learning) and
appropriate service and template classes. Furthermore, each sub-framework contains a specific pool of reusable classes for 
algorithms, data sources, training subjects, etc. Numerous sample programs for self-study complete the scope.

The sub-frameworks mentioned are in turn based on an overarching layer of basic functions. This is a common and obvious 
approach. What is special about MLPro, however, is the scope and internal structure of this base layer. 
A spectrum of elementary functions for logging and plotting through multitasking and numerics to the basics of machine 
learning is covered in a hierarchy of sub-layers that build on one another. This is also the key to the far-reaching 
recombinability of higher functions of MLPro. In fact, with each new feature, we think about how deep we can sink it 
into MLPro. The deeper the place the more universal is the usability and thus the range within MLPro.

.. image:: images/MLPro_Architecture.drawio.png
   :scale: 85 %


Standardized Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A special feature of MLPro is that machine learning standards are already defined in the basic functions. 
Templates for adaptive models and their hyperparameters as well as for executable ML scenarios are introduced 
in the top layer of MLPro-BF. Furthermore, standards for training and hyperparameter tuning are defined. These 
basic machine learning elements are reused and specifically extended in all higher sub-frameworks. On the one hand, 
this facilitates the creation of new sub-frameworks and, on the other hand, the recombination of higher functions 
from MLPro in your own hybrid ML applications.

Learn more: :ref:`Basic Functions, Machine Learning <target_bf_ml>`


Example programs in double function: self-study and validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numerous executable example programs (we call them "howtos") illustrate the essential functions of MLPro.
They are also used for validation and are therefore an integral part of our automatic unit tests.
With this we ensure two things: the operability of all howtos and thus also the operability of the 
demonstrated functionalities (keyword: test driven development).

Learn more: :ref:`Example Pool <target_appendix1>`


Third Party Support
^^^^^^^^^^^^^^^^^^^

MLPro integrates an increasing number of selected frameworks into its own process landscapes.
This is done at different levels of MLPro using so-called wrapper classes that are compatible with 
the corresponding MLPro classes.

Learn more: :ref:`Wrappers <target_wrappers>`


Real-World Applications in Focus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MLPro was designed not only for simulations but for use in real-world applications. To this end, various 
basic functions have been implemented that make diagnostics easier and make optimal use of the 
available system resources. These are for example

- Detailed logging
- Precise time management of simulated and real processes on a microsecond time scale
- Creation of detailed training data files (ASCII/CSV)
- Multithreading/multiprocessing 

In addition, powerful templates for state-based systems are provided. They allow the standardized implementation 
of your own systems, which can then be controlled, for example, by adaptive controllers based on reinforcement 
learning or game theory. Furthermore, a wrapper for the popular physics engine `MuJoCo <https://mujoco.org/>`_ is 
provided, which can be used for the simulation and visualization of externally designed system models. The MLPro 
templates are also prepared for connection to industrial components like controllers, sensors, and actuators.

Learn more: :ref:`Elementary Functions <target_bf_elementary>`, :ref:`Computation <target_bf_computation>`, :ref:`State-based Systems <target_bf_systems>`
