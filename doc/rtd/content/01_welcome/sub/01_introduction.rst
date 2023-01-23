.. _target_mlpro_introduction:
Introduction
============

MLPro ist ein übergreifendes und integratives middleware-framework für standardisierte Anwendungen des Maschinellen Lernens (ML) in Python. 

Die Zielsetzung besteht darin, Prozesse und Vorlagen für ein breites Spektrum relevanter ML-Teilgebiete zur Verfügung zu stellen, ohne dabei
auf den Einsatz bereits etablierter und bewährter ML-Frameworks wie Scikit-learn, TensorFlow, PyTorch, Optuna, etc. verzichten zu müssen. Letztere
werden vielmehr nahtlos in die Prozesslandschaft von MLPro integriert. Forscher, Entwickler, Ingenieure, und Studenten können sich 
durch die Verwendung von MLPro somit auf die wesentlichen Kernaufgaben konzentrieren, ohne sich um das Zusammenspiel verschiedener
Frameworks kümmern zu müssen oder existente Algorithmen neu implementieren zu müssen.

MLPro ist dabei architektonisch auf beliebige Erweiterbarkeit und Rekombinierbarkeit ausgelegt, was insbesondere die Erstellung hybrider ML-Anwendungen
über verschiedene Lernparadigmen hinweg ermöglicht.


Key Features
------------
   - Continuously growing number of sub-frameworks for relevant ML areas like reinforcement learning, game theory, online machine learning, etc.
   - Powerful substructure of overarching basic functionalities for mathematics, data management, training and tuning of ML models, and much more
   - Integration of proven 3rd party packages
   - Open Source, open design


Development
-----------
MLPro is developed at the `South Westphalia University of Applied Sciences <https://www.fh-swf.de/en/international_3/index.php>`_ in the 
`Department for Electrical Power Engineering <https://www.fh-swf.de/en/ueber_uns/standorte_4/soest_4/fb_eet/index.php>`_ in the `Lab 
for Automation Technology and Learning Systems <https://www.fh-swf.de/en/forschung___transfer_4/labore_3/labs/labor_fuer_automatisierungstechnik__soest_1/standardseite_57.php>`_ 
and is therefore freely available to all interested users from research and development as well as industry and economy.


The development team consistently applies the following principles:

   * Quality first
      Our aim is to provide ML functionalities at the highest level. We put these up for discussion in scientific :ref:`publications <target_publications>`. 
      Open feedback and suggestions for improvement are always welcome.

   * Design first
      In MLPro, new functions are not created in the code editor but in a class diagram. We provide the latter in the 
      :ref:`Appendix A2 - API Reference <target_appendix2>`. A color system documents the respective development status.

   * Clean Code Paradigm
      We firmly believe that a clearly structured and legible source text has a significant influence on both the acceptance and the life 
      cycle of a software. Anyone who opens any source code in MLPro knows immediately what we mean :-)


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
basic ML elements are reused and specifically extended in all higher sub-frameworks. On the one hand, 
this facilitates the creation of new sub-frameworks and, on the other hand, the recombination of higher functions 
from MLPro in your own hybrid ML applications.

Learn more: :ref:`Basics of Machine Learning <target_bf_ml>`


Example Pool
^^^^^^^^^^^^

Numerous executable example programs (we call them "howtos") illustrate the essential functions of MLPro.
They are also used for validation and are therefore an integral part of our automatic unit tests.
With this we ensure two things: the operability of all howtos and thus also the operability of the 
demonstrated functionalities (tdd - test driven development).

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
