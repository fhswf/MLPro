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

MLPro besteht aus einer kontinuierlich wachsenden Zahl sub-frameworks, die verschiedene Teilgebiete des machinellen Lernens abdecken.
Diese beinhalten ein oder meherere fundamentale Prozessmodelle (z.B. beim Reinforcement Learning den Markovian Decision Process) und dazu
passende Service- und Template-Klassen. Ferner beinhaltet jedes sub-framework einen spezifischen Pool an wiederverwendbaren Klassen für Algorithmen, Lernbeispielen,
Datenquellen etc. Zahlreiche Beispielprogramme für das Selbststudium (wir nennen sie "howtos") runden den Umfang ab.

Die genannten sub-frameworks setzen wiederum auf einer übergreifenden Schicht von Basisfunktionen auf. Dies ist ein gängiger und naheliegender
Ansatz. Besonders an MLPro ist jedoch der Umfang und die innere Struktur dieser Basisschicht. In einer Hierarchie aufeinenander aufbauender 
Unterschichten wird hier ein Spektrum von elementaren Funktionen für Logging und Plotting über Multitasking und Numerik bis hin zu den 
Grundlagen des maschinellen Lernens abgedeckt. Darin liegt auch der Schlüssel für die weitreichende Rekombinierbeit höherer Funktionen von MLPro begründet.

Tatsächlich denken wir bei jeder neuen Funktionalität darüber nach, wie tief wir sie in MLPro einsinken lassen können. Je tiefer der Ort,
desto universeller ist die Verwendbarkeit und damit die Reichweite innerhalb von MLPro. 

.. image:: images/MLPro_Architecture.drawio.png
   :scale: 80 %


Standardized Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A special feature of MLPro is that machine learning standards are already defined in the basic functions. 
Templates for adaptive models and their hyperparameters as well as for executable ML scenarios are introduced 
in the top layer of MLPro-BF. Furthermore, standards for training and hyperparameter tuning are defined. These 
basic machine learning elements are reused and specifically extended in all higher sub-frameworks. On the one hand, 
this facilitates the creation of new sub-frameworks and, on the other hand, the recombination of higher functions 
from MLPro in your own hybrid ML applications.

Learn more: :ref:`Basic Functions, Layer 4: Machine Learning <target_bf_ml>`


Real-World Applications in Focus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MLPro wurde zur Lösung realer Problemstellungen geschaffen...

- Detailliertes Logging
- Präzises Zeitmanagement von simulierten und realen Prozessen auf der Ebene von Mikrosekunden
- Erstellung detaillierter Trainingsdaten in CSV-Dateien
...


Third Party Support
^^^^^^^^^^^^^^^^^^^

MLPro integrates an increasing number of selected frameworks into its own process landscapes.
This is done at different levels of MLPro using so-called wrapper classes that are compatible with the corresponding MLPro classes.

Learn more: :ref:`Wrappers <target_wrappers>`


Example programs in double function: self-study and validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numerous executable example programs (we call them "howtos") illustrate the essential functions of MLPro.
They are also used for validation and are therefore an integral part of our automatic unit tests.
With this we ensure two things: the operability of all howtos and thus also the operability of the 
demonstrated functionalities (keyword: test driven development).

Learn more: :ref:`Appendix A1 - Example Pool <target_appendix1>`
