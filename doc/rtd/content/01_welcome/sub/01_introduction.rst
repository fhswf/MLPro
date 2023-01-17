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
- test driven development
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


Maschinelles Lernen standardisiert
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...

Reale Anwendungen im Fokus
^^^^^^^^^^^^^^^^^^^^^^^^^^
...

Beispielprogramme in Doppelfunktion: Demonstration und Validierung
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Zahlreiche ausführbare Beispielprogramme veranschaulichen die wesentlichen Funktionen. Sie dienen
darüber hinaus der Validierung und sind daher auch fester Bestandteil unserer automatischen Unit tests. Damit stellen wir zweierlei sicher:
die Lauffähigkeit aller howtos und somit auch die Lauffähigkeit der demonstrierten Funktionalitäten (Stichwort: test driven development).

Learn more: link!
