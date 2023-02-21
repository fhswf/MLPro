.. _target_bf_ml_scenario:
ML Scenarios
============

Wie eingangs bereits erwähnt, werden adaptive Modelle in MLPro zusammen mit ihrem konkreten Kontext zu einem ML Szenario
zusammengefasst. Hierfür stellt MLPro die abstrakte Template-Klasse **bf.ml.Scenario** zur Verfügung. Diese ist auf dieser 
Ebene noch nicht für die Verwendung in eigenen Kundenanwendungen gedacht sondern dient hier lediglich der Standardisierung 
der grundlegenden Eigenschaften eines ML-Szenarios. Von der Wurzelklasse aller Szenarien in MLPro 
:ref:`bf.ops.ScenarioBase <target_bf_ops>` erbt sie die folgenden Eigenschaften

- ein Betriebsmodus (Simulation oder Realbetrieb)
- Ausführung von Zyklen
- Persistierbarkeit
- Visualiserung

und fügt auf dieser Ebene das Management eines internen adaptiven Modells hinzu.

[ Klassen-Image ]

...



**Cross Reference**

- :ref:`Class bf.ops.ScenarioBase <target_bf_ops>`
- :ref:`API Reference MLPro-BF-ML <target_api_bf_ml>`