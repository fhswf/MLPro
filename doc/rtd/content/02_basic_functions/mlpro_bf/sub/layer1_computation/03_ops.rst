.. _target_bf_ops:
Operations
==========

In this module, classes are made available that are required for the operative execution of higher functions of MLPro. 
In particular, the **ScenarioBase** class is introduced here, which serves as an abstract template for various 
scenarios, such as :ref:`Stream Scenarios <target_bf_streams_processing_01>` or 
:ref:`ML Scenarios <target_bf_ml_scenario>`. In this respect, the scenario in MLPro is one of the fundamental concepts, 
which already introduces the following properties at this low level:

- Runtime mode (simulation or real operation)
- Execution of cycles
- Internal Timer-Management
- Persistence

All scenario classes in MLPro are ultimately template classes for implementing your own concrete applications. 
Therefore, special attention should be paid to the custom methods that are already introduced here.


**Cross Reference**

- :ref:`API Reference BF-OPS - Operations <target_api_bf_ops>`

