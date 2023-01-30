.. _target_scenario_RL:
Scenarios
------------

A scenario is where the interaction between RL agents and an environment with unique and specific settings takes place.
A class **RLScenario** inherits the functionality from class **Scenario** in the basic function level, where the **RLScenario** class combines RL agents and an environment into an executable unit.

One of the MLPro's features is enabling the user to apply a template class for an RL scenario consisting of an environment and agents.
Moreover, you can create either a single-agent scenario or a multi-agent scenario in a simple manner.
Here is the commented class diagram for creating a custom scenario in MLPro-RL:

.. image:: images/MLPro-RL-Scenario_Class_Commented.png

For setting-up a scenarion for single-agent and multi-agent in MLPro-RL, you can refer to :ref:`this page<target_training_RL>`.