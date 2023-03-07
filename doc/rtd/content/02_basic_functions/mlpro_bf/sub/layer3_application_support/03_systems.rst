.. _target_bf_systems:
State-based Systems
===================

MLPro aims to standardize machine learning processes to accommodate complex applications in simplified reusable APIs.
MLPro's Systems module standardizes state based systems and their operation in a modular design. The keyword
**state-based** implies the possibility to characteristically represent a system's unique status as a vector
corresponding to a given timestep.

A state-based system has a definite condition at any given point in time, defined by a fixed number of variables that
completely defines the condition. A system transits from a state to next state at each timestep based on the inherent
state transition dynamics. However, this state transition is triggered by an external source of action, for example
an Actuator.

In real application of state-based systems, such as controlled systems, it is highly interesting to maintain a desired
system state, reach a system state or maximize system output, through optimum state transitions. Additionally, it is
also an important concern to verify if the system is performing within the objective of the application or if the
system has failed.


.. image::
    images/systems.drawio.png
    :width: 550 px

As shown in the figure above, MLPro's Systems module encapsulates the aforementioned functionalities into a standard
template. The System object of MLPro can be reused to define any custom system with default methods to handle surrounding standard operations.

The system's module provides following objects and templates:

    1. **System**:

    The System class standardizes and provides the base template for any State-based System along with standard MLPro
functionalities such as Logging, Timer, Cycle Management, Persistence, Real/Simulated mode and Reset. The System
class additionally provides room for custom functionalities such as Reaction Simulation, Terminal State Monitoring
such as Success and Broken. These custom functionalities can be incorporated by implementing the
:code:`_simulate_reaction()`, :code:`_compute_success()`, :code:`_compute_broken()` methods on Systems class or
corresponding function classes (described below), which are then passed as a parameter to the system.

.. note::
    The System class of also supports operation in modes: Real and Simulated, based on which it enables working with
a real hardware or a simulated system respectively.

    2. **FctStrans**:

    The FctStrans (State Transition Function) standardizes the process of simulating the primary State Transition
process of a System. The :code:`simulate_reaction(p_state, p_action)` method of this class takes the current state of
the environment and the action from the corresponding actuator as a parameter, and maps it to the next state of the
system, based on the inherent dynamics.

.. note::
    Please implement the :code:`_simulate_reaction()` method of FctStrans, in order to re-use in a custom
implementation.

    3. **FctSuccess**:

    A System state can be monitored through FctSuccess (Success Function) to determine if the system has reached the
expected objective state/output. It maps the current state of the system to a boolean value indicating the success of
a system.

.. note::
    Please implement :code:`_compute_success()` method of FctSuccess, in order to re-use it in a custom implementation.


    4. **FctBroken**:

    Similar to FctSuccess class, the FctBroken class standardizes the process of monitoring whether the system has
reached a broken terminal state, by mapping the current state to a boolean value indicating the broken state.

.. note::
    Please implement :code:`_compute_broken()` method of FctBroken, in order to re-use it in custom implementation.

    5. **State**:

    The state object represents the current state of the system with respect to time. A state object inherits from
the Element class of MLPro, which represents an element in a Multi-dimensional Set object, a State-Space in this case.
The state consists information about the System for corresponding dimension of the related State-Space.

    6. **Action**:

    The Action object standardizes external input to the system. For example, input from a controller, input from an
actuator or an agent in case of Reinforcement Learning. The standard Action object consists of an ActionElement or a
list of ActionElements, in case of more than one action sources. The action element is similar to a state object,
consisting corresponding values for all the dimension in the related action-space.

MLPro also provides the possibility to integrate real world hardware, such as controllers and hardware to the System
object. Furthermore, Systems module integrates optional visualization and simulation functionalities from MuJoCo into
MLPro for re-usability. Check out following sections to know more:


.. toctree::
   :maxdepth: 2
   :glob:

   systems/*


**Cross References**

- :ref:`BF-SYSTEMS - State-based Systems <target_ap_bf_systems>`
- :ref:`Howto BF-SYSTEMS-001: System, Controller, Actuator, Sensor <Howto BF SYSTEMS 001>`
- :ref:`Howto BF-SYSTEMS-002: Systems wrapped with MuJoCo <Howto BF SYSTEMS 002>`
