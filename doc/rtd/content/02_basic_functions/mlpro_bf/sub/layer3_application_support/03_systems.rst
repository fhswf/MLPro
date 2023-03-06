.. _target_bf_systems:
State-based Systems
===================

MLPro aims to standardize machine learning processes to accommodate complex applications in simplified reusable APIs.
MLPro's Systems module standardize state based systems and their operation in a modular design. The keyword
**state-based** implies the possibility to characteristically represent a system's unique status vector corresponding
to a given timestep.

A state-based system has a definite condition at any given point in time, defined by a fixed number of variables that
completely defines the condition. A system transits from a state to next state at each timestep based on the inherent
state transition dynamics. However, this state transition is triggered by an external source of action, for example
an Actuator.

In real application of state-based systems, such as controlled systems, it is highly interested to maintain a desired
system state, reach a system state or maximize system output, through optimum state transitions. Additionally, it is
also a important concern to verify if the system is performing within the objective of the application or if the
system has failed.

MLPro's Systems module encapsulates the aforementioned functionalities into a standard template. The System object of
MLPro can be reused to define any custom system with default methods to handle surrounding standard operations.


.. image::
    images/systems.drawio.png
    :width: 650 px

The system's module provides following objects and templates:

    1. System: This is the base template class to implement custom System objects.

    2. FctStrans: A custom template for State Transition functions, FctStrans is used to provide custom
implementations for state transition dynamics of a system.

    3. FctSuccess: The success function, FctSuccess is a template class to implement functions to determine if the
system has reached to a goal state.

    4. FctBroken: The broken function, FctBroken is a template class to implement functions to determine if the
system has reached to a terminating broken state.

    5. State: The state object represents the current state of the system with respect to time. State object contains
information about the system corresponding all the dimensions of the State Space of the system.

    6. Action: The action object represents the current Action to be simulated on the system.

MLPro also provides the possibility to integrate real world hardware, such as controllers and hardwares to the System
object. Furthermore, Systems module integrates optional visualization and simulation functionalities from MuJoCo into
MLPro for re-usability. Check out following sections to know more:


.. toctree::
   :maxdepth: 2
   :glob:

   systems/*


- cross references: api