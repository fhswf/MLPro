.. _target_bf_control:
Closed-loop control
===================

Closed-loop control, also known as feedback control, is a fundamental principle in classical control engineering. In this approach, 
the controller continuously measures the actual output of a system and compares it to a defined reference or setpoint. The resulting 
error signal is used to adjust the control input in real time to reduce deviations and ensure that the system output follows the desired 
trajectory.

This feedback mechanism makes closed-loop control robust against disturbances and modeling inaccuracies, as it automatically compensates 
for unexpected changes in the system or its environment. Typical examples include proportional-integral-derivative (PID) controllers, 
which are widely applied due to their simplicity and effectiveness.

Closed-loop control is essential in automation technology, electrical engineering, and mechanical engineering whenever precise and stable 
regulation of dynamic systems is required — such as in temperature control, motor speed regulation, and industrial process automation.

The MLPro-BF-Control sub-framework implements the standard process model of the control loop and provides common operators and templates for building 
custom controllers. It inherits the basic technology of the :ref:`BF-Streams <target_bf_streams>` sub-framework and transfers its structural elements 
to the world of control engineering. Thus, an individual control loop can be constructed as a **control workflow** consisting of various **control tasks** 
(controllers, operators, controlled system, sub-control loops, etc.). Stream tasks can also be integrated, which, for example, perform data 
preprocessing or data analysis. A **control system** is ultimately a derivative of the stream scenario, which embeds the control workflow and handles 
process control and time management. The **controlled system**, with its latency, sets the pace in the control system. Accordingly, different timing 
patterns can occur in **cascaded control loops**.

The visualization from the :ref:`BF-Streams <target_bf_streams>` sub-framework is taken over here, enabling real-time monitoring of control applications.
The functionalities from the sub-framework :ref:`BF-Systems <target_bf_systems>` can be used to create controlled systems.

[**Image: Basic elements of a closed-loop control system in MLPro-BF-Control**]

**Learn more**

.. toctree::
   :maxdepth: 2
   :glob:

   control/*


**Cross Reference**

- `Control theory (Wikipedia) <https://en.wikipedia.org/wiki/Control_theory>`_
- :ref:`Howtos BF-Control <target_howto_bf_control>`
- :ref:`API Reference BF-Control <target_api_bf_control>`
- :ref:`API Reference BF-Control Pool Objects <target_pool_bf_control>`
- :ref:`BF-Systems - Basics of state-based systems <target_bf_systems>`
- :ref:`BF-Streams - Basics of stream processing <target_bf_streams>`
