.. _target_bf_systems_04:
PID Controller
==============

A **PID controller** (Proportional-Integral-Derivative controller) is one of the most widely used control algorithms in automation and control engineering. It provides a robust and flexible way to regulate a system’s behavior by adjusting its output based on the error between the desired setpoint and the measured process variable.

Structure of a PID Controller
-----------------------------

The PID controller consists of three main components:

1. **Proportional (P):**
   Reacts to the present error.
   - Output is proportional to the error signal.
   - Formula: ``u_P(t) = K_P * e(t)``
   - **Effect:** Reduces the error, but does not always eliminate it (steady-state error may remain).

2. **Integral (I):**
   Reacts to the accumulated error over time.
   - Eliminates steady-state error.
   - Formula: ``u_I(t) = K_I * ∫ e(τ) dτ``
   - **Effect:** Improves accuracy but may introduce slower response or instability if too aggressive.

3. **Derivative (D):**
   Reacts to the rate of change of the error.
   - Predicts system behavior and helps dampen oscillations.
   - Formula: ``u_D(t) = K_D * de(t)/dt``
   - **Effect:** Improves system stability and reduces overshoot but is sensitive to noise.

The **control output** is the sum of these three components:

.. math::

   u(t) = K_P \cdot e(t) + K_I \cdot \int_0^t e(\tau) \, d\tau + K_D \cdot \frac{de(t)}{dt}

Transfer Function of a PID Controller
-------------------------------------

In the Laplace domain, the PID controller’s transfer function is:

.. math::

   G_{PID}(s) = K_P + \frac{K_I}{s} + K_D \cdot s

- :math:`K_P:` Proportional gain  
- :math:`K_I:` Integral gain  
- :math:`K_D:` Derivative gain  
- :math:`s:` Complex frequency variable from the Laplace transform  

Tuning a PID Controller
-----------------------

To achieve the desired system behavior, the PID parameters :math:`K_P, K_I,K_D` need to be tuned. Common tuning methods include:

1. **Manual Tuning:**

   - Adjust :math:`K_P, K_I,K_D` based on system response (trial-and-error).

2. **Ziegler-Nichols Method:**

   - Set :math:`K_I = 0` and :math:`K_D = 0`, increase :math:`K_P` until the system oscillates critical gain (:math:`K_{crit}`) 

   - Calculate parameters using standard formulas.

3. **Software-Based Tuning:**

   - Use simulation tools or optimization algorithms to find optimal gains.

Effects of PID Components
-------------------------

+--------------+----------------------------------------+---------------------------------------+
| Component    | Effect on System Behavior              | Typical Trade-Offs                    | 
+--------------+----------------------------------------+---------------------------------------+
| Proportional | Reduces rise time, reduces error       | May leave steady-state error          |
+--------------+----------------------------------------+---------------------------------------+
| Integral     | Eliminates steady-state error          | Slower response, possible instability |
+--------------+----------------------------------------+---------------------------------------+
| Derivative   | Improves stability, reduces overshoot  | Sensitive to noise                    |
|              |                                        |                                       |
|              |                                        |                                       |
|              |                                        |                                       |
|              |                                        |                                       |
|              |                                        |                                       |
|              |                                        |                                       |
+--------------+----------------------------------------+---------------------------------------+


Applications of PID Controllers
-------------------------------

PID controllers are versatile and are used in various industries, including:

- **Process Control:** Temperature, pressure, and flow regulation.
- **Motion Control:** Robotics, servos, and positioning systems.
- **Power Systems:** Voltage and frequency control.
- **Automotive:** Cruise control, engine speed regulation.

Advantages and Limitations
---------------------------

**Advantages:**
- Simple and effective.
- Can handle a wide range of systems with proper tuning.

**Limitations:**
- Sensitive to parameter tuning.
- Derivative term can amplify noise if not filtered.
- May not work well for highly nonlinear or time-varying systems.



**PID Controller can be imported and used as following:**

.. code-block:: python

    #import PID controller  model
    from mlpro.bf.control.controllers.pid_controller import PIDController

    #create a PID controller object
    my_ctrl = PIDController( p_input_space = my_ctrl_sys.get_state_space(),
                         p_output_space = my_ctrl_sys.get_action_space(),
                         p_Kp = 1.5,
                         p_Tn = 1.4,
                         p_Tv = 0,
                         p_integral_off = False,
                         p_derivitave_off = True,
                         p_name = 'PID Controller',
                         p_visualize = visualize,
                         p_logging = logging )



**Cross Reference**


- :ref:`Howto BF-CONTROL-101: PID-Controller with PT1 system <Howto_BF_CONTROL_101>`

- :ref:`API References <target_api_bf_control_controllers_pid_controller>`

- `Further information <https://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=ControlPID>`_



