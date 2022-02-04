`Double Pendulum <https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/pool/envs/doublependulum.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. image:: images/double_pendulum.gif
    :width: 600
    
By default the lengths and weights of the pendulum are set to be 0.5 meters each and 0.5 kg each.
The user can customize this parameter and many other parameter to better suit the research
purpose. The other customizable parameter includes the starting pendulum positions and speeds, 
maximum torque and speed of the motor, the action frequency, and the time step. In addition, for 
visualization purpose, the history lengths can also be modified to a higher value to add more 
of the orange traces as shown on the figrue above. The environment is not episodical, which means
that the cycle limit should be defined manually to fit some training algorihtms. 

The double pendulum environment can be imported via:

.. code-block:: python

    import mlpro.rl.pool.envs.doublependulum
    
Prerequisites
=============

    - `NumPy <https://pypi.org/project/numpy/>`_
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
    - `SciPy <https://pypi.org/project/scipy/>`_


General Information
===================

+------------------------------------+-------------------------------------------------------+
|         Parameter                  |                         Value                         |
+====================================+=======================================================+
| Agents                             | 1                                                     |
+------------------------------------+-------------------------------------------------------+
| Native Source                      | MLPro                                                 |
+------------------------------------+-------------------------------------------------------+
| Action Space Dimension             | [1,]                                                  |
+------------------------------------+-------------------------------------------------------+
| Action Space Base Set              | Real number                                           |
+------------------------------------+-------------------------------------------------------+
| Action Space Boundaries            | Depends on max_torque                                 |
+------------------------------------+-------------------------------------------------------+
| State Space Dimension              | [4,]                                                  |
+------------------------------------+-------------------------------------------------------+
| State Space Base Set               | Real number                                           |
+------------------------------------+-------------------------------------------------------+
| State Space Boundaries             | Pi for position and None for speed                    |
+------------------------------------+-------------------------------------------------------+
| Reward Structure                   | Overall reward                                        |
+------------------------------------+-------------------------------------------------------+
 
Action Space
============

The continuous action is interpreted as a torque applied to the pendulum for a given time step. 
Depending on the max_speed parameter, this might not affect the system due to the pendulum
moving faster than the motor can handle.

State Space
===========

The state space of the system is a continuous space in the order of:
    - Position of Pendulum 1
    - Speed of Pendulum 1
    - Position of Pendulum 2
    - Speed of Pendulum 2
    
The position of the pendulum is guaranteed to be within -pi and pi, however the speed is not 
limited within a boundary due to the effects of gravitational acceleration.

  
Reward Structure
================

.. code-block:: python
    
    reward = Reward(Reward.C_TYPE_OVERALL)

    state = p_state_new.get_values()

    count = 0
    for th1 in self.y[:, 0]:
        if np.degrees(th1) > 179 or np.degrees(th1) < 181 or \
                np.degrees(th1) < -179 or np.degrees(th1) > -181:
            count += 1

    speed_costs = np.pi * abs(state[1]) / self.max_speed
    reward.set_overall_reward((abs(state[0]) - speed_costs) * count / len(self.y))
    
The reward calculation only takes into account the position of the first pendulum and
the speed of the new state. This is formulated with the purpose of giving high reward
whenever the pendulum stays upright while also minding the speed of the pendulum.     


Change Log
==========
    
+--------------------+---------------------------------------------+
| Version            | Changes                                     |
+====================+=============================================+
| 1.0.0              | First public version                        |
+--------------------+---------------------------------------------+
  
Cross Reference
===============
    + :ref:`API Reference <Double Pendulum>`