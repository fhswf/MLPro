Time Measurement
----------------

MLPro provides an internal timing mechanism that is introduced by class property **Timer**.
This class uses the built-in python package, namely **datetime**, to deal with the time management system.
This class also has a simple lap management, in which each time the maximum number of laps ``C_LAP_LIMIT`` is reached, then the lap counter restarts to 0.
Timer class can be accessed as follows:

.. code-block:: python

    from mlpro.bf.various import Timer

The time measurement can cover two different modes, such as:
 * **C_MODE_REAL** : real time mode
 * **C_MODE_VIRTUAL** : virtual time mode


The following are the functionalities of the timer:
    * ``reset`` : to reset timer
    * ``get_time`` : to get the actual time
    * ``get_lap_time`` : to get the actual lap time
    * ``get_lap_id`` : to get an id of a actual lap
    * ``add_time`` : to add actual time, which is specifically for virtual mode
    * ``finish_lap`` : to end the current lap

**Cross Reference**
    + :ref:`API Reference <Various>`
    + :ref:`Howto BF-002: Timer <Howto BF 002>`