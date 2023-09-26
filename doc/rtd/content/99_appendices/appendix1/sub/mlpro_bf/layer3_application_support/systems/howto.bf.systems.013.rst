.. _Howto BF SYSTEMS 013:
Howto BF-SYSTEMS-013: MuJoCo Simulation with Camera
===================================================

**Prerequisites**

Please install the following packages to run this examples properly:
    
    - `MuJoCo <https://pypi.org/project/mujoco/>`_
    - `lxml <https://pypi.org/project/lxml/>`_
    - `glfw <https://pypi.org/project/glfw/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../../test/howtos/bf/howto_bf_systems_013_box_on_table_mujoco_simulation.py
	:language: python



**Results**

The MuJoCo window appears and shows the simulation of a random box on the table. A Matplotlib window appears also and shows
the current image data from the camera.

.. image:: images/mujoco_boxontable.gif



**Cross Reference**

    - :ref:`API Reference: Systems <target_ap_bf_systems>`