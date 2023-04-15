.. _Howto BF SYSTEMS 004:
Howto BF-SYSTEMS-004: MuJoCo Simulation with Camera
====================================================================

Ver. 1.0.1 (2023-04-12)

This module demonstrates the principles of using classes System and uses MuJoCo wrapper to simulate 
the pre-defined model. A camera is integrated in the simulation model. The camera is extracted from
the simulation and shown with Matplotlib.

You will learn:
    
1) How to set up a custom System for MuJoCo with custom reset function.

2) How to include the MuJoCo model in the System.

3) How to visualize image data from the simulation with Matplotlib.

**Prerequisites**


    Please install the following packages to run this examples properly:
        + `MuJoCo <https://pypi.org/project/mujoco/>`_
        + `lxml <https://pypi.org/project/lxml/>`_
        + `glfw <https://pypi.org/project/glfw/>`_


**Executable code**

.. literalinclude:: ../../../../../../../../../src/mlpro/bf/examples/howto_bf_systems_004_box_on_table_mujoco_simulation.py
	:language: python



**Results**

The MuJoCo window appears and shows the simulation of a random box on the table. A Matplotlib window appears also and shows
the current image data from the camera.

.. image:: images/mujoco_boxontable.gif


**Cross Reference**

+ :ref:`API Reference: Systems <target_ap_bf_systems>`