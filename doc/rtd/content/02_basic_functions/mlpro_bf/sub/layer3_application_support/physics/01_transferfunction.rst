.. _target_basics_physics_tf:

Transfer functions
==================

A transfer function is a mathematical representation of the relationship between the input and output of a time-invariant system.
It is commonly used in control theory and electrical engineering to analyze and design systems with inputs and outputs, but it is not restricted only to those aspects.
The transfer function provides valuable functionality to process inputs to outputs within a period of time.
It can be used to design controllers that regulate the behaviour of the system, predict its response to inputs, and analyze the performance of the system in the frequency domain.
Transfer functions play an important role in the design and analysis of control systems, communication systems, and signal processing systems.

.. image:: images/bf_physics_transferlearning.drawio.png
    :width: 500

In MLPro, there are three possibilities for transfer functions, which are:

    1. Linear function

    2. Custom function

    3. Function approximation (future work)

Transfer Function class can be accessed as follows:

.. code-block:: python

    from mlpro.bf.physics import TransferFunction


**Cross reference**
    + :ref:`Howto BF-PHYSICS-001: Transfer functions <Howto BF PHYSICS 001>`
    + :ref:`Howto BF-PHYSICS-002: Unit converter <Howto BF PHYSICS 002>`
    + :ref:`API reference <target_ap_bf_physics_basics>`