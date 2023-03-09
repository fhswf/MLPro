.. _target_bf_streams_tasks_deriver:
Deriver
----------

The deriver task in MLPro provides a functionality to derive a selected feature and extend the feature with its derivative.
In mathematics, the derivative is a concept that measures the rate of change of a function at a particular point.
It is represented as the slope of the tangent line to a function at that point.
The derivative is a fundamental tool in calculus and is used to study the properties of functions, such as their maxima, minima, and inflection points.
It is also used in various scientific and engineering fields to study the behavior of systems that change over time, such as in the modeling of physical processes and in the analysis of dynamic systems.
The derivative can be calculated using limits or through a number of differentiation rules for common functions, such as power, exponential, and logarithmic functions.

In the current implementation, we set up the basic formula of the derivation, as follows:

:math:`f′(x) = limΔx→0 (f(x+Δx) − f(x)) /Δx`

.. note::
    The order of derivative can be selected through :code:`p_order_derivative`.
    If you would like to have two orders of derivative, then you have to add two separate tasks to the workflow.


**Cross Reference**

- :ref:`Howto BF-STREAMS-114: Deriver <Howto BF STREAMS 114>`
- :ref:`API Reference: Streams <target_ap_bf_streams>`
