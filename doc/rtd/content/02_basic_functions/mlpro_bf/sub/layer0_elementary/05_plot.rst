.. _DataPlotting:
Plotting and Visualization
==========================

In order to be able to visualize processes on different levels in a standardized way, MLPro provides a property class 
**Plottable**. This inherits to higher classes (custom) methods for initializing and updating plot output during 
execution. There are three different views:

- 2-dimensional plot output
- 3-dimensional plot output
- n-dimensional plot output

Additional parameters can be set using the **PlotSettings** class.

MLPro and in particular the Plottable class is intended for plot outputs with the standard package `Matplotlib <https://matplotlib.org/>`_
in connection with the output backend `TkAgg <https://matplotlib.org/stable/api/backend_tk_api.html#module-matplotlib.backends.backend_tkagg>`_. 
In this combination, a good user experience is made possible. In principle, however, other packages can also be used for visualization.


**Cross Reference**

- :ref:`API Reference BF-PLOT - Plotting and Visualization <target_api_bf_plot>`
- :ref:`Stream Plotting <target_bf_streams_processing_01>`
