.. _target_streams_intro:
Streams
=======

A data stream is a live data source that delivers instances sequentially. Unlike offline datasets, data
instances cannot be scanned on demand in case of streams. Data instances are only available at the order they
arrive. For example, think of a live RADIO signal that delivers new data with time, where complete access to entire
data is not possible.

.. image::
    images/stream_processor.png
    :width: 700 px


As shown in the figure above, at every timestep, new information is available. However, the number of instances
delivered at each instance and availability of historical instances depends on the type of stream and the processing
task respectively.

In industrial scenarios, with more and more complex systems, the amount of live data delivered by the systems increases
rapidly. This high amount of live data can be leveraged to take optimal decisions for processes. This real-time data
is a data stream because of its live nature and processing such real-time data is a relevant field of data-mining
and machine learning known as Stream Processing and Online Machine Learning respectively.



After loading the stream provider (MLPro's native stream provider for example), the list of available streams can be
loaded as following:

.. code-block:: python

    # Import the stream provider class
    from mlpro.bf.streams.native import NativeStreamProvider
    # Create an object of the stream provider
    mlpro = NativeStreamProvider()
    # Get a list of streams
    mlpro.get_stream_list()



Know more about stream handling in MLPro:

.. toctree::

    streams_handling/stream_handling.rst

**Cross Reference**

+ :ref:`Howto BF-STREAMS-101: Basics of Streams <Howto BF STREAMS 101>`
+ :ref:`API Reference: Streams <target_ap_bf_streams>`
