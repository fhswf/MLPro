.. _target_bf_streams_handling_01:
Streams Handling
----------------

MLPro's stream module provides a stream handling and bundle of stream processing functionalities. The
stream handling architecture in MLPro is as shown in the following figure:


.. image::
    images/stream_providers.drawio.png
    :width: 650 px


The figure shows a collection of stream provider apis, which in turn, contain a list of corresponding stream objects.
Currently, MLPro supports stream provider api for :ref:`MLPro's native streams <target_native_streams_pool>` and three external data providers:

    - `OpenML <https://www.openml.org>`_
    - `River <https://riverml.xyz/>`_
    - `Scikit-Learn <https://scikit-learn.org/>`_



Stream Provider
---------------
Access to real-time data stream is not always possible for the purpose of testing and evaluations. MLPro's streams
module provides Stream Provider functionality. A stream provider in MLPro is a data resource that provides stream
objects for various operations.


MLPro's streams module provides native stream providers, that generate stream objects with user-defined parameters
such as number of features and labels and pre-defined statistical properties such as feature boundaries. Currently
MLPro's native stream provider supports random streams with random feature and label values. Along with native stream
provider MLPro also supports data resources from popular external data resources including OpenML, ScikitLearn and
River. MLPro's stream provider object accesses datasets from these resources and provide them as stream
objects that imitate the sequential behaviour.

A stream provider in MLPro can be imported by including:


.. code-block:: python

    # import mlpro native stream provider
    from mlpro.bf.streams.native import NativeStreamProvider
    # import openml stream provider
    from mlpro.wrappers.openml import WrOpenMLStreamProvider
    # import river stream provider
    from mlpro.wrappers.river import WrRiverStreamProvider
    # import scikit learn stream provider
    from mlpro.wrappers.sklearn import WrSKLearnStreamProvider


After loading the stream provider (MLPro's native stream provider for example), the list of available streams can be
loaded as following:

.. code-block:: python

    # Import the stream provider class
    from mlpro.bf.streams.native import NativeStreamProvider
    # Create an object of the stream provider
    mlpro = NativeStreamProvider()
    # Get a list of streams
    mlpro.get_stream_list()


Stream
------
In MLPro, a stream is a special iterator object that delivers new data instances with each iteration. A stream cannot be
read directly for all the instances, instead an instance is only available when requested by a workflow. An instance
in MLPro consists of feature and label data for that specific instance.

From a stream provider a specific stream of interest can be accessed with a stream id:

.. code-block:: python

    mystreamobject = mlpro.get_stream(p_id = '1')


After accessing the stream from the stream provider, a new instance can be accessed from the data stream by iterating
over it.


Stream Instance
---------------

An instance in MLPro is a data element available at each time step, when processing a stream. An instance consists of
a unique id, feature data and label data.

.. code-block:: python

    # Accessing an instance from stream
    instance = next(iter(mystreamobject))

    # Accessing the stream ID
    id = instance.get_id()

    # Accessing feature data
    feature_element = instance.get_feature_data()
    feature_data = feature_element.get_values()

    # Accessing label data
    label_element = instance.get_label_data()
    label_data = label_element.get_values()



.. note::
    - The ids of the stream instances are managed internally by a Stream Workflow, and are also used for stream plotting functionalities. Changing instance ids might affect the performance of stream functionalities of MLPro.



Stream Sampler
--------------
In MLPro, a stream has an optional component, which is a stream sampler.
A sampler is a component that selects a subset of instances from a continuous stream of data.
The purpose of a sampler is to reduce the volume of data that needs to be processed, while still providing a representative sample of the data.

Each streaming instance is going through the **omit_instance** method that is provided by a sampler.
If the output is True, then the instance is omitted and not part of the subset of instances being sampled.
Otherwise, the instance is added to the subset of instances.

A stream sampler can be attached to a stream during its instantiation or after instantiation through the public method **setup_sampler**.

.. note::
    There are several different ready-to-use samplers in the pool of objects that can be used in MLPro stream processing, including random samplers, `min-wise samplers <https://doi.org/10.1145/1031495.1031525>`_, `reservoir samplers with Algorithm R <https://doi.org/10.1145/3147.3165>`_, and more.
    Each type of sampler has its characteristics and is suitable for different types of data and processing scenarios.


**Cross References**

- :ref:`Howto BF-STREAMS-005: Streams Sampler <Howto_BF_STREAMS_005>`
- :ref:`API Reference: Streams <target_ap_bf_streams>`