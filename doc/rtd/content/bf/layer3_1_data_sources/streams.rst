Streams
=======

A data stream is a live data source that delivers instances sequentially. Unlike offline datasets, data
instances cannot be scanned on demand in case of data streams. Data instances are only available at the order they
arrive. For example, a data stream can be thought of as a live signal from RADIO sensors, where new data instances
are available with time, therefore complete data is not available directly.

In industrial scenarios, with more and more complex systems the amount of live data delivered by the systems increases
rapidly. This high amount of live data can be leveraged to take optimal decisions for processes. This real-time data
is a data stream because of its live nature and processing such real-time data is a relevant field of data-mining
and machine learning known as Stream Processing and Online Machine Learning respectively.

MLPro's stream module provides a stream handling and bundle of stream processing functionalities. The
stream handling architecture in MLPro is as seen in the following figure:




Streams Handling
____________________

Stream
------
A stream is a special iterator object in MLPro that delivers new data instances with each iteration. A stream cannot be
read directly for all the instances, instead an instance is only available when requested by a workflow. An instance
in MLPro consists of feature and label data for that specific instance.

Stream Provider
---------------
Access to real-time data stream is not always possible for the purpose of testing and evaluations. MLPro's stream
module provides Stream Provider functionality. A stream provider in MLPro is a data resource that provides stream
objects for various operations.

MLPro's streams module provides native stream providers, that generate stream objects with user-defined parameters
such as number of features and labels and pre-defined statistical properties such as feature boundaries. Currently
MLPro's native stream provider supports random streams with random feature and label values. Along with native stream
provider MLPro also supports data resources from popular external data resources including OpenML, ScikitLearn and
River. MLPro's stream provider module accesses popular datasets from these resources and provide them as stream
objects that imitate the sequential behaviour.

A stream provider in MLPro can be imported by including:

code.......

After loading the stream provider, the list of available streams can be loaded as following:

code....

From a stream provider a specific stream of interest can be accessed with a stream id:

code....

After accessing the stream from the stream provider, a new instance can be accessed from the data stream by iterating
over it.


Stream Processing
_________________

Because of the live nature of data streams, mining knowledge and handling such data sources require special types of
processing tasks. Stream processing operations process new instances as they are available at every step.Along with a
number of external and internal stream resources, MLPro's stream module provides multiple processing functionalities
like sliding window, rearranger, etc. specialized for streaming data.

In MLPro streaming data is processed with a task and workflow architecture. A StreamTask is single operation
performed on new stream instances and a StreamWorkflow is a list of tasks arranged sequentially with defined
dependencies. StreamTask and StreamWorkflows are specialized classes inherited from MLPro's multiprocessing module.

Stream Task
-----------

A StreamTask is a special stream processing task that takes a new instance as an input and delivers the processed
output. StreamTask class in MLPro also provide provide plotting functionalities in 2D, 3D and nD, that plot the
streaming instances by default. (A link to know more). Inherit from this class and implement the :code:`_run(p_inst_new, p_inst_del)`
method to implement custom stream tasks with inbuilt default plotting functionalities. This can be imported by
including following:


Currently MLPro provides following stream task implementations:

1. Window
2. Rearranger

More StreamTask implementations will be available will later updates.

Stream Workflow
---------------

A StreamWorkflow in MLPro is a list of StreamTasks arranged hierarchically with pre-defined dependencies on prior
tasks in the workflow. StreamWorkflow also provides default plotting functionalities in 2D, 3D and nD, that plot all
the instances in the workflow. (A link to know more about plotting.) StreamWorkflow can be imported as following

