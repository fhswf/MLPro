Streams
=======

A data stream is a live data source that delivers instances sequentially. Unlike offline datasets, data
instances cannot be scanned on demand in case of data streams. Data instances are only available at the order they
arrive. For example, a data stream can be thought of a as a live signal from RADIO sensors, where new data instances
are available with time, therefore complete data is not available directly.

In industrial scenarios, with more and more complex systems the amount of live data delivered by the systems increases
rapidly. This high amount of live data can be leveraged to take optimal decisions for processes. This real-time data
is a data stream because of  its live nature and processing such real-time data is a relevant field of data-mining
and machine learning known as Stream Processing and Online Machine Learning respectively.

MLPro's stream module provides a bundle of stream processing functionalities and stream handling technology. The
stream handling architecture in MLPro is as seen in the following figure:




Streams Architecture
____________________

Stream
------
A stream is a special iterator object in MLPro that delivers new data instances with each new call to the iterator
object. A stream cannot be read directly for all the instances, instead an instance is only available when requested
by a workflow. An instance in MLPro consists of features and labels, which consists the values of features and labels
respectively for that specific instance.

Stream Provider
---------------
Access to real-time data stream is not always possible for the purpose of testing and evaluations. MLPro's stream
module provides Stream Provider functionality. A stream provider in MLPro is a data resource that provides stream
objects for stream operations. A stream provider provides api to access stream objects with additional stream
specific settings.

MLPro's streams module provides native stream providers, that generate stream objects with user-defined stream
parameters such as number of features and labels and pre-defined statistical properties such as feature boundaries.
Currently MLPro's native stream provider supports random streams with random feature and label values. Along with
native stream provider MLPro also supports data resources from popular external data resources including OpenML,
ScikitLearn and River. MLPro's stream provider module accesses popular datasets from these resources and provide them
as stream objects that imitate the sequential behaviour.

