Streams
=======

Stream
------

A data stream is a live data source that delivers instances sequentially. Unlike offline datasets, data
instances cannot be scanned on demand in case of data streams. Data instances are only available at the order they
arrive. For example, a data stream can be thought of a as a live signal from RADIO sensors, where new data instances
are available with time, therefore complete data is not available directly.

A stream consists of a unique data instance X_t at time step t. A data instance is a tuple features (f_1, f_2, ... ,
f_n), defining the stream properties at the timestep.


Stream Provider
---------------



