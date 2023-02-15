.. _target_bf_streams_tasks_window:
Window
======

In streaming scenarios, data is available sequentially, and the amount of data received is directly proportional to
the time for which the stream is active. In practice, this data accumulates in tremendous amounts as the application
becomes complex. Processing data with minimum use of storage is important. A window task stores a small amount of
data from the incoming stream, that can be used to process subsequent tasks based on a smaller amount of data that
represents the stream behaviour.


.. image::
    images/window.png
    :width: 800 px


The window task in MLPro, stores the most recent instances received from the stream in the buffer. The buffer size
of the window is fixed and defined by the user. As soon as the buffer is full, the oldest instance is deleted form
the buffer to add the latest instance to the buffer. The subsequent tasks in the workflow with dependency on window,
have access to the data in the buffer.

.. note::
    The availability of the buffered instances to the subsequent tasks can be delayed by setting the :code:`p_delay` parameter to :code:`True`. In this case, the buffered instances are only available once the buffer is completely full.


The window task of MLPro, also provides functionality to get statistical information about the buffered instances,
such as Boundaries, Mean, Variance and Standard Deviation of the features of the instances in the buffer.
Additionally, MLPro also provides visualization functionality for window, as shown below.



**Cross Reference**

- :ref:`Streams <target_streams_intro>`
- Tutorial: :ref:`Howto BF STREAMS 110 <Howto BF STREAMS 110>`

