Stream Processing
=================

Because of the live nature of data streams, mining knowledge and handling such data sources require special types of
processing tasks. Stream processing operations process new instances as they are available at every step.Along with a
number of external and internal stream resources, MLPro's stream module provides multiple processing functionalities
like sliding window, rearranger, etc. specialized for streaming data.

.. image::
    images/stream_processing.png
    :width: 700 px
    :align: center


In MLPro streaming data is processed with a task and workflow architecture. A StreamTask is single operation
performed on new stream instances and a StreamWorkflow is a list of tasks arranged sequentially with defined
dependencies. StreamTask and StreamWorkflows are specialized classes inherited from MLPro's multiprocessing module.

Stream Task
-----------

A StreamTask is a special stream processing task that takes a new instance as an input and delivers the processed
output. StreamTask class in MLPro also provide provide plotting functionalities in 2D, 3D and nD, that plot the
streaming instances by default. (A link to know more). Inherit from this class and implement the :code:`_run(p_inst_new, p_inst_del)`
method to implement custom stream tasks with inbuilt default plotting functionalities. This can be imported and used by
including following:

.. code-block:: python

    #import stream models
    from mlpro.bf.streams.models import StreamTask

    #create a stream object
    myStreamTask = Task(p_name = 'Task 1',
                        p_visualize = True,
                        p_logging = Log.C_LOG_ALL)


Currently MLPro provides following stream task implementations:

1. Window
2. Rearranger

More StreamTask implementations will be available will later updates.

Stream Workflow
---------------

A StreamWorkflow in MLPro is a list of StreamTasks arranged hierarchically with pre-defined dependencies on prior
tasks in the workflow. StreamWorkflow also provides default plotting functionalities in 2D, 3D and nD, that plot all
the instances in the workflow. (A link to know more about plotting.) StreamWorkflow can be imported and used as
following:

.. code-block:: python

    #import stream models
    from mlpro.bf.streams.models import *

    #create a stream workflow object
    myStreamWorkflow = StreamWorkflow( p_name='My Workflow',
                                       p_range_max=StreamWorkflow.C_RANGE_NONE,
                                       p_visualize=True,
                                       p_logging=Log.C_LOG_ALL))



A stream workflow consists a list of tasks within in a defined order and instance dependency. The instances of task
processes instances processed by its predecessor task in the workflow. The code block below shows how to add a task
to an existing stream workflow:

.. code-block:: python


    # add task myStreamTask to the workflow myStreamWorkflow
    myStreamWorkflow.add_task(p_task = 'Task 1')

    #create another task
    myStreamTask2 = StreamTask(p_name = 'Task 1',
                               p_visualize = True,
                               p_logging = Log.C_LOG_ALL)

    # add the task to the workflow with task 1 as its predecessor
    myStreamWorkflow.add_task(p_task = 'Task 2')


Now, the input instances to task 2 are processed instances output from the task 1. Each workflow has shared stream
object


Stream Plotting
---------------
Below images show the stream plotting functionalities, further explanation coming soon...
.. image::
    images/stream_plot_nd.gif
    :width: 350 px


.. image::
    images/stream_plot_2d.gif
    :width: 350 px

.. image::
    images/stream_plot_3d.gif
    :width: 350 px