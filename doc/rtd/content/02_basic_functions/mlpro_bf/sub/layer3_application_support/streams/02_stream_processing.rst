Stream Processing
=================
Handling streaming data sources and mining knowledge from them requires special types of processing tasks because of
their live behaviour. Stream operations process new instances as they are available at every step. Along with a
number of external and internal stream resources, MLPro's stream module provides processing functionalities
like sliding window, rearranger, etc. specialized for streaming data.

.. image::
    images/stream_processing.png
    :width: 700 px
    :align: center


In MLPro, streaming data is processed with a task and workflow architecture. A StreamTask is single operation
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

    1. :ref:`Window <target_bf_streams_tasks_window>`
    2. :ref:`Rearranger <target_bf_streams_tasks_rearranger>`
    3. :ref:`Deriver <target_bf_streams_tasks_deriver>`

More StreamTask implementations will be available with future updates.

Stream Workflow
---------------

A StreamWorkflow in MLPro is a list of StreamTasks arranged hierarchically with pre-defined dependencies on prior
tasks in the workflow. A stream workflow receives new instance of the stream from the surrounding StreamScenario
object at every step.

.. note::
    A stream workflow stores a list of new instances and deleted instances at every run. Both new and deleted
    instances are forwarded to subsequent instances to be processed.


A stream workflow takes care of following functionalities:
    1. Executing the tasks inside the workflow
    2. Storing task specific results in the StreamShared Object
    3. Fetching and delivering new and deleted instances among different tasks as per the defined dependency


**StreamWorkflow can be imported and used as following:**

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
    myStreamWorkflow.add_task(p_task = 'Task 2', p_predecessor = 'Task 1')


Each workflow has a shared object that stores instances and results of the stream task that can be accessed from
other tasks in the workflow. StreamWorkflow also provides default plotting functionalities in 2D, 3D and nD, that plot all
the instances in the workflow. Know more about MLPro's plotting functionalities.

Stream Plotting
---------------
MLPro's streams module also provide plotting functionalities by default. The stream workflow and stream tasks can
plot instances within the workflow and the task respectively. The default plotting functionality is available in 2
dimensional, 3 dimensional and N dimensional views. The plot view and specific plot properties can be set using a
PlotSetting object. Below images show an example of the default plotting functionality in ND, 2D, 3D, respectively, in
MLPro's streams module.

.. image::
    images/streams_plot_nd.gif
    :width: 350 px

.. image::
    images/streams_plot_2d.gif
    :width: 350 px

.. image::
    images/streams_plot_3d.gif
    :width: 350 px