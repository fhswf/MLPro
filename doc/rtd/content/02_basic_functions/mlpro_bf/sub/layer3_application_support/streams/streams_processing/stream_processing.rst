.. _target_bf_streams_processing_01:
Stream Task
-----------

A StreamTask is a special stream processing task that takes a new instance as an input and delivers the processed
instances as an output. A StreamTask also processes the obsolete/deleted instances from the workflow for following
tasks.

StreamTask class in MLPro also provide provide plotting functionalities in 2D, 3D and nD, that plot the
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

A StreamWorkflow in MLPro is a list of StreamTasks arranged hierarchically with user-defined dependencies on prior
tasks in the workflow. A stream workflow receives new instance of the stream from the surrounding StreamScenario
object at every step.

.. note::
    A stream workflow carries a list of new instances and deleted/obsolete instances at every run. Both new and deleted
    instances are forwarded to subsequent tasks to be processed.


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



A stream workflow consists a list of tasks within in a defined order and instance dependency. The instances processed
by a task are forwarded to it's following task. The code block below shows how to add a task to an existing stream
workflow:

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


Stream Scenario
---------------

A stream scenario in MLPro inherits from MLPro's scenario base class. The idea of a scenario in MLPro is to have all
the elements together, required for a specific application, whether it is a training application or just a sample run.
A scenario set's up the process parameters and runs the process for a given number of cycles as defined in the
specific scenario implementation.

A stream scenario consists of two main elements:
        - A stream object
        - A streamtask workflow

.. note::
    To plug these elements into the StreamScenario class, please implement the :code:`_setup(p_mode, p_visualize,
    p_logging)` method of the same


A StreamScenario class takes care of the following tasks in a Stream processing application:
    1. Fetching new instance at every step
    2. Running the plugged in StreamWorkflow
    3. Managing and updating the visualization windows
    4. Storing the results of the workflow



**Cross Reference**

- :ref:`Stream <target_streams_intro>`

- :ref:`How To to be included`

- :ref:`API References`




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
