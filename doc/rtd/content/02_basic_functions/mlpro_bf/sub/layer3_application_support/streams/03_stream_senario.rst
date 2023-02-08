.. _target_stream_scenario:
Stream Scenario
===============

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