## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_101_stream_workflow_scenario1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-27  0.0.0     DA       Creation
## -- 2022-11-11  1.0.0     DA       First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-11-11)

This module demonstrates the principles of stream processing with MLPro. To this regard, a stream of
a stream provider is combined with a stream workflow and just one simple stream task within to a
stream scenario. The latter one is used to process some instances.

You will learn:

1) How to implement an own custom stream task.

2) How to set up a stream workflow based on stream tasks.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

4) How to run a stream scenario dark or with default visualization.

"""


from mlpro.bf.streams import *
from mlpro.wrappers.openml import WrStreamProviderOpenML



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyTask (StreamTask):
    """
    Demo implementation of a stream task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'Custom'

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario (StreamScenario):
    """
    Example of a custom stream scenario including a stream and a stream workflow. See class 
    mlpro.bf.streams.models.StreamScenario for further details and explanations.
    """

    C_NAME      = 'Demo #1'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging):

        # 1 Import a stream from OpenML
        openml  = WrStreamProviderOpenML(p_logging=p_logging)
        stream  = openml.get_stream(p_id=75, p_mode=p_mode, p_logging=p_logging)


        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a task
        task = MyTask( p_name='t1', p_visualize=p_visualize, p_logging=logging )

        # 2.2 Creation of a workflow
        workflow = StreamWorkflow( p_name='wf1', 
                                   p_range_max=StreamWorkflow.C_RANGE_NONE,    #StreamWorkflow.C_RANGE_THREAD, 
                                   p_visualize=p_visualize,
                                   p_logging=logging )

        # 2.3 Addition of the task to the workflow
        workflow.add_task( p_task=task )


        # 3 Return stream and workflow
        return stream, workflow




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 10
    logging     = Log.C_LOG_ALL
    visualize   = True
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False


# 2 Instantiate the stream scenario
myscenario = MyScenario( p_mode=Mode.C_MODE_SIM,
                         p_cycle_limit=cycle_limit,
                         p_visualize=visualize,
                         p_logging=logging )


# 3 Reset and run own stream scenario
myscenario.reset()
myscenario.run()

if __name__ == '__main__':
    input('Press key to exit...')