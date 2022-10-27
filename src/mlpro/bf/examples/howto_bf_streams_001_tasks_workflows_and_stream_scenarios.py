## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_001_tasks_workflows_and_stream_scenarios.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-27  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-10-27)

This module demonstrates the principles of stream processing with MLPro. To this regard, stream tasks
are added to a stream workflow. This in turn is combined with a stream of a stream provider to a
a stream scenario. The latter one can be executed.


-> serial processing vs multithreading

You will learn:

1) How to implement an own custom stream task.

2) How to set up a stream workflow based on stream tasks.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

4) How to run a stream scenario dark or with default visualization.

"""


from mlpro.bf.streams import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyStreamTask (StreamTask):
    """
    Demo implementation of a stream task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'My stream task'


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        raise NotImplementedError






# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
  
else:
    # 1.2 Parameters for internal unit test
    logging     = Log.C_LOG_NOTHING
    visualize   = False


# 2 Set up a stream workflow based on a custom stream task

# 2.1 Creation of 9 tasks
# t1a = MyTask( p_duration=duration, p_name='t1a', p_logging=logging )
# t1b = MyTask( p_duration=duration, p_name='t1b', p_logging=logging )
# t1c = MyTask( p_duration=duration, p_name='t1c', p_logging=logging )

# t2a = MyTask( p_duration=duration, p_name='t2a', p_logging=logging )
# t2b = MyTask( p_duration=duration, p_name='t2b', p_logging=logging )
# t2c = MyTask( p_duration=duration, p_name='t2c', p_logging=logging )

# t3a = MyTask( p_duration=duration, p_name='t3a', p_logging=logging )
# t3b = MyTask( p_duration=duration, p_name='t3b', p_logging=logging )
# t3c = MyTask( p_duration=duration, p_name='t3c', p_logging=logging )

# # 2.2 Create a workflow and add the tasks
# wf = mt.Workflow( p_name='wf1', 
#                   p_range_max=mt.Workflow.C_RANGE_THREAD, 
#                   p_class_shared=mt.Shared, 
#                   p_logging=logging )

# # 2.2.1 At first we add three tasks that build the starting points of our workflow
# wf.add_task( p_task=t1a )
# wf.add_task( p_task=t1b )
# wf.add_task( p_task=t1c )

# # 2.2.2 Then, we add three further tasks that shall start when their predecessor tasks have finished
# wf.add_task( p_task=t2a, p_pred_tasks=[t1a] )
# wf.add_task( p_task=t2b, p_pred_tasks=[t1b] )
# wf.add_task( p_task=t2c, p_pred_tasks=[t1c] )

# # 2.2.3 Finally, we add three further tasks that build the end of our task chains
# wf.add_task( p_task=t3a, p_pred_tasks=[t2a, t2b, t2c] )
# wf.add_task( p_task=t3b, p_pred_tasks=[t2a, t2b, t2c] )
# wf.add_task( p_task=t3c, p_pred_tasks=[t2a, t2b, t2c] )


# 3 Prepare a stream from the OpenML project


# 4 Set up a stream scenario based on a stream and a stream workflow


# 5 Run the stream scenario
# 5.1 Synchronous execution with visualization
# 5.2 Asynchronous execution with visualization
                    
