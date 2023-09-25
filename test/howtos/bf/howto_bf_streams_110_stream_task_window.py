## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_110_stream_task_window.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-11-27  1.0.0     LSB      Creation
## -- 2022-12-14  1.1.0     DA       - Changed the stream provider from OpenML to MLPro 
## --                                - Added a custom task behind the window task
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-12-14)

This module demonstrates the functionality of stream window task in MLPro.

You will learn:

1) How to implement an own custom stream task.

2) How to set up a stream workflow based on stream tasks.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

4) How to run a stream scenario dark or with default visualization.
"""


from mlpro.bf.streams.streams import *
from mlpro.bf.streams.tasks import Window



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
class MyStreamScenario(StreamScenario):

    C_NAME      = 'Demo Window'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging):

        # 1 Import a native stream from MLPro
        provider_mlpro = StreamProviderMLPro(p_logging=p_logging)
        stream = provider_mlpro.get_stream('Rnd10Dx1000', p_mode=p_mode, p_logging=p_logging)


        # 2 Set up a stream workflow 
        workflow = StreamWorkflow( p_name='wf-window',
                                   p_range_max=StreamWorkflow.C_RANGE_NONE,    
                                   p_visualize=p_visualize,
                                   p_logging=logging)

        # 2.1 Set up and add a window task
        task_window = Window( p_buffer_size=30, 
                              p_name = 't1', 
                              p_delay = True,
                              p_visualize = p_visualize, 
                              p_enable_statistics = True )
        workflow.add_task(task_window)

        # 2.2 Set up and add an own custom task
        task_custom = MyTask( p_name='t2', p_visualize=p_visualize, p_logging=logging )
        workflow.add_task( p_task=task_custom, p_pred_tasks=[task_window] )


        # 3 Return stream and workflow
        return stream, workflow




if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    cycle_limit = 100
    logging = Log.C_LOG_ALL
    visualize = True

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging = Log.C_LOG_NOTHING
    visualize = False
 

# 2 Instantiate the stream scenario
myscenario = MyStreamScenario(p_mode=Mode.C_MODE_REAL,
    p_cycle_limit=cycle_limit,
    p_visualize=visualize,
    p_logging=logging)


# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot()
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')