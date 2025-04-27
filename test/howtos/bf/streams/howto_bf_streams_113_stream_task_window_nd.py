## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_113_stream_task_window_nd.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-11-27  1.0.0     LSB      Creation
## -- 2022-12-14  1.1.0     DA       - Changed the stream provider from OpenML to MLPro 
## --                                - Added a custom task behind the window task
## -- 2024-05-22  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2024-05-22)

This module demonstrates the functionality of stream window task in MLPro.

You will learn:

1) How to implement an own custom stream task.

2) How to set up a stream workflow based on stream tasks.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

4) How to run a stream scenario dark or with default visualization.
"""


from mlpro.bf.streams.streams import *
from mlpro.bf.streams.tasks.windows import RingBuffer



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyTask (StreamTask):
    """
    Demo implementation of a stream task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'Custom'

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict ):
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
        workflow = StreamWorkflow( p_name=self.C_NAME,
                                   p_range_max=StreamWorkflow.C_RANGE_NONE,    
                                   p_visualize=p_visualize,
                                   p_logging=logging)

        # 2.1 Set up and add a window task
        task_window = RingBuffer( p_buffer_size=50, 
                                  p_name = 'T1 - Ring Buffer', 
                                  p_delay = True,
                                  p_visualize = p_visualize, 
                                  p_enable_statistics = True,
                                  p_logging = p_logging )

        workflow.add_task(task_window)

        # 2.2 Set up and add an own custom task
        task_custom = MyTask( p_name='T2 - My Task', p_visualize=p_visualize, p_logging=logging )
        workflow.add_task( p_task=task_custom, p_pred_tasks=[task_window] )


        # 3 Return stream and workflow
        return stream, workflow




if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    cycle_limit = 200
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
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view=PlotSettings.C_VIEW_ND,
                                                        p_plot_horizon=100,
                                                        p_data_horizon=150) ) 
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')