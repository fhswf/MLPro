## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_103_stream_task_window.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-11-27  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-11-07)

This module demonstrates the functionality of stream window task in MLPro.

You will learn:

1) How to implement an own custom stream task.

2) How to set up a stream workflow based on stream tasks.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

4) How to run a stream scenario dark or with default visualization.
"""

from  mlpro.bf.streams.tasks.windows import *
from mlpro.wrappers.openml import *

class MyStreamScenario(StreamScenario):

    def _setup(self, p_mode, p_visualize:bool, p_logging):
        openml = WrStreamProviderOpenML(p_logging=p_logging)
        stream  = openml.get_stream(p_id=75, p_mode=p_mode, p_logging=p_logging)

        workflow = StreamWorkflow( p_name='wf-window',
                                   p_range_max=StreamWorkflow.C_RANGE_NONE,    #StreamWorkflow.C_RANGE_THREAD,
                                   p_visualize=p_visualize,
                                   p_logging=logging)

        window_task = Window(p_buffer_size=30, p_name = 'Window-1', p_visualize = p_visualize, p_enable_statistics=True)

        workflow.add_task(window_task)

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