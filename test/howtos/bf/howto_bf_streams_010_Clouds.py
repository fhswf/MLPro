## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_010_Clouds.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-28  0.0.0     SP       Creation
## -- 2023-09-28  1.0.0     SP       First draft implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-09-28)

This module demonstrates how to use the class StreamMLProClouds.

You will learn:

1) How to access MLPro's native random clouds data stream.

2) How to iterate the instances of the stream.


"""

# Import the necessary libraries
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.streams.clouds import StreamMLProClouds
from mlpro.bf.streams.tasks import Window
from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario (StreamScenario):
    """
    Implementation of a custom stream scenario including a custom StreamMLProClouds stream 
    and a stream workflow. See class mlpro.bf.streams.models.StreamScenario for further details and
    explanations.
    
    """

    C_NAME      = ''

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging):

        # 1 Initialise the stream object using the StreamMLProClouds class
        stream = StreamMLProClouds( p_num_dim=2,
                                    p_num_instances=500,
                                    p_num_clouds=5,
                                    p_radii=[70, 90, 100, 60, 100],
                                    p_behaviour='static',
                                    p_logging=logging )

        # 2 Set up the stream workflow

        # 2.1 Set up and add a window task and an empty task
        task_window = Window( p_buffer_size=50, 
                              p_name = 't1', 
                              p_delay = True,
                              p_visualize = p_visualize, 
                              p_enable_statistics = True,
                              p_logging=p_logging )

        # 2.2 Create a workflow and add the tasks
        workflow = StreamWorkflow( p_name='wf1', 
                                   p_range_max=StreamWorkflow.C_RANGE_NONE, 
                                   p_visualize=p_visualize,
                                   p_logging=p_logging )
        
        # 2.2.1 Add the tasks to our workflow
        workflow.add_task( p_task=task_window )

        # 3 Return stream and workflow
        return stream, workflow
    


# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = False
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False


# 2 Instantiate the stream scenario
myscenario = MyScenario( p_mode=Mode.C_MODE_SIM,
                         p_visualize=visualize,
                         p_logging=logging )


# 3 Reset and run own stream scenario
myscenario.reset()


if __name__ == '__main__':
    myscenario.init_plot()
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')


