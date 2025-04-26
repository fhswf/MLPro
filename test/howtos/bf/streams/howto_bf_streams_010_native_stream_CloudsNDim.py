## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Module  : howto_bf_streams_010_native_stream_CloudsNDim.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- 2024-02-06  1.0.0     DA       Creation/First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-02-06)

This module demonstrates and visualizes the native stream CloudsNDim. It is the freely configurable 
n-dimensional random point cloud generator behind native streams like Clouds2D4C1000Static, etc.

You will learn:

1) The properties and use of native stream Clouds3D8C10000Dynamic.

2) How to set up a stream workflow without a stream task.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

4) How to run a stream scenario dark or with default visualization.

"""


from mlpro.bf.streams import *
from mlpro.bf.streams.streams import *
from mlpro.bf.various import Log



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario (StreamScenario):
    """
    Example of a custom stream scenario including a stream and a stream workflow. See class 
    mlpro.bf.streams.models.StreamScenario for further details and explanations.
    """

    C_NAME      = 'My stream scenario'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging):

        # 1 Import a native stream from MLPro
        stream = StreamMLProClouds( p_num_dim = 3,
                                    p_num_instances = 2000,
                                    p_num_clouds = 5,
                                    p_seed = 1,
                                    p_radii = [100, 150, 200, 250, 300],
                                    p_weights = [2,3,4,5,6],
                                    p_logging=Log.C_LOG_NOTHING )


        # 2 Set up a stream workflow
        workflow = StreamWorkflow( p_name='wf1', 
                                   p_range_max=StreamWorkflow.C_RANGE_NONE, 
                                   p_visualize=p_visualize,
                                   p_logging=logging )


        # 3 Return stream and workflow
        return stream, workflow





# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    cycle_limit = 2000
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 5
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1


# 2 Instantiate the stream scenario
myscenario = MyScenario( p_mode=Mode.C_MODE_SIM,
                         p_cycle_limit=cycle_limit,
                         p_visualize=visualize,
                         p_logging=logging )


# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                        p_view_autoselect = True,
                                                        p_step_rate = step_rate ) )
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')