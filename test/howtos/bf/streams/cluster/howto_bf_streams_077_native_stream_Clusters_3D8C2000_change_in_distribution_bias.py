## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Module  : howto_bf_streams_077_native_stream_Clusters_3D8C2000_change_in_distribution_bias.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-17  1.0.0     SK       Creation/First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-06-17)

This module demonstrates and visualizes the native stream Clusters which generates a
specified number of n-dimensional instances placed around specified number of centers, resulting in
clouds or clusters whose numbers, size, velocity, acceleration and density can be varied over time.

You will learn:

1) The properties and use of native stream Clusters.

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

    C_NAME      = 'Clusters3D8C2000'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging):

        # 1 Import a native stream from MLPro
        stream = StreamMLProClusterGenerator(p_num_dim= 3,
                                             p_num_instances= 2000,
                                             p_num_clusters= 8,
                                             p_radii= [100],
                                             p_distribution_bias= [1, 2, 1, 1],
                                             p_change_distribution_bias= True,
                                             p_points_of_change_distribution_bias= [400, 200, 250, 300],
                                             p_num_clusters_for_change_distribution_bias= 4,
                                             p_visualize= p_visualize,
                                             p_seed= 2,
                                             p_logging= p_logging)


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
    step_rate   = 2
  
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
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_3D,
                                                        p_view_autoselect = False,
                                                        p_step_rate = step_rate ) )
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')

    