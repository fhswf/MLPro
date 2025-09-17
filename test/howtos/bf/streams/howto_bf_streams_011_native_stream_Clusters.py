## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Module  : howto_bf_streams_011_native_stream_Clusters.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- 2025-08-19  1.0.0     DA       Creation/First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-08-19)


"""


from mlpro.bf import Log, Mode, PlotSettings
from mlpro.bf.streams import *
from mlpro.bf.streams.streams import StreamMLProClusterGenerator



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario (StreamScenario):
    """
    Example of a custom stream scenario including a stream and a stream workflow. See class 
    mlpro.bf.streams.models.StreamScenario for further details and explanations.
    """

    C_NAME      = 'My stream scenario'

## -------------------------------------------------------------------------------------------------
    def _setup( self, p_mode, p_visualize, p_logging, **p_kwargs):

        # 1 Import a native stream from MLPro
        stream = StreamMLProClusterGenerator( p_num_dim = 3,
                                              p_num_instances = 2000,
                                              p_num_clusters = 5,
                                              p_feature_boundaries = [ [-100,100], [100,200], [-20,-10] ],
                                              p_outlier_appearance = True,
                                              p_outlier_rate = 0.02,
                                              p_radii = [100, 150, 200, 250, 300],
                                              p_velocities = [0.5],
                                              p_distribution_bias = [1],
                                              p_change_radii  = False,
                                              p_rate_of_change_of_radius = 0.001,
                                              p_points_of_change_radii = None,
                                              p_num_clusters_for_change_radii = None,
                                              p_change_velocities = False,
                                              p_points_of_change_velocities = None,
                                              p_num_clusters_for_change_velocities = None,
                                              p_changed_velocities = None,
                                              p_change_distribution_bias = False,
                                              p_points_of_change_distribution_bias = None,
                                              p_num_clusters_for_change_distribution_bias = None,
                                              p_appearance_of_clusters = False,
                                              p_points_of_appearance_of_clusters = None,
                                              p_num_new_clusters_to_appear = None,
                                              p_disappearance_of_clusters = False,
                                              p_points_of_disappearance_of_clusters = None,
                                              p_num_clusters_to_disappear = None,
                                              p_clusters_split_and_merge = False,
                                              p_clusters_split = False,
                                              p_points_of_split = None,
                                              p_velocities_after_split = None,
                                              p_num_clusters_to_split_into = 2,
                                              p_max_clusters_affected = 0.75,
                                              p_seed = 1,
                                              p_logging = Log.C_LOG_NOTHING )


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
                                                        p_data_horizon = cycle_limit,
                                                        p_step_rate = step_rate ) )
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')