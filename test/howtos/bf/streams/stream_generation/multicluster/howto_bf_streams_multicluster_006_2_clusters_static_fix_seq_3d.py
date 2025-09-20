## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Module  : howto_bf_streams_multiclusters_011_2_clusters_static_fix_seq_2d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-09-19  1.0.0     DA       Creation/First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-09-19)

This module demonstrates ...

"""

from mlpro.bf.ops import Mode
from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import *
from mlpro.bf.streams.streams.multiclusters import *
from mlpro.bf.various import Log



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario (StreamScenario):
    """
    Example of a custom stream scenario including a stream and a stream workflow. See class 
    mlpro.bf.streams.models.StreamScenario for further details and explanations.
    """

    C_NAME      = '2 Clusters rnd, static'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging):

        # 1 Set up MLPro's cluster generator
        stream1 = StreamCluster( p_num_dim = 3, 
                                 p_seed = 1,
                                 p_num_instances = self._cycle_limit/2,
                                 p_states = [ ClusterState( p_center = [-500, -500, -500], p_radii = [200, 100, 50] ) ] )
        
        stream2 = StreamCluster( p_num_dim = 3, 
                                 p_seed = 2,
                                 p_num_instances = self._cycle_limit/2,
                                 p_states = [ ClusterState( p_center = [ 500, 500, 500], p_radii = [100, 200, 150] ) ] )

        mstream = MultiStream()
        mstream.add_stream( p_stream = stream1, p_batch_size=1, p_start_instance=0 )
        mstream.add_stream( p_stream = stream2, p_batch_size=2, p_start_instance=self._cycle_limit / 3 )


        # 2 Set up a stream workflow
        workflow = StreamWorkflow( p_name='wf1', 
                                   p_range_max=StreamWorkflow.C_RANGE_NONE, 
                                   p_visualize=p_visualize,
                                   p_logging=logging )


        # 3 Return stream and workflow
        return mstream, workflow





# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    cycle_limit = 1000
    logging     = Log.C_LOG_WE
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

    