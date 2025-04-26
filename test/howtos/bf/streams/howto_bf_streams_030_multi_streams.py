## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Module  : howto_bf_streams_030_multi_streams.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- 2025-04-05  1.0.0     DA       Creation/First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-04-05)

This howto demonstates the use of MLPro's multi-streams that combine and orchestrate several single 
streams. In detail, it combines two random point cloud streams. The first one generates 5 static
random point clouds while the second one creates one dynamic point cloud. By varying the 
batch sizes of both streams, the plot priority and order can be changed.

You will learn:

1) How to set up a multi-stream as part of a simple stream scenario.

2) The meaning of batch sizes of single streams as part of the multi-stream.

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
    def _setup( self, 
                p_mode, 
                p_visualize:bool, 
                p_logging, 
                p_batch_size1 : int, 
                p_batch_size2 : int, 
                **p_kwargs ):

        # 1 Set up a multi stream container
        multistream = MultiStream( p_name = 'Multi-stream',
                                   p_num_instances = self._cycle_limit,
                                   p_mode = p_mode,
                                   p_logging = p_logging )


        # 1.1 Set up and add the first native stream for random point clouds
        stream1 = StreamMLProClouds( p_num_dim = 3,
                                     p_num_instances = self._cycle_limit * 0.5,
                                     p_num_clouds = 5,
                                     p_seed = 1,
                                     p_radii = [100,200,300,400,500],
                                     p_weights = [1,2,3,4,5],
                                     p_logging=p_logging )

        multistream.add_stream( p_stream = stream1, p_batch_size = p_batch_size1 )


        # 1.2 Set up and add the second native stream
        stream2 = StreamMLProClouds( p_num_dim = 3,
                                     p_num_instances = self._cycle_limit * 0.5,
                                     p_num_clouds = 1,
                                     p_seed = 2,
                                     p_radii = [200],
                                     p_velocity = 2,
                                     p_weights = [1],
                                     p_logging=p_logging )

        multistream.add_stream( p_stream = stream2, p_batch_size = p_batch_size2 )


        # 2 Set up a stream workflow
        workflow = StreamWorkflow( p_name='wf1', 
                                   p_range_max=StreamWorkflow.C_RANGE_NONE, 
                                   p_visualize=p_visualize,
                                   p_logging=logging )


        # 3 Return stream and workflow
        return multistream, workflow





# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    cycle_limit = 2000
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 5

    i = input('\n\nEnter the batch size of the first stream (ENTER = 1): ')
    if i != '':
        batch_size1 = int(i)
    else:
        batch_size1 = 1
    i = input('Enter the batch size of the second stream (ENTER = 1): ')
    if i != '':
        batch_size2 = int(i)
    else:
        batch_size2 = 1
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1
    batch_size1 = 1
    batch_size2 = 1


# 2 Instantiate the stream scenario
myscenario = MyScenario( p_mode = Mode.C_MODE_SIM,
                         p_cycle_limit = cycle_limit,
                         p_batch_size1 = batch_size1,
                         p_batch_size2 = batch_size2,
                         p_visualize = visualize,
                         p_logging = logging )


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