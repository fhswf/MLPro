## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Module  : howto_bf_streams_030_multi_streams.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- 2025-04-05  1.0.0     DA       Creation/First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-04-05)

...

You will learn:

1) The properties and use of native stream Clouds3D8C10000Dynamic.

2) How to set up a stream workflow without a stream task.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

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
    def _setup(self, p_mode, p_visualize:bool, p_logging, **p_kwargs):

        # 1 Set up a multi stream container
        multistream = MultiStream( p_name = 'Multi-stream',
                                   p_num_instances = self._cycle_limit,
                                   p_mode = p_mode,
                                   p_logging = p_logging )


        # 1.1 Set up and add the first native stream for random point clouds
        stream1 = StreamMLProClouds( p_num_dim = 3,
                                     p_num_instances = self._cycle_limit,
                                     p_num_clouds = 5,
                                     p_seed = 1,
                                     p_radii = [100, 150, 200, 250, 300],
                                     p_weights = [2,3,4,5,6],
                                     p_logging=p_logging )

        multistream.add_stream( p_stream = stream1, p_batch_size = 1 )


        # 1.2 Set up and add the second native stream
        stream2 = StreamMLProPOutliers( p_num_dim = 3,
                                        p_num_instances = self._cycle_limit,
                                        p_functions = [ 'sin', 'cos', 'const' ],
                                        p_outlier_rate = 0.05,
                                        p_seed = 1,                                    
                                        p_logging = p_logging )

        multistream.add_stream( p_stream = stream2, p_batch_size = 1 )


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