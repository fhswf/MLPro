## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_streams_pp_121_complex_preprocessing.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-07-27  1.0.0     LSB      Creation/Release
## -- 2024-05-02  1.1.0     DA       Review/minor adjustments
## -- 2024-05-12  1.2.0     DA       Event-based adaption Window->BoundaryDetector added
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-05-12)

This howto shows an example of complex preprocessing with parallel tasks. It addresses several
problems like feature rearanging, numerical feature derivation, data windowing, and self-adapting
realtime normalization.

You will learn:

1) How to set up an online adaptive stream workflow based on stream tasks.

2) How to set up a stream scenario based on a stream and a processing stream workflow.

3) How to combine various preprocessing tasks like a rearranger, a deriver, a sliding window, etc.

4) How to normalize stream data in realtime

5) How to add parallel running tasks

6) How to use the event-based adaptation in stream workflows

7) How a sliding window affects the adaptation of successor tasks in the workflow
        
8) How to run a stream scenario dark or with visualization.

"""


from mlpro.bf.streams.streams import *
from mlpro.bf.streams.tasks import *

from mlpro.oa.streams import *
from mlpro.oa.streams.tasks import BoundaryDetector, NormalizerMinMax



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario (OAScenario):
    """
    Example of a custom stream scenario including a stream and a stream workflow. See class 
    mlpro.oa.streams.OAScenario for further details and explanations.
    """

    C_NAME      = 'Demo Complex Preprocessing'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Import a native stream from MLPro
        provider_mlpro = StreamProviderMLPro(p_logging=p_logging)
        stream = provider_mlpro.get_stream('DoubleSpiral2D', p_mode=p_mode, p_logging=p_logging)


        # 2 Set up a stream workflow 
        workflow = OAWorkflow( p_name = 'Input Signal - "DoubleSpiral2D"', 
                               p_range_max = Task.C_RANGE_NONE, 
                               p_ada = p_ada,
                               p_visualize = p_visualize,
                               p_logging = logging )
        

        # 2.1 Set up and add a rearranger task to reduce the feature and label space
        features = stream.get_feature_space().get_dims()
        features_new = [ ( 'F', features[0:1] ) ]

        task1_rearranger = Rearranger( p_name = '1 - Rearranger',
                                       p_range_max = Task.C_RANGE_THREAD,
                                       p_visualize = p_visualize,
                                       p_logging = p_logging,
                                       p_features_new = features_new )

        workflow.add_task( p_task=task1_rearranger )


        # 2.2 Set up and add a deriver task to extend the feature and label space (1st derivative)
        features = task1_rearranger._feature_space.get_dims()
        derived_feature = features[0]

        task2_deriver1 = Deriver( p_name = '2 - Deriver #1',
                                  p_range_max = Task.C_RANGE_THREAD,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging,
                                  p_features = features,
                                  p_label = None,
                                  p_derived_feature = derived_feature,
                                  p_derived_label = None,
                                  p_order_derivative = 1 )

        workflow.add_task( p_task = task2_deriver1, 
                           p_pred_tasks = [task1_rearranger] )


        # 2.3 Set up and add a deriver task to extend the feature and label space (2nd derivative)
        features = task2_deriver1._feature_space.get_dims()
        derived_feature = features[0]
        
        task3_deriver2 = Deriver( p_name = '3 - Deriver #2',
                                  p_range_max = Task.C_RANGE_THREAD,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging,
                                  p_features = features,
                                  p_label = None,
                                  p_derived_feature = derived_feature,
                                  p_derived_label = None,
                                  p_order_derivative = 2 )

        workflow.add_task( p_task = task3_deriver2, 
                           p_pred_tasks = [task1_rearranger, task2_deriver1] )


        # 2.4 Set up and add a Sliding Window 
        task4_window = RingBuffer( p_buffer_size = 100,
                                   p_name = '4 - Ring Buffer',
                                   p_delay = True,
                                   p_visualize = p_visualize,
                                   p_enable_statistics = True,
                                   p_logging = p_logging )
        
        workflow.add_task( p_task = task4_window, 
                           p_pred_tasks = [task3_deriver2] )


        # 2.5 Setup and add a Boundary Detector
        task5_bd = BoundaryDetector( p_name = '5 - Boundary Detector', 
                                     p_ada = p_ada, 
                                     p_visualize = p_visualize,   
                                     p_logging = p_logging )

        workflow.add_task( p_task = task5_bd, 
                           p_pred_tasks = [task4_window] )

        # 2.5.1 Here the event-based adaptation mechanism of the Boundary Betector is connected to the predecessor Windoow task
        task4_window.register_event_handler( p_event_id = RingBuffer.C_EVENT_DATA_REMOVED, 
                                             p_event_handler = task5_bd.adapt_on_event )


        # 2.6 Setup Z Transform-Normalizer in Parallel
        task6_norm_ztrans = NormalizerZTransform( p_name = '6 - Normalizer Z-Trans',
                                                  p_ada = p_ada,
                                                  p_visualize = p_visualize,
                                                  p_logging = p_logging )
        
        workflow.add_task( p_task = task6_norm_ztrans, 
                           p_pred_tasks = [task4_window] )


        # 2.7 Setup MinMax-Normalizer
        task7_norm_minmax = NormalizerMinMax( p_name = '7 - Normalizer MinMax', 
                                              p_ada = p_ada, 
                                              p_visualize = p_visualize, 
                                              p_logging = p_logging )

        workflow.add_task( p_task = task7_norm_minmax, 
                           p_pred_tasks = [task5_bd] )

        # 2.7.1 Here the event-based adaptation mechanism of the MinMax-Normalizer is connected to the predecessor Boundary Detector task
        task5_bd.register_event_handler( p_event_id = BoundaryDetector.C_EVENT_ADAPTED, 
                                         p_event_handler = task7_norm_minmax.adapt_on_event )


        # 3 Return stream and workflow
        return stream, workflow




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 721
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 3
 
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 102
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 5



# 2 Instantiate the stream scenario
myscenario = MyScenario( p_mode=Mode.C_MODE_SIM,
                         p_cycle_limit=cycle_limit,
                         p_visualize=visualize,
                         p_logging=logging )



# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                        p_view_autoselect = False,
                                                        p_step_rate = step_rate ) )
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')