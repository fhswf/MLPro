## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_pp_009_parallel_tasks.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-07-27  1.0.0     LSB      Creation/Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-07-28)

This howto shows an example of howto add parallel tasks to an Online Adaptive stream workflow

You will learn:

1) How to set up an online adaptive stream workflow based on stream tasks.

2) How to set up a stream scenario based on a stream and a processing stream workflow.

3) How to add a task Deriver and how to extend the features.

4) How to normalize even derived data based on a boundary detector

5) How to add parallel task Z-trans normalizer parallel to MinMax normalizer
        
5) How to run a stream scenario dark or with visualization.

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
    mlpro.bf.streams.models.StreamScenario for further details and explanations.
    """

    C_NAME      = 'Demo Deriver'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Import a native stream from MLPro
        provider_mlpro = StreamProviderMLPro(p_logging=p_logging)
        stream = provider_mlpro.get_stream('DoubleSpiral2D', p_mode=p_mode, p_logging=p_logging)

        # 2 Set up a stream workflow 
        workflow = OAWorkflow( p_name='wf1', 
                               p_range_max=Task.C_RANGE_NONE, 
                               p_ada=p_ada,
                               p_visualize=p_visualize,
                               p_logging=logging )
        
        # 2.1 Set up and add a rearranger task to reduce the feature and label space
        features = stream.get_feature_space().get_dims()
        features_new = [ ( 'F', features[0:1] ) ]

        task_rearranger = Rearranger( p_name='t1',
                                      p_range_max=Task.C_RANGE_THREAD,
                                      p_visualize=p_visualize,
                                      p_logging=p_logging,
                                      p_features_new=features_new )

        workflow.add_task( p_task=task_rearranger )

        # 2.2 Set up and add a deriver task to extend the feature and label space (1st derivative)
        features = task_rearranger._feature_space.get_dims()
        derived_feature = features[0]

        task_deriver_1 = Deriver( p_name='t2',
                                  p_range_max=Task.C_RANGE_THREAD,
                                  p_visualize=p_visualize,
                                  p_logging=p_logging,
                                  p_features=features,
                                  p_label=None,
                                  p_derived_feature=derived_feature,
                                  p_derived_label=None,
                                  p_order_derivative=1 )

        workflow.add_task( p_task=task_deriver_1, p_pred_tasks=[task_rearranger] )

        # 2.3 Set up and add a deriver task to extend the feature and label space (2nd derivative)
        features = task_deriver_1._feature_space.get_dims()
        derived_feature = features[0]
        
        task_deriver_2 = Deriver( p_name='t3',
                                  p_range_max=Task.C_RANGE_THREAD,
                                  p_visualize=p_visualize,
                                  p_logging=p_logging,
                                  p_features=features,
                                  p_label=None,
                                  p_derived_feature=derived_feature,
                                  p_derived_label=None,
                                  p_order_derivative=2 )

        workflow.add_task( p_task=task_deriver_2, p_pred_tasks=[task_rearranger, task_deriver_1] )

        # 2.4 Set up and add a window task
        task_window = Window( p_buffer_size=30,
                              p_name = 't1',
                              p_delay = True,
                              p_visualize = p_visualize,
                              p_enable_statistics = True,
                              p_logging=p_logging)
        workflow.add_task(task_window, p_pred_tasks=[task_deriver_2])


        # 2.5 Setup Boundary detector
        task_bd = BoundaryDetector( p_name='t4', 
                                    p_ada=True, 
                                    p_visualize=True,   
                                    p_logging=p_logging )

        workflow.add_task( p_task = task_bd, p_pred_tasks=[task_window])

        # 2.6 Setup MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax( p_name='t5', 
                                             p_ada=True, 
                                             p_visualize=p_visualize, 
                                             p_logging=p_logging )

        task_bd.register_event_handler( p_event_id=BoundaryDetector.C_EVENT_ADAPTED, p_event_handler=task_norm_minmax.adapt_on_event )

        workflow.add_task(p_task = task_norm_minmax, p_pred_tasks=[task_bd])

        # 2.6 Setup Z Trasnform-Normalizer in Parallel
        task_norm_ztrans = NormalizerZTransform(p_name='Demo ZTrans Normalizer',
                                         p_ada=p_ada,
                                         p_visualize=p_visualize,
                                         p_logging=p_logging)
        workflow.add_task(p_task=task_norm_ztrans, p_pred_tasks=[task_bd])

        # 3 Return stream and workflow
        return stream, workflow




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 800
    logging     = Log.C_LOG_ALL
    visualize   = True
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False


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
                                                        p_step_rate = 2 ) )
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')