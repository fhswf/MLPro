## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_pp_007_rearranger_window_bd_normminmax_2d_3d_nD_multithreading.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-09  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-01-09)

This module is an example of adaptive normalization of streaming data using MinMax normalizer. To 
this regard, an online-adadptive custom scenario is set up. It combines a native 10-dimensional 
sample stream with an online-adaptive workflow. The latter one consists of four tasks: a rearranger
to reduce the stream data to 3 dimensions, a window that buffers the last 50 instances, a boundary
detector and finally the MinMax normalizer. 

You will learn:

1. How to set up online-adaptive custom stream scenarios.

2. How to set up online-adaptive workflows reusing various adaptive/non-adaptive MLPro stream tasks

3. How to run and visualize your own custom stream scenario.
"""



from mlpro.bf.streams import *
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.tasks import Window, Rearranger
from mlpro.oa.streams import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyAdaptiveScenario (OAScenario):

    C_NAME = 'Dummy'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging):

        # 1 Prepare a native stream from MLPro
        mlpro  = StreamProviderMLPro(p_logging=p_logging)
        stream = mlpro.get_stream( p_name=StreamMLProRnd10D.C_NAME, 
                                   p_mode=p_mode, 
                                   p_visualize=p_visualize, 
                                   p_logging=p_logging )


        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAWorkflow( p_name='wf',
                               p_range_max=OAWorkflow.C_RANGE_THREAD,
                               p_ada=True,
                               p_visualize=p_visualize, 
                               p_logging=p_logging )


        # 2.2 Creation of a task chain 2D

        # 2.2.1 Rearranger to reduce the number of features
        features     = stream.get_feature_space().get_dims()
        features_new = [ ( 'F', features[1:3] ) ]

        task_rearranger = Rearranger( p_name='Chain 2D, Task T1',
                                      p_duplicate_data=True,
                                      p_range_max=Task.C_RANGE_THREAD,
                                      p_visualize=p_visualize,
                                      p_logging=p_logging,
                                      p_features_new=features_new )

        workflow.add_task( p_task=task_rearranger )
      
        # 2.2.2 Window to buffer some data
        task_window = Window( p_buffer_size=50, 
                              p_delay=True,
                              p_enable_statistics=True,
                              p_name='Chain 2D, Task T2',
                              p_duplicate_data=True,
                              p_visualize=p_visualize,
                              p_logging=p_logging )

        workflow.add_task(p_task=task_window, p_pred_tasks=[task_rearranger])

        # 2.2.3 Boundary detector 
        task_bd = BoundaryDetector( p_name='Chain 2D, Task T3', 
                                    p_ada=True, 
                                    p_visualize=True,   
                                    p_logging=p_logging,
                                    p_window=task_window )

        workflow.add_task(p_task = task_bd, p_pred_tasks=[task_window])

        # # 2.2.4 MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax( p_name='Chain 2D, Task T4', 
                                             p_ada=True, 
                                             p_visualize=p_visualize, 
                                             p_logging=p_logging )

        task_bd.register_event_handler( p_event_id=BoundaryDetector.C_EVENT_ADAPTED, p_event_handler=task_norm_minmax.adapt_on_event )

        workflow.add_task(p_task = task_norm_minmax, p_pred_tasks=[task_bd])


        # 2.3 Creation of a task chain 3D

        # 2.3.1 Rearranger to reduce the number of features
        features     = stream.get_feature_space().get_dims()
        features_new = [ ( 'F', features[1:4] ) ]

        task_rearranger = Rearranger( p_name='Chain 3D, Task T1',
                                      p_range_max=Task.C_RANGE_THREAD,
                                      p_duplicate_data=True,
                                      p_visualize=p_visualize,
                                      p_logging=p_logging,
                                      p_features_new=features_new )

        workflow.add_task( p_task=task_rearranger )
      
        # 2.3.2 Window to buffer some data
        task_window = Window( p_buffer_size=50, 
                              p_delay=True,
                              p_enable_statistics=True,
                              p_name='Chain 3D, Task T2',
                              p_duplicate_data=True,
                              p_visualize=p_visualize,
                              p_logging=p_logging )

        workflow.add_task(p_task=task_window, p_pred_tasks=[task_rearranger])

        # 2.3.3 Boundary detector 
        task_bd = BoundaryDetector( p_name='Chain 3D, Task T3', 
                                    p_ada=True, 
                                    p_visualize=True,   
                                    p_logging=p_logging,
                                    p_window=task_window )

        workflow.add_task(p_task = task_bd, p_pred_tasks=[task_window])

        # # 2.3.4 MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax( p_name='Chain 3D, Task T4', 
                                             p_ada=True, 
                                             p_visualize=p_visualize, 
                                             p_logging=p_logging )

        task_bd.register_event_handler( p_event_id=BoundaryDetector.C_EVENT_ADAPTED, p_event_handler=task_norm_minmax.adapt_on_event )

        workflow.add_task(p_task = task_norm_minmax, p_pred_tasks=[task_bd])



        # 2.5 Creation of a task chain nD

        # 2.5.1 Rearranger to reduce the number of features
        features     = stream.get_feature_space().get_dims()
        features_new = [ ( 'F', features[1:5] ) ]

        task_rearranger = Rearranger( p_name='Chain nD, Task T1',
                                      p_range_max=Task.C_RANGE_THREAD,
                                      p_duplicate_data=True,
                                      p_visualize=p_visualize,
                                      p_logging=p_logging,
                                      p_features_new=features_new )

        workflow.add_task( p_task=task_rearranger )
      
        # 2.5.2 Window to buffer some data
        task_window = Window( p_buffer_size=50, 
                              p_delay=True,
                              p_enable_statistics=True,
                              p_name='Chain nD, Task T2',
                              p_duplicate_data=True,
                              p_visualize=p_visualize,
                              p_logging=p_logging )

        workflow.add_task(p_task=task_window, p_pred_tasks=[task_rearranger])

        # 2.5.3 Boundary detector 
        task_bd = BoundaryDetector( p_name='Chain nD, Task T3', 
                                    p_ada=True, 
                                    p_visualize=True,   
                                    p_logging=p_logging,
                                    p_window=task_window )

        workflow.add_task(p_task = task_bd, p_pred_tasks=[task_window])

        # # 2.5.4 MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax( p_name='Chain nD, Task T4', 
                                             p_ada=True, 
                                             p_visualize=p_visualize, 
                                             p_logging=p_logging )

        task_bd.register_event_handler( p_event_id=BoundaryDetector.C_EVENT_ADAPTED, p_event_handler=task_norm_minmax.adapt_on_event )

        workflow.add_task(p_task = task_norm_minmax, p_pred_tasks=[task_bd])


        # 3 Return stream and workflow
        return stream, workflow




# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True

    try:
        cycle_limit = min(1000, max(1, int(input('\nPlease enter number of cycles (1 - 1000, default = 200): '))))
    except:
        cycle_limit = 200

    try:
        step_rate   = max(1, int(input('\nPlease enter update step rate for visualization (1 = update after every cycle): ')))
    except:
        step_rate = 1

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1


# 2 Instantiate the stream scenario
myscenario = MyAdaptiveScenario(p_mode=Mode.C_MODE_REAL,
    p_cycle_limit=cycle_limit,
    p_visualize=visualize,
    p_logging=logging)




# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                        p_step_rate = step_rate ) )
    input('\nPlease arrange all windows and press ENTER to start stream processing...')

tp_before = datetime.now()
myscenario.run()
tp_after = datetime.now()
tp_delta = tp_after - tp_before
duraction_sec = ( tp_delta.seconds * 1000000 + tp_delta.microseconds + 1 ) / 1000000
myscenario.log(Log.C_LOG_TYPE_S, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(cycle_limit/duraction_sec,2))

if __name__ == '__main__':
    input('Press ENTER to exit...')