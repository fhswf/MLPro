## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_pp_005_rearranger_window_bd_normminmax_nd.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-12  0.0.0     LSB      Creation
## -- 2022-12-20  0.1.0     DA       Supplements
## -- 2023-01-01  1.0.0     DA       Completion
## -- 2023-01-09  1.1.0     DA       User input of cycles and visualization step rate
## -- 2023-04-10  1.2.0     DA       Refactoring after changes on class OAScenario
## -- 2023-05-20  1.2.1     DA       Registered handler of boundary detector to window
## -- 2024-10-29  1.2.2     DA       Minor corrections
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2024-10-29)

This module is an example of adaptive normalization of streaming data using MinMax normalizer. To 
this regard, an online-adadptive custom scenario is set up. It combines a native 10-dimensional 
sample stream with an online-adaptive workflow. The latter one consists of four tasks: a rearranger
to reduce the stream data to 6 dimensions, a window that buffers the last 50 instances, a boundary
detector and finally the MinMax normalizer. 

You will learn:

1. How to set up online-adaptive custom stream scenarios.

2. How to set up online-adaptive workflows reusing various adaptive/non-adaptive MLPro stream tasks

3. How to run and visualize your own custom stream scenario.

"""

from datetime import datetime

from mlpro.bf import Log, Mode
from mlpro.bf.mt import Task
from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import *
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.tasks import RingBuffer, Rearranger
from mlpro.oa.streams import *
from mlpro.oa.streams.tasks import *

 

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyAdaptiveScenario (OAStreamScenario):

    C_NAME = 'Demo'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Prepare a native stream from MLPro
        mlpro  = StreamProviderMLPro(p_logging=p_logging)
        stream = mlpro.get_stream( p_name=StreamMLProRnd10D.C_NAME, 
                                   p_mode=p_mode, 
                                   p_visualize=p_visualize, 
                                   p_logging=p_logging )


        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='Input Signal',
                               p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                               p_ada=p_ada,
                               p_visualize=p_visualize, 
                               p_logging=p_logging )


        # 2.2 Creation of a task

        # 2.2.1 Rearranger to reduce the number of features
        features     = stream.get_feature_space().get_dims()
        features_new = [ ( 'F', features[1:7] ) ]

        task_rearranger = Rearranger( p_name='T1 - Rearranger',
                                      p_range_max=Task.C_RANGE_THREAD,
                                      p_visualize=p_visualize,
                                      p_logging=p_logging,
                                      p_features_new=features_new )

        workflow.add_task( p_task=task_rearranger )
      
        # 2.2.2 Window to buffer some data
        task_window = RingBuffer( p_buffer_size=50, 
                                  p_delay=True,
                                  p_enable_statistics=True,
                                  p_name='T2 - Ring Buffer',
                                  p_duplicate_data=True,
                                  p_visualize=p_visualize,
                                  p_logging=p_logging )

        workflow.add_task(p_task=task_window, p_pred_tasks=[task_rearranger])

        # 2.2.3 Boundary detector 
        task_bd = BoundaryDetector( p_name='T3 - Boundary Detector', 
                                    p_ada=True, 
                                    p_visualize=p_visualize,   
                                    p_logging=p_logging,
                                    p_boundary_provider = task_window )

        workflow.add_task(p_task = task_bd, p_pred_tasks=[task_window])

        # # 2.2.4 MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax( p_name='T4 - MinMax Normalizer', 
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
    cycle_limit = 60
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
                                                        p_plot_horizon = 100,
                                                        p_data_horizon = 150,
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