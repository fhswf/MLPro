## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_pp_003_rearranger_window_bd_normminmax.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-12  0.0.0     LSB      Creation
## -- 2022-12-20  0.1.0     DA       Supplements
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2022-12-20)

This module is an example of adaptive normalization of streaming data using MinMax Normalizer

You will learn:

1. ...

2.

3.

"""


from mlpro.bf.streams import *
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.tasks import Window, Rearranger
from mlpro.oa import *
from mlpro.oa.tasks import BoundaryDetector, NormalizerMinMax




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyAdaptiveScenario (StreamScenario):

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
        workflow = OAWorkflow( p_name='wf1',
                               p_range_max=OAWorkflow.C_RANGE_NONE,
                               p_ada=True,
                               p_visualize=p_visualize, 
                               p_logging=p_logging )


        # 2.2 Creation of a task

        # 2.2.1 Rearranger to reduce the number of features
        features     = stream.get_feature_space().get_dims()
        features_new = [ ( 'F', features[1:3] ) ]

        task_rearranger = Rearranger( p_name='t1',
                                      p_range_max=Task.C_RANGE_THREAD,
                                      p_visualize=p_visualize,
                                      p_logging=p_logging,
                                      p_features_new=features_new )

        workflow.add_task( p_task=task_rearranger )
      
        # 2.2.2 Window to buffer some data
        task_window = Window( p_buffer_size=50, 
                              p_delay=True,
                              p_enable_statistics=True,
                              p_name='t2',
                              p_duplicate_data=True,
                              p_visualize=p_visualize,
                              p_logging=p_logging )

        workflow.add_task(p_task=task_window, p_pred_tasks=[task_rearranger])

        # 2.2.3 Boundary detector 
        task_bd = BoundaryDetector( p_name='t3', 
                                    p_ada=True, 
                                    p_visualize=True,   #not yet implemented
                                    p_logging=p_logging,
                                    p_window=task_window )

        workflow.add_task(p_task = task_bd, p_pred_tasks=[task_window])

        # 2.2.4 MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax( p_name='t4', 
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
    cycle_limit = 100
    logging = Log.C_LOG_ALL
    visualize = True

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging = Log.C_LOG_NOTHING
    visualize = False


# 2 Instantiate the stream scenario
myscenario = MyAdaptiveScenario(p_mode=Mode.C_MODE_REAL,
    p_cycle_limit=cycle_limit,
    p_visualize=visualize,
    p_logging=logging)




# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot()
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')