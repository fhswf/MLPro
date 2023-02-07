## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_114_stream_task_deriver.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-02  0.0.0     SY       Creation
## -- 2023-02-05  1.0.0     SY       First version release
## -- 2023-02-07  1.1.0     SY       Change the dataset to doublespiral2d
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2023-02-07)

This module demonstrates the principles of stream processing with MLPro. To this regard, a stream of
a stream provider is combined with a stream workflow to a stream scenario. The workflow consists of 
a standard task Deriver and a custom task. The stream scenario is used to process some instances.

You will learn:

1) How to implement an own custom stream task.

2) How to set up a stream workflow based on stream tasks.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

4) How to add a task Deriver and how to extend the features.
        
5) How to run a stream scenario dark or with visualization.

"""


from mlpro.bf.streams import *
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.tasks import Rearranger, Deriver



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyTask (StreamTask):
    """
    Demo implementation of a stream task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'Custom'

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario (StreamScenario):
    """
    Example of a custom stream scenario including a stream and a stream workflow. See class 
    mlpro.bf.streams.models.StreamScenario for further details and explanations.
    """

    C_NAME      = 'Demo Deriver'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging):

        # 1 Import a native stream from MLPro
        provider_mlpro = StreamProviderMLPro(p_logging=p_logging)
        stream = provider_mlpro.get_stream('DoubleSpiral2D', p_mode=p_mode, p_logging=p_logging)

        # 2 Set up a stream workflow 
        workflow = StreamWorkflow( p_name='wf1', 
                                   p_range_max=Task.C_RANGE_NONE, 
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

        # 3 Return stream and workflow
        return stream, workflow




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 200
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
                                                        p_step_rate = 2 ) )
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')