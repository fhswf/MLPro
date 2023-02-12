## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples.howto_oa_002_normalization_of_streamed_data_minmax
## -- Module  : howto_oa_pp_001_normalization_of_streamed_data_minmax.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-07  0.0.0     LSB      Creation
## -- 2022-12-09  1.0.0     LSB      Release
## -- 2022-12-13  1.0.1     LSB      Refctoring
## -- 2022-12-31  1.0.2     LSB      Using native stream
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-12-31)
This module is an example of adaptive normalization of streaming data using MinMax Normalizer

You will learn:

1. Creating tasks and workflows in MLPro-OA.

2. Registering Event handlers for events and tasks.

3. Normalizing streaming data using MinMax Normalizer, with boundary detector as a predecessor task.

"""

from mlpro.oa.tasks.normalizers import *
from mlpro.oa.tasks.boundarydetectors import *
from mlpro.bf.streams.models import *
from mlpro.wrappers.openml import *






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyAdaptiveScenario(OAScenario):

    C_NAME = 'Dummy'


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging):
        # 1 Import a stream from OpenML
        mlpro = StreamProviderMLPro(p_logging=p_logging)
        stream = mlpro.get_stream(p_name=StreamMLProRnd10D.C_NAME,
            p_mode=p_mode,
            p_visualize=p_visualize,
            p_logging=p_logging)
        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a task
        TaskBoundaryDetector = BoundaryDetector(p_name='Demo Boundary Detector', p_ada=True, p_visualize=True,
            p_logging=p_logging)
        TaskNormalizerMinMax = NormalizerMinMax(p_name='Demo MinMax Normalizer', p_ada=True, p_visualize=True,
            p_logging=p_logging)

        # 2.2 Creation of a workflow
        workflow = OAWorkflow(p_name='wf1',
            p_range_max=OAWorkflow.C_RANGE_NONE,  # StreamWorkflow.C_RANGE_THREAD,
            p_visualize=p_visualize, 
            p_logging=p_logging)

        # 2.3 Addition of the task to the workflow
        workflow.add_task(p_task = TaskBoundaryDetector)
        workflow.add_task(p_task = TaskNormalizerMinMax)


        # 3 Registering event handlers for normalizer on events raised by boundaries
        TaskBoundaryDetector.register_event_handler(BoundaryDetector.C_EVENT_ADAPTED, TaskNormalizerMinMax.adapt_on_event)


        # 3 Return stream and workflow
        return stream, workflow


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