## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_bf_streams_010_point_outliers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-02-06  1.0.0     DA      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-02-06)

This module demonstrates MLPro's benchmark stream set for point outliers. It shows various baselines
with random outliers. 

You will learn:

1. How to reuse the benchmark stream for point outliers.

2. How to set up stream workflows without tasks

3. How to run and visualize your own custom stream scenario.

"""


from mlpro.bf.streams import *
from mlpro.bf.streams.streams import point_outliers

 

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyStreamScenario (StreamScenario):

    C_NAME = 'Dummy'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging):

        # 1 Prepare a native stream from MLPro
        stream = StreamMLProPOutliers( p_functions = ['sin', 'cos', 'const'],
                                       p_outlier_frequency = 25,
                                       p_visualize=p_visualize, 
                                       p_logging=p_logging )


        # 2 Set up a stream workflow without a task
        workflow = StreamWorkflow( p_name='wf1', 
                                   p_range_max=StreamWorkflow.C_RANGE_NONE, 
                                   p_visualize=p_visualize,
                                   p_logging=logging )


        # 3 Return stream and workflow
        return stream, workflow




# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True

    try:
        cycle_limit = min(1000, max(1, int(input('\nPlease enter number of cycles (1 - 1000, default = 360): '))))
    except:
        cycle_limit = 360

    try:
        step_rate   = max(1, int(input('\nPlease enter update step rate for visualization (1 = update after every cycle): ')))
    except:
        step_rate = 1

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging = Log.C_LOG_NOTHING
    visualize = False


# 2 Instantiate the stream scenario
myscenario = MyStreamScenario( p_mode=Mode.C_MODE_REAL,
                               p_cycle_limit=cycle_limit,
                               p_visualize=visualize,
                               p_logging=logging )




# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                        p_view_autoselect = False,
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