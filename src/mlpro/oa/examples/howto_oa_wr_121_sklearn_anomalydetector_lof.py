## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_wr_121_sklearn_anomalydetector_oneclasssvm.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-23  0.0.0     SY       Creation
## -- 2023-05-23  1.0.0     SY       First version release
## -- 2023-05-25  1.0.1     SY       Refactoring related to ClusterCentroid
## -- 2023-06-05  1.0.2     SY       Renaming module
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-06-05)

This module demonstrates the principles of stream processing with MLPro. To this regard, a stream of
a stream provider is combined with a stream workflow to a stream scenario. The workflow consists of 
a standard task 'Cluster Analyzer'. The stream scenario is used to process some instances. Moreover,
we reuse a number of cluster analyzer algorithms from river package, which will also be demonstrated
in this howto file.

You will learn:

1) How to set up a stream workflow based on stream tasks.

2) How to set up a stream scenario based on a stream and a processing stream workflow.

3) How to add a task ClusterAnalyzer.

4) How to reuse a cluster analyzer algorithm from river (https://www.riverml.xyz/), specifically
DBSTREAM

"""

from mlpro.bf.streams.streams import *
from mlpro.bf.streams.models import *
from mlpro.bf.various import Log
from mlpro.wrappers.openml import WrStreamProviderOpenML
from mlpro.oa.streams import *
from mlpro.wrappers.sklearn import LocalOutlierFactor




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdScenario4ADlof (OAScenario):

    C_NAME = 'AdScenario4ADlof'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        openml = WrStreamProviderOpenML(p_logging = logging)

        # 3 Get stream from the stream provider OpenML
        mystream = openml.get_stream( p_id='42397', p_name='CreditCardFraudDetection', p_logging=logging)

        # 2 Set up a stream workflow based on a custom stream task
        #print(mystream._dataset)

        # 2.1 Creation of a workflow
        workflow = OAWorkflow( p_name='wf1',
                               p_range_max=OAWorkflow.C_RANGE_NONE,
                               p_ada=p_ada,
                               p_visualize=p_visualize, 
                               p_logging=p_logging )


        # Initiailise the lof anomaly detctor class
        anomalydetector = LocalOutlierFactor(p_neighbours = 1, p_visualize=p_visualize)

        # Add anomaly detection task to workflow
        workflow.add_task( p_task=anomalydetector )

        # 3 Return stream and workflow
        return mystream, workflow





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
    cycle_limit = 1000
    step_rate   = 1

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 5
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1





# 2 Instantiate the stream scenario
myscenario = AdScenario4ADlof( p_mode=Mode.C_MODE_REAL,
                                 p_cycle_limit=cycle_limit,
                                 p_visualize=visualize,
                                 p_logging=logging )


# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot()
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')