## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_wr_121_sklearn_anomalydetector_lof.py
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
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase

from mlpro.oa.streams import *
from mlpro.wrappers.sklearn import OneClassSVM
from mlpro.wrappers.river.clusteranalyzers import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream4ADsvm (StreamMLProBase):

    C_ID                = 'St4DBStream'
    C_NAME              = 'Stream4DBStream'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = 12

    C_SCIREF_URL        = 'https://riverml.xyz/latest/api/cluster/DBSTREAM/'


## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        
        feature_space : MSpace = MSpace()

        for i in range(2):
            feature_space.add_dim( Feature( p_name_short = 'f' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):

        # Prepare a test dataset from https://riverml.xyz/latest/api/cluster/DBSTREAM/
        
        X = [ [1, 0.5], [1, 0.625], [1, 0.75], [1, 1.125], [1, 1.5], [1, 1.75], [4, 1.5], [4, 2.25],
             [4, 2.5], [4, 3], [4, 3.25], [4, 3.5] ]

        self._dataset   = np.array(X)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdScenario4ADsvm (OAScenario):

    C_NAME = 'AdScenario4ADsvm'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Get stream from Stream4DBStream
        stream = Stream4ADsvm( p_logging=0 )

        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAWorkflow( p_name='wf1',
                               p_range_max=OAWorkflow.C_RANGE_NONE,
                               p_ada=p_ada,
                               p_visualize=p_visualize, 
                               p_logging=p_logging )


        # 2.2 Creation of a cluster analzer task
        anomalydetector = OneClassSVM(p_kernel='rbf', p_nu=0.01)


        workflow.add_task( p_task=anomalydetector )

        # 3 Return stream and workflow
        return stream, workflow





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
    cycle_limit = 12
    step_rate   = 1

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 5
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1



# 2 Instantiate the stream scenario
myscenario = AdScenario4ADsvm( p_mode=Mode.C_MODE_REAL,
                                 p_cycle_limit=cycle_limit,
                                 p_visualize=visualize,
                                 p_logging=logging )


