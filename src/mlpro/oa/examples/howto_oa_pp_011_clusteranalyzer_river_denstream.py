## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_pp_011_clusteranalyzer_river_denstream.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-05  0.0.0     SY       Creation
## -- 2023-06-05  1.0.0     SY       First version release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-06-05)

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
DenStream

"""

from mlpro.bf.streams.streams import *
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase

from mlpro.oa.streams import *
from mlpro.wrappers.river.clusteranalyzers import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream4DenStream (StreamMLProBase):

    C_ID                = 'St4DenStream'
    C_NAME              = 'Stream4DenStream'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = 24

    C_SCIREF_URL        = 'https://riverml.xyz/latest/api/cluster/DenStream/'


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

        # Prepare a test dataset from https://riverml.xyz/latest/api/cluster/DenStream/
        
        X = [ [-1, -0.5], [-1, -0.625], [-1, -0.75], [-1, -1], [-1, -1.125],
             [-1, -1.25], [-1.5, -0.5], [-1.5, -0.625], [-1.5, -0.75], [-1.5, -1],
             [-1.5, -1.125], [-1.5, -1.25], [1, 1.5], [1, 1.75], [1, 2],
             [4, 1.25], [4, 1.5], [4, 2.25], [4, 2.5], [4, 3],
             [4, 3.25], [4, 3.5], [4, 3.75], [4, 4],
             ]

        self._dataset   = np.array(X)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdScenario4DenStream (OAScenario):

    C_NAME = 'AdScenario4DenStream'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Get stream from Stream4DenStream
        stream = Stream4DenStream( p_logging=0 )

        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAWorkflow( p_name='wf1',
                               p_range_max=OAWorkflow.C_RANGE_NONE,
                               p_ada=p_ada,
                               p_visualize=p_visualize, 
                               p_logging=p_logging )


        # 2.2 Creation of a cluster analzer task
        clusterer = WrRiverDenStream2MLPro( p_decaying_factor=0.01,
                                           p_beta=0.5,
                                           p_mu=2.5,
                                           p_epsilon=0.5,
                                           p_n_samples_init=10 )

        workflow.add_task( p_task=clusterer )

        # 3 Return stream and workflow
        return stream, workflow





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
    cycle_limit = 24
    step_rate   = 1

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 5
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1



# 2 Instantiate the stream scenario
myscenario = AdScenario4DenStream( p_mode=Mode.C_MODE_REAL,
                                  p_cycle_limit=cycle_limit,
                                  p_visualize=visualize,
                                  p_logging=logging )



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



# 4 Validating the number of clusters between original algorithm and wrapper
river_n_clusters    = myscenario.get_workflow()._tasks[0].get_algorithm().n_clusters
wr_n_clusters       = len(myscenario.get_workflow()._tasks[0].get_clusters())

if river_n_clusters == wr_n_clusters:
    print("The number of clusters from river and mlpro matches!")
else:
    print("The number of clusters from river and mlpro does not match!")



if __name__ == '__main__':
    input('Press ENTER to exit...')