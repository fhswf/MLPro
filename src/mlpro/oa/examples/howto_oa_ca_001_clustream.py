## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_ca_001_clustream.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-08-23  0.0.0     SY       Creation
## -- 2023-08-XX  1.0.0     SY       First version release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-08-XX)

This module demonstrates the combination of several tasks in a workflow, which includes:

1) Boundary Detector,

2) Min/Max-Normalizer, and

3) Wrapped CluStream Algorithm (River).

Two data stream are incorporated in this module, such as static 3D point clouds and dynamic 3D point
clouds. This module is prepared for the MLPro-OA scientific paper and going to be stored as Code
Ocean Capsule, thus the result is reproducible.

"""


from mlpro.bf.streams.streams import *
from mlpro.bf.streams.streams.clouds3d_static import StreamMLProStaticClouds3D
from mlpro.bf.streams.streams.clouds3d_dynamic import StreamMLProDynamicClouds3D
from mlpro.bf.various import Log

from mlpro.oa.streams import *
from mlpro.wrappers.river.clusteranalyzers import *





# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
    visualize   = True
else:
    logging     = Log.C_LOG_NOTHING
    visualize   = False





# 1 Prepare a scenario for Dynamic 3D Point Clouds
class Dynamic3DScenario (OAScenario):

    C_NAME = 'AdScenario4DBStream'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1.1 Get stream from StreamMLProDynamicClouds3D
        stream = StreamMLProDynamicClouds3D(p_pattern = 'random', 
                                            p_no_clouds = 8,
                                            p_variance = 7.0, 
                                            p_velocity = 1.0, 
                                            p_logging = logging)
        stream.set_random_seed(3)

        # 1.2 Set up a stream workflow based on a custom stream task

        # 1.2.1 Creation of a workflow
        workflow = OAWorkflow( p_name='wf_3D',
                               p_range_max=OAWorkflow.C_RANGE_NONE,
                               p_ada=p_ada,
                               p_visualize=p_visualize, 
                               p_logging=p_logging )


        # 1.2.2 Creation of tasks and add them to the workflow

        # Boundary detector 
        task_bd = BoundaryDetector(p_name='t1', 
                                   p_ada=True, 
                                   p_visualize=True,   
                                   p_logging=p_logging)
        workflow.add_task(p_task = task_bd)

        # MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax(p_name='t2', 
                                            p_ada=True,
                                            p_visualize=p_visualize, 
                                            p_logging=p_logging )

        task_bd.register_event_handler(
            p_event_id=BoundaryDetector.C_EVENT_ADAPTED,
            p_event_handler=task_norm_minmax.adapt_on_event
            )
        workflow.add_task(p_task = task_norm_minmax, p_pred_tasks=[task_bd])

        # Cluster Analyzer
        task_clusterer = WrRiverDBStream2MLPro(p_name='t3',
                                               p_clustering_threshold = 1.5,
                                               p_fading_factor = 0.05,
                                               p_cleanup_interval = 4,
                                               p_intersection_factor = 0.5,
                                               p_minimum_weight = 1.0)
        workflow.add_task(p_task=task_clusterer, p_pred_tasks=[task_norm_minmax])

        # 1.3 Return stream and workflow
        return stream, workflow