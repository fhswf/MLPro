## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased
## -- Module  : deformation_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-28  0.0.0     DS       Creation
## -- 2025-05-06  0.0.1     DA       Added default type 'DriftCBDeformation' to param p_cls_drift
## -- 2025-06-10  0.0.2     DA/DS    Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.2 (2025-06-10)

This module provides a cluster-based deformation detector.
"""

from mlpro.bf.various import Log

from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.deformation_index import cprop_deformation_index1
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased import DriftCBDeformation
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.clusterbased.generic import DriftDetectorCBGenSingleGradient



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGenDeformation (DriftDetectorCBGenSingleGradient):
    """
    Cluster based Drift detector for the perticular cluster deformation detector.
    """

    C_NAME = 'Deformation'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_thrs_lower : float,
                  p_thrs_upper : float,
                  p_cls_drift : type = DriftCBDeformation,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL ):

        super().__init__( p_clusterer = p_clusterer,
                          p_property = cprop_deformation_index1,
                          p_thrs_lower = p_thrs_lower,
                          p_thrs_upper = p_thrs_upper,
                          p_cls_drift = p_cls_drift,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging )