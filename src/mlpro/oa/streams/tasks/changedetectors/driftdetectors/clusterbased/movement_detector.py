## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased
## -- Module  : movement_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-09  0.0.0     DS       Creation
## -- 2025-06-10  0.0.1     DA/DS    Refactoring
## -- 2025-07-18  0.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-07-18) 

This module provides a cluster-based movement detector.
"""

from mlpro.bf import Log
from mlpro.bf.math.geometry import cprop_center_geo1

from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_centroid1
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased import DriftCBCenterGeo, DriftCBMovement
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.clusterbased.generic import DriftDetectorCBGenSingleGradient



# Export list for public API
__all__ = [ 'DriftDetectorCBGenMovementGeo', 
            'DriftDetectorCBGenMovement' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGenMovementGeo (DriftDetectorCBGenSingleGradient):
    """
    Cluster based Drift detector for the perticular cluster movement detector.
    """

    C_NAME = 'Movement'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_thrs_lower : float,
                  p_thrs_upper : float,
                  p_cls_drift : type = DriftCBCenterGeo,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_drift_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  p_thrs_clusters : int = 1 ):

        super().__init__( p_clusterer = p_clusterer,
                          p_property = cprop_center_geo1,
                          p_thrs_lower = p_thrs_lower,
                          p_thrs_upper = p_thrs_upper,
                          p_cls_drift = p_cls_drift,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          p_drift_buffer_size = p_drift_buffer_size,
                          p_thrs_inst = p_thrs_inst,
                          p_thrs_clusters = p_thrs_clusters )
        



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class DriftDetectorCBGenMovement (DriftDetectorCBGenSingleGradient):
    """
    Cluster based Drift detector for the perticular cluster movement detector.
    """

    C_NAME = 'Movement'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_thrs_lower : float,
                  p_thrs_upper : float,
                  p_cls_drift : type = DriftCBMovement,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_drift_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  p_thrs_clusters : int = 1 ):

        super().__init__( p_clusterer = p_clusterer,
                          p_property = cprop_centroid1,
                          p_thrs_lower = p_thrs_lower,
                          p_thrs_upper = p_thrs_upper,
                          p_cls_drift = p_cls_drift,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          p_drift_buffer_size = p_drift_buffer_size,
                          p_thrs_inst = p_thrs_inst,
                          p_thrs_clusters = p_thrs_clusters )