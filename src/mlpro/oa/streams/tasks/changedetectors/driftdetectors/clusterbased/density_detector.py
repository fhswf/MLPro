## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased
## -- Module  : density_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-09  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-06-09)

This module provides a cluster-based density detector.
"""

from mlpro.bf.various import Log
from mlpro.bf.exceptions import *
from mlpro.bf.math.properties import *
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.deformation_index import cprop_deformation_index1
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased import DriftCBDensity
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.clusterbased.generic import DriftDetectorCBGenSingleMovement



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGenDensity (DriftDetectorCBGenSingleMovement):
    """
    Cluster based Drift detector for the perticular cluster density detector.
    """

    C_NAME = 'Density'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_thrs_lower : float,
                  p_thrs_upper : float,
                  p_cls_drift : type = DriftCBDensity,
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