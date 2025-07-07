## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased
## -- Module  : deformation_detoctor.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-28  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-03-28)

This module provides a cluster-based deformation detector.
"""

from mlpro.bf.various import Log
from mlpro.bf.exceptions import *
from mlpro.bf.math.properties import *
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.driftdetectors.clusterbased.generic import DriftDetectorCBGenSingleMovement

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGenDeformation (DriftDetectorCBGenSingleMovement):
    """
    Cluster based Drift detector for the perticular cluster deformation detector.
    """

    C_NAME = 'Deformation'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition,
                  p_thrs_lower : float,
                  p_thrs_upper : float,
                  p_cls_drift : type,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL ):

        super().__init__( p_clusterer = p_clusterer,
                          p_property = p_property,
                          p_thrs_lower = p_thrs_lower,
                          p_thrs_upper = p_thrs_upper,
                          p_cls_drift = p_cls_drift,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
