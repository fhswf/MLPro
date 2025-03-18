## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased
## -- Module  : movement.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 202         0.0.0              Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (202)

This module provides a cluster-based drift detector algorithm determining cluster deformation.
"""

from mlpro.oa.streams.tasks.driftdetectors.clusterbased import DriftDetectorCB
from mlpro.oa.streams.tasks.driftdetectors.drifts.clusterbased import DriftCBDeformation
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import DeformationIndex





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBDeformation (DriftDetectorCB):
    """
    This is the class for detecting the deformation of clusters.

    """

    C_TYPE = 'Cluster-based Drift Detector (Deformation)'
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = []

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_name:str = None,
                  p_range_max = StreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        super().__init__( p_clusterer = p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )