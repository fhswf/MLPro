## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : geo_size_change_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-12  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-02-12)

This module provides cluster deformation detector algorithm.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.clusterbased.basics import AnomalyDetectorCB
from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDeformationDetector(AnomalyDetectorCB):
    """
    This is the class for detecting the deformation of clusters.

    """
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = []

## -------------------------------------------------------------------------------------------------
def __init__(self,
            p_clusterer : ClusterAnalyzer = None,
            p_name : str = None,
            p_range_max = StreamTask.C_RANGE_THREAD,
            p_ada : bool = True,
            p_duplicate_data : bool = False,
            p_visualize : bool = False,
            p_logging=Log.C_LOG_ALL,
            **p_kwargs):

        super().__init__(p_name = p_name,
                        p_range_max = p_range_max,
                        p_ada = p_ada,
                        p_duplicate_data = p_duplicate_data,
                        p_visualize = p_visualize,
                        p_logging = p_logging,
                        **p_kwargs)
        

    

