## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-28  1.3.0     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2024-05-28)

This module provides template for cluster-based anomaly detection algorithms to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.basics import AnomalyDetector
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.math.properties import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCB(AnomalyDetector):
    """
    This is the base class for cluster-based online anomaly detectors. It raises an event when an
    anomaly is detected in a cluster dataset.

    """

    C_TYPE = 'Cluster based Anomaly Detector'

    # List of cluster properties necessary for the algorithm
    C_REQ_CLUSTER_PROPERTIES : PropertyDefinitions = []


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
        
        self._clusterer = p_clusterer

        for x in self.C_PROPERTY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        #if len(unknown_prop) >0:
        #    raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)
        
        self._visualize = p_visualize

