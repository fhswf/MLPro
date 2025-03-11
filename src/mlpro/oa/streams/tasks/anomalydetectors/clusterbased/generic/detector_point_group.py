## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased.generic
## -- Module  : detector_point_group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-11  0.1.0     DS/DA    Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-03-11)

This module provides an implementation of a generic cluster-based detector for point and group anomalies.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size
from mlpro.oa.streams.tasks.anomalydetectors.clusterbased.generic.basics import AnomalyDetectorCBGenSingle



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBGenPAGA ( AnomalyDetectorCBGenSingle ):
    """
    Implementation of a generic cluster-based detector for point and group anomalies.

    Parameters
    ----------
    ...
    p_property : PropertyDefinition
        Cluster property to be observed.
    p_cls_anomaly : type
        Type of anomaly events to be raised.
    ...
    """

    C_NAME = 'Point and group'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition = cprop_size,
                  p_cls_anomaly : type,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  **p_kwargs ):
        
        super().__init__( p_clusterer=p_clusterer,
                          p_property = p_property,
                          p_cls_anomaly = p_cls_anomaly,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,
                          p_anomaly_buffer_size = p_anomaly_buffer_size,
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def _detect_anomaly( self, 
                         p_cluster : Cluster, 
                         p_properties : PropertyDefinitions, 
                         **p_kwargs ) -> bool:
        """
        Custom method to detect an anomaly of the given cluster.

        Parameters
        ----------
        p_cluster : Cluster
            Cluster to be observed.
        p_properties : PropertyDefinitions
            List of cluster properties to be processed.
        **p_kwargs
            Dictionary with further keyword arguments parameterizing the detection.

        Returns
        -------
        bool
            True, if the cluster is an anomaly. False otherwise.
        """
        
        raise NotImplementedError
