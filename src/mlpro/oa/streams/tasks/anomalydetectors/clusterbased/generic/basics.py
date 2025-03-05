## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased.generic
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-05  0.1.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-03-05)

This module provides template classes for generic cluster-based anomaly detection
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *
from mlpro.bf.streams import InstDict
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer
from mlpro.oa.streams.tasks.anomalydetectors.clusterbased import AnomalyDetectorCB



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBGeneric(AnomalyDetectorCB):
    """
    Template for generic cluster-based anomaly detectors observing multiple properties.

    Parameters
    ----------
    ...
    p_property : PropertyDefinition
        Cluster property to be observed.
    p_cls_anomaly : type
        Type of anomaly events to be raised.
    ...
    """

    C_TYPE = 'Cluster-based Generic Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                  p_clusterer : ClusterAnalyzer,
                  p_properties : PropertyDefinitions,
                  p_cls_anomaly : type,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  **p_kwargs ):
        
        self.C_REQ_CLUSTER_PROPERTIES = p_properties
        self._cls_anomaly = p_cls_anomaly

        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,
                          p_anomaly_buffer_size = p_anomaly_buffer_size,
                          **p_kwargs )
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBGenSingle(AnomalyDetectorCBGeneric):
    """
    Template for generic cluster-based anomaly detectors observing a single property.

    Parameters
    ----------
    ...
    p_property : PropertyDefinition
        Cluster property to be observed.
    p_cls_anomaly : type
        Type of anomaly events to be raised.
    ...
    """

    C_TYPE = 'Cluster-based Generic Single Property Anomaly Detector'

## -------------------------------------------------------------------------------------------------

    def __init__( self, 
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition,
                  p_cls_anomaly : type,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):  
             
        super().__init__( p_clusterer = p_clusterer,
                          p_properties = [p_property],
                          p_cls_anomaly = p_cls_anomaly,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,
                          **p_kwargs ) 
        
## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
class AnomalyDetectorCBGenMulti(AnomalyDetectorCBGeneric):
    """
    Template for generic cluster-based anomaly detectors observing multiple properties.

    Parameters
    ----------
    ...
    p_properties : PropertyDefinitions
        Cluster properties to be observed.
    p_cls_anomaly : type
        Type of anomaly events to be raised.
    ...
    """

    C_TYPE = 'Cluster-based Generic Multi Property Anomaly Detector'

## -------------------------------------------------------------------------------------------------

    def __init__( self, 
                  p_clusterer : ClusterAnalyzer,
                  p_properties : PropertyDefinitions,
                  p_cls_anomaly : type,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):  
             
        super().__init__( p_clusterer = p_clusterer,
                          p_properties = p_properties,
                          p_cls_anomaly = p_cls_anomaly,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,
                          **p_kwargs )