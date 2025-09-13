## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
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
## -- 2024-12-11  1.3.1     DA       Pseudo classes if matplotlib is not installed
## -- 2025-06-09  2.0.0     DA       Refactoring: new parent ChangeCB
## -- 2025-06-11  2.0.1     DA       Corrections
## -- 2025-06-13  2.1.0     DA       Class Change: param p_id is now initialized to -1
## -- 2025-07-18  2.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.2.0 (2025-07-18)

This module provides a template class for cluster-based anomalies to be used in anomaly detection algorithms.
"""


from mlpro.bf.various import TStampType
from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.oa.streams.tasks.changedetectors.clusterbased import ChangeCB
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.basics import Anomaly
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster



# Export list for public API
__all__ = [ 'AnomalyCB' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyCB (ChangeCB, Anomaly):
    """
    Subtype for cluster-based anomaly events.
    
    Parameters
    ----------
    p_id : int = -1
        Anomaly ID. Default value = -1, indicating that the ID is not set. In that case, the id is
        automatically generated when raising the anomaly.
    p_status : bool = True
        Status of the change.
    p_tstamp : TStampType = None
        Time of occurance of change. Default = None.
    p_visualize : bool = False
        Boolean switch for visualisation. Default = False.
    p_raising_object : object = None
        Reference of the object raised. Default = None.
    p_clusters : dict[Cluster] = {}
        Clusters associated with the anomaly.
    p_properties : PropertyDefinitions = []
        List of properties of clusters associated with the anomaly.
    **p_kwargs
        Further optional keyword arguments.
    """
    
    def __init__( self,
                  p_id : int = 0,
                  p_status : bool = True,
                  p_tstamp : TStampType = None,
                  p_visualize : bool = False,
                  p_raising_object : object = None,
                  p_clusters : dict[Cluster] = {},
                  p_properties : PropertyDefinitions = [],
                  **p_kwargs ):
        
        ChangeCB.__init__( self,
                           p_id = p_id,
                           p_status = p_status,
                           p_tstamp = p_tstamp,
                           p_visualize = p_visualize, 
                           p_raising_object = p_raising_object,
                           p_clusters = p_clusters,
                           p_properties = p_properties,
                           **p_kwargs )
        
        Anomaly.__init__( self,
                          p_id = p_id,
                          p_status = p_status,
                          p_tstamp = p_tstamp,
                          p_visualize = p_visualize, 
                          p_raising_object = p_raising_object,
                          **p_kwargs )