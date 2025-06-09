## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.clusterbased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-12  0.1.0     DA       Creation
## -- 2025-03-04  0.2.0     DA       Simplification
## -- 2025-03-19  0.3.0     DA       Methods DriftCB._update_plot*: recovery of origin color
## -- 2025-06-09  1.0.0     DA       Refactoring: new parent ChangeCB
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-06-09)

This module provides a template class for cluster-based drifts to be used in cluster-based drift 
detection algorithms.
"""

from mlpro.bf.various import TStampType
from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.oa.streams.tasks.changedetectors.clusterbased import ChangeCB
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.basics import Drift
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCB (ChangeCB, Drift):
    """
    Sub-type for cluster-based drift events.
    
    Parameters
    ----------
    p_id : int
        Change ID. Default value = 0.
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
        
        ChangeCB(self).__init__( p_id = p_id,
                                 p_status = p_status,
                                 p_tstamp = p_tstamp,
                                 p_visualize = p_visualize, 
                                 p_raising_object = p_raising_object,
                                 p_clusters = p_clusters,
                                 p_properties = p_properties,
                                 **p_kwargs )
        
        Drift(self).__init__( p_id = p_id,
                              p_status = p_status,
                              p_tstamp = p_tstamp,
                              p_visualize = p_visualize, 
                              p_raising_object = p_raising_object,
                              **p_kwargs )
