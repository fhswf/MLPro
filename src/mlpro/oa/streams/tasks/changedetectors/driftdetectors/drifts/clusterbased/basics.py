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
## -- 2025-06-11  1.0.1     DA       Corrections
## -- 2025-06-13  1.1.0     DA       Class Change: param p_id is now initialized to -1
## -- 2025-07-18  1.2.0     DA       Refactoring
## -- 2025-10-07  1.2.1     DA       Bugfix: set p_id to -1 in DriftCB.__init__()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.1 (2025-10-07)

This module provides a template class for cluster-based drifts to be used in cluster-based drift 
detection algorithms.
"""

from mlpro.bf import TStampType
from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.oa.streams.tasks.changedetectors.clusterbased import ChangeCB
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.basics import Drift
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster



# Export list for public API
__all__ = [ 'DriftCB' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCB (ChangeCB, Drift):
    """
    Sub-type for cluster-based drift events.
    
    Parameters
    ----------
    p_id : int = -1
        Drift ID. Default value = -1, indicating that the ID is not set. In that case, the id is
        automatically generated when raising the drift.
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
    
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id : int = -1,
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
        
        Drift.__init__( self,
                        p_id = p_id,
                        p_status = p_status,
                        p_tstamp = p_tstamp,
                        p_visualize = p_visualize, 
                        p_raising_object = p_raising_object,
                        **p_kwargs )
