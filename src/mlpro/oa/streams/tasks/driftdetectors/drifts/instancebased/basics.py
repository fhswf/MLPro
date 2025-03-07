## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.instancebased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-04  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-03-04)

This module provides a template class for instance-based drifts to be used in instance-based drift 
detection algorithms.
"""

from datetime import datetime

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.mt import PlotSettings

from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.driftdetectors.drifts.basics import Drift



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftIB (Drift):
    """
    Sub-type for instance-based drift events.
    
    Parameters
    ----------
    p_id : int
        Drift ID. Default value = 0.
    p_tstamp : datetime
        Time stamp of drift detection. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    p_clusters : dict[Cluster]
        Clusters associated with the anomaly. Default = None.
    p_properties : dict
        Poperties of clusters associated with the anomaly. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_drift_status : bool,
                 p_instances : list[Instance],
                 p_id : int = 0,
                 p_tstamp : datetime = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 **p_kwargs):
        
        super().__init__( p_drift_status = p_drift_status,
                          p_id = p_id,
                          p_tstamp = p_tstamp,
                          p_visualize = p_visualize, 
                          p_raising_object = p_raising_object,
                          **p_kwargs )
        
        self.instances : list[Instance] = p_instances