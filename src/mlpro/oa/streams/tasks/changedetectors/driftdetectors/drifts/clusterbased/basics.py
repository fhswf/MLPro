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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-03-19)

This module provides a template class for cluster-based drifts to be used in cluster-based drift 
detection algorithms.
"""

from datetime import datetime

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.mt import PlotSettings
from mlpro.oa.streams.tasks.driftdetectors.drifts.basics import Drift
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCB (Drift):
    """
    Sub-type for cluster-based drift events.
    
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

    C_PLOT_ACTIVE   = True

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_drift_status : bool,
                 p_id : int = 0,
                 p_tstamp : datetime = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_clusters : dict[Cluster] = None,
                 p_properties : dict = None,
                 **p_kwargs):
        
        super().__init__( p_drift_status = p_drift_status,
                          p_id = p_id,
                          p_tstamp = p_tstamp,
                          p_visualize = p_visualize, 
                          p_raising_object = p_raising_object,
                          **p_kwargs )
        
        self.clusters : dict[Cluster] = p_clusters
        self.properties : dict = p_properties


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_2d(p_figure=p_figure, p_settings=p_settings)

        cluster : Cluster = None

        for cluster in self.clusters.values(): 
            if self.drift_status:
                cluster.color_bak = cluster.color
                cluster.color = "red"
            else:
                try:
                    cluster.color = cluster.color_bak
                except:
                    pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_3d(p_figure=p_figure, p_settings=p_settings)
    
        cluster : Cluster = None

        for cluster in self.clusters.values(): 
            if self.drift_status:
                cluster.color_bak = cluster.color
                cluster.color = "red"
            else:
                try:
                    cluster.color = cluster.color_bak
                except:
                    pass
