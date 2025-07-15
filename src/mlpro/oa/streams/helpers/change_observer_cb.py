## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.helper
## -- Module  : change_observer_cb.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-27  0.1.1     DS       Creation
## -- 2025-06-29  0.1.2     DS       Bug fixes
## -- 2025-07-15  0.2.0     DA/DS    Class ChangeObserverCB: new parameter p_cluster_ids
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2025-07-15)

This module provides the ChangeObserver class to be used for observation and visualization of stream
adaptation events.

"""

from mlpro.bf.various import Log, KWArgs
from mlpro.bf.exceptions import ParamError
from mlpro.bf.plot import PlotSettings

from mlpro.oa.streams.tasks.changedetectors import Change 
from mlpro.oa.streams import OAStreamHelper, OAStreamTask
from mlpro.oa.streams.helpers.change_observer import ChangeObserver
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ChangeObserverCB(ChangeObserver):
    """
    This class observes adaptations of particular oa stream tasks. Its event handler method can be 
    registered for adaptation events of a stream task.

    Parameters
    ----------
    
    """

    C_TYPE              = 'Helper'
    C_NAME              = 'Event Observer (CB)'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = True
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_related_task : OAStreamTask,
                  p_change_event_ids : list = [],
                  p_cluster_ids : list = [],
                  p_logarithmic_plot : bool = True,
                  p_visualize : bool = True,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        super().__init__( p_related_task = p_related_task, 
                          p_change_event_ids = p_change_event_ids, 
                          p_logarithmic_plot = p_logarithmic_plot, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )
        
        self._cluster_ids = p_cluster_ids


## -------------------------------------------------------------------------------------------------
    def _event_handler(self, p_event_id, p_event_object):

        if self._cluster_ids and not set(self._cluster_ids).issubset(list(p_event_object.clusters.keys())):
            return
        
        super()._event_handler(p_event_id, p_event_object)

        try:
            clusters = getattr(p_event_object, "clusters", None)
            centroids = getattr(p_event_object, "centroids", None)
            cluster_size = getattr(p_event_object, "cluster_size", None)

            if clusters is not None:
                self.log(Log.C_LOG_TYPE_I, f"Related cluster IDs: {clusters}")

            if centroids is not None:
                self.log(Log.C_LOG_TYPE_I, f"Centroid coordinates: {centroids}")

            if cluster_size is not None:
                if hasattr(cluster_size, "values"):
                    self.log(Log.C_LOG_TYPE_I, f"Cluster sizes: {cluster_size.values}")
                elif hasattr(cluster_size, "value"):
                    self.log(Log.C_LOG_TYPE_I, f"Cluster size: {cluster_size.value}")
                else:
                    self.log(Log.C_LOG_TYPE_I, f"Cluster size: {cluster_size}")
        except:
            self.log(Log.C_LOG_TYPE_W, f"Could not extract cluster infomation")