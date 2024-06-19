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
## -- 2024-05-28  1.2.1     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.1 (2024-05-28)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import Instance
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.basics import Anomaly
from mlpro.bf.mt import Figure, PlotSettings
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster
from matplotlib.figure import Figure
from matplotlib.text import Text





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CBAnomaly (Anomaly):
    """
    Event class to be raised when cluster-based anomalies are detected.
    
    """
    
    C_NAME      = 'Cluster based Anomaly'
    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

    C_ANOMALY_COLORS        = [ 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan' ]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id : int = 0,
                 p_instances : list[Instance] = None,
                 p_clusters : dict[Cluster] = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_id=p_id,
                         p_instances=p_instances,
                         p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, 
                         p_raising_object=p_raising_object,
                         p_det_time=p_det_time,
                         **p_kwargs)
        
        self._colour_id = 0
        self._clusters : dict[Cluster] = p_clusters

## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        return self._clusters
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        self._plot_line = None
        self._plot_label : Text = None


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, p_axlimits_changed: bool, p_ylim, **p_kwargs):

        if ( self._plot_line is not None ) and not p_axlimits_changed: return
        
        inst = self.get_instances()[-1]

        inst_id = inst.get_id()
        xpos    = [inst_id, inst_id]
        
        if self._plot_line is None:
            label = 'PO(' + str(self.get_id()) + ')'
            self._plot_line  = p_settings.axes.plot(xpos, p_ylim, color='r', linestyle='dashed', lw=1)[0]
            self._plot_label = p_settings.axes.text(inst_id, p_ylim[1], label, color='r' )

        else:
            self._plot_line.set_data( xpos, p_ylim )
            self._plot_label.set(position=(inst_id, p_ylim[1]))


## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        if self._plot_line is None: return
        self._plot_line.remove()
        self._plot_label.remove()
