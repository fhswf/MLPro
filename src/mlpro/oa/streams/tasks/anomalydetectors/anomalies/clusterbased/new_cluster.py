## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : new_cluster.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-22  1.2.1     SK       Refactoring
## -- 2024-05-28  1.2.2     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2024-05-28)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import Instance
from mlpro.bf.mt import Figure, PlotSettings
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.basics import CBAnomaly





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NewClusterAppearance (CBAnomaly):
    """
    Event class to be raised when a new cluster appears.
    
    """

    C_NAME      = 'New cluster appearance'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : list[Instance] = None,
                 p_clusters : dict[Cluster] = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)
        
        self._clusters = p_clusters

## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_2d(p_figure=p_figure, p_settings=p_settings)

        cluster : Cluster = None

        for cluster in self._clusters.values(): 

            center = cluster.get_properties()['centroid']._get()

            for r in np.linspace(0, 10, 25):
                alpha = 1 - r / 10
                self.circle = plt.Circle(center[:1], r, color=self.C_ANOMALY_COLORS[self._colour_id], alpha=alpha)
                #ax.add_patch(circle)
            self._colour_id +=1
            if self._colour_id > 9:
                self._colour_id = 0

        """if self._rect is None:
            self._rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.5)
            self._plot_rectangle = p_settings.axes.add_patch(self._rect)
            self._plot_rectangle_t = p_settings.axes.text((x1+x2)/2, 0, label, color='b' )

        else:
            self._rect.set_x(x1)
            self._rect.set_y(y1)
            self._rect.set_width(x2 - x1)
            self._rect.set_height(y2 - y1)
            self._plot_rectangle_t.set_position(((x1+x2)/2, 0))"""
  

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_3d(p_figure=p_figure, p_settings=p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_nd(p_figure=p_figure, p_settings=p_settings)


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_2d(p_settings, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_3d(p_settings, **p_kwargs) 


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_nd(p_settings, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        super()._remove_plot_2d()


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        super()._remove_plot_3d()
  

## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        super()._remove_plot_nd()
