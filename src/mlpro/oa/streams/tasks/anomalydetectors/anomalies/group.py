## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-04-10)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.basics import Anomaly
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib import patches



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GroupAnomaly (Anomaly):
    """
    Event class for anomaly events when group anomalies are detected.
    
    """

    C_NAME      = 'Group'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 p_mean : float= None,
                 p_mean_deviation : float = None,
                 **p_kwargs):
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)
        
        self.instances = p_instances
        p_ano_scores = p_ano_scores


## -------------------------------------------------------------------------------------------------
    def set_instances(self, p_instances, p_ano_scores):
        self.instances = p_instances
        self.ano_scores = p_ano_scores


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        self._rect = None
        self._plot_rectangle = None
        self._plot_rectangle_t : Text = None


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        """
        Draw a shaded rectangular region on a plot.

        Parameters:
        ax (matplotlib.axes.Axes): The axes object to draw the shaded region on.
        x1, x2 (float): x-coordinates of the left and right edges of the rectangle.
        y1, y2 (float): y-coordinates of the bottom and top edges of the rectangle.
        color (str): Color of the shaded region.
        alpha (float): Transparency of the shaded region (default is 0.5).
        """
        super()._update_plot_nd(p_settings, **p_kwargs)
    
        label = self.C_NAME[0]
        x1 = self.get_instance()[0].get_id()
        x2 = self.get_instance()[-1].get_id()
        a=[]
        b=[]
        for instance in self.get_instance():
            a.append(instance.get_feature_data().get_values())
        for x in a:
            b.extend(x)
        y1 = min(b)
        y2 = max(b)

        if self._rect is None:
            self._rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0, edgecolor='none', facecolor='yellow', alpha=0.3)
            self._plot_rectangle = p_settings.axes.add_patch(self._rect)
            self._plot_rectangle_t = p_settings.axes.text((x1+x2)/2, 0, label, color='b' )

        else:
            self._rect.set_x(x1)
            self._rect.set_y(y1)
            self._rect.set_width(x2 - x1)
            self._rect.set_height(y2 - y1)
            self._plot_rectangle_t.set_position(((x1+x2)/2, 0))

    
## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        """
        Remove all shaded regions from a plot.

        """

        if self._plot_rectangle is not None: self._plot_rectangle .remove()
        if self._plot_rectangle_t is not None: self._plot_rectangle_t.remove()
