## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-11  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-03-11)

This module provides a class for group anomalies to be used in anomaly detection algorithms.
"""

from datetime import datetime

try:
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib import patches
except:
    class Figure : pass
    class Text : pass
    class patches : pass
    
from mlpro.bf.plot import PlotSettings
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.clusterbased.basics import AnomalyCB



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GroupAnomaly (AnomalyCB):
    """
    Event class for anomaly events when group anomalies are detected.
    """

    pass

# ## -------------------------------------------------------------------------------------------------
#     def __init__(self,
#                  p_instances : list[Instance],
#                  p_id = 0,
#                  p_tstamp : datetime = None,
#                  p_visualize : bool = False,
#                  p_raising_object : object = None,
#                  p_mean : float= None,
#                  p_mean_deviation : float = None,
#                  **p_kwargs):
        
#         super().__init__( p_instances = p_instances, 
#                           p_id = p_id, 
#                           p_visualize = p_visualize, 
#                           p_raising_object = p_raising_object,
#                           p_tstamp = p_tstamp,
#                           **p_kwargs )
        
#         self.plot_update = True
#         self._mean = p_mean
#         self._mean_deviation = p_mean_deviation


# ## -------------------------------------------------------------------------------------------------
#     def _get_mean(self) -> float:
#         """
#         Method that returns the mean value of the anomaly.
        
#         Returns
#         -------
#         float
#             The mean value of the anomaly.
#         """
#         return self._mean


# ## -------------------------------------------------------------------------------------------------
#     def _get_mean_deviation(self) -> float:
#         """
#         Method that returns the mean deviation of anomaly from the normal distribution of data.
        
#         Returns
#         -------
#         float
#             The mean devaition of the anomaly from the normal data distribution.
#         """
#         return self._mean_deviation
    

# ## -------------------------------------------------------------------------------------------------
#     def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
#         self._rect = None
#         self._plot_rectangle = None
#         self._plot_rectangle_t : Text = None


# ## -------------------------------------------------------------------------------------------------
#     def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
#         """
#         Draw a shaded rectangular region on a plot.

#         Parameters:
#         ax (matplotlib.axes.Axes): The axes object to draw the shaded region on.
#         x1, x2 (float): x-coordinates of the left and right edges of the rectangle.
#         y1, y2 (float): y-coordinates of the bottom and top edges of the rectangle.
#         color (str): Color of the shaded region.
#         alpha (float): Transparency of the shaded region (default is 0.5).
#         """

#         if not self.plot_update: return

#         x1 = self.instances[0].tstamp
#         x2 = self.instances[-1].tstamp

#         y_values = []

#         for instance in self.instances:
#             y_values.extend(instance.get_feature_data().get_values())

           
#         y1 = min(y_values)
#         y2 = max(y_values)

#         if self._rect is None:
#             label = 'GA(' + str(self.id) + ')'
#             self._rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.5)
#             self._plot_rectangle = p_settings.axes.add_patch(self._rect)
#             self._plot_rectangle_t = p_settings.axes.text((x1+x2)/2, 0, label, color='b' )

#         else:
#             self._rect.set_x(x1)
#             self._rect.set_y(y1)
#             self._rect.set_width(x2 - x1)
#             self._rect.set_height(y2 - y1)
#             self._plot_rectangle_t.set_position(((x1+x2)/2, 0))

    
# ## -------------------------------------------------------------------------------------------------
#     def _remove_plot_nd(self):
#         """
#         Remove all shaded regions from a plot.

#         """

#         if self._plot_rectangle is not None: self._plot_rectangle .remove()
#         if self._plot_rectangle_t is not None: self._plot_rectangle_t.remove()


# ## -------------------------------------------------------------------------------------------------
#     mean            = property( fget = _get_mean )
#     mean_deviation  = property( fget = _get_mean_deviation )

class SpatialGroupAnomaly (GroupAnomaly):

    pass


class TemporalGroupAnomaly (GroupAnomaly):

    pass