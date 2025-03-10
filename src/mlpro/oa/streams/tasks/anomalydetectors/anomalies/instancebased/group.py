## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.instancebased
## -- Module  : group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-07  1.2.1     SK       Bug fix on groupanomaly visualisation
## -- 2024-05-22  1.3.0     SK       Refactoring
## -- 2024-11-27  1.3.1     DA       Bugfix in method GroupAnomaly.__init__()
## -- 2024-12-11  1.3.2     DA       Pseudo classes if matplotlib is not installed
## -- 2025-03-05  1.4.0     DA       Code optimization
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2025-03-05)

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
from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.instancebased.basics import AnomalyIB



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GroupAnomaly (AnomalyIB):
    """
    Event class for anomaly events when group anomalies are detected.
    
    Parameters
    ----------
    p_instances : Instance
        List of instances. Default value = None.
    p_id : int
        Anomaly ID. Default value = 0.
    p_tstamp : datetime = None
        Time of occurance of anomaly. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    p_mean : float
        The mean value of the anomaly. Default = None.
    p_mean_deviation : float
        The mean deviation of the anomaly. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : list[Instance],
                 p_id = 0,
                 p_tstamp : datetime = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_mean : float= None,
                 p_mean_deviation : float = None,
                 **p_kwargs):
        
        super().__init__( p_instances = p_instances, 
                          p_id = p_id, 
                          p_visualize = p_visualize, 
                          p_raising_object = p_raising_object,
                          p_tstamp = p_tstamp,
                          **p_kwargs )
        
        self.plot_update = True
        self._mean = p_mean
        self._mean_deviation = p_mean_deviation


## -------------------------------------------------------------------------------------------------
    def _get_mean(self) -> float:
        """
        Method that returns the mean value of the anomaly.
        
        Returns
        -------
        float
            The mean value of the anomaly.
        """
        return self._mean


## -------------------------------------------------------------------------------------------------
    def _get_mean_deviation(self) -> float:
        """
        Method that returns the mean deviation of anomaly from the normal distribution of data.
        
        Returns
        -------
        float
            The mean devaition of the anomaly from the normal data distribution.
        """
        return self._mean_deviation
    

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

        if not self.plot_update: return

        x1 = self.instances[0].tstamp
        x2 = self.instances[-1].tstamp

        y_values = []

        for instance in self.instances:
            y_values.extend(instance.get_feature_data().get_values())

           
        y1 = min(y_values)
        y2 = max(y_values)

        if self._rect is None:
            label = 'GA(' + str(self.id) + ')'
            self._rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.5)
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


## -------------------------------------------------------------------------------------------------
    mean            = property( fget = _get_mean )
    mean_deviation  = property( fget = _get_mean_deviation )