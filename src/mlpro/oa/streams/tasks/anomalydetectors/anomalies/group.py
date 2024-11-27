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
## -- 2024-05-07  1.2.1     SK       Bug fix on groupanomaly visualisation
## -- 2024-05-22  1.3.0     SK       Refactoring
## -- 2024-11-27  1.3.1     DA       Bugfix in method GroupAnomaly.__init__()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.1 (2024-11-27)

This module provides a template class for group anomaly to be used in anomaly detection algorithms.
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
    
    Parameters
    ----------
    p_id : int
        Anomaly ID. Default value = 0.
    p_instances : Instance
        List of instances. Default value = None.
    p_ano_scores : list
        List of anomaly scores of instances. Default = None.
    p_det_time : str
        Time of occurance of anomaly. Default = None.
    p_mean : float
        The mean value of the anomaly. Default = None.
    p_mean_deviation : float
        The mean deviation of the anomaly. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_NAME      = 'Group'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id = 0,
                 p_instances : list[Instance] = None,
                 p_ano_scores : list = None,
                 p_det_time : str = None,
                 p_mean : float= None,
                 p_mean_deviation : float = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 **p_kwargs):
        
        super().__init__( p_id = p_id, 
                          p_instances = p_instances, 
                          p_ano_scores = p_ano_scores,
                          p_visualize = p_visualize, 
                          p_raising_object = p_raising_object,
                          p_det_time = p_det_time, 
                          **p_kwargs )
        
        self.plot_update = True
        self._mean = p_mean
        self._mean_deviation = p_mean_deviation


## -------------------------------------------------------------------------------------------------
    def set_instances(self, p_instances = None, p_ano_scores = None):
        """
        Method to set the instances associated with the anomaly.
        
        Parameters
        ----------
        p_instances : list[Instance]
            List of instances. Default is None.
        p_ano_scores : list
            List of anomaly scores.
        """
        self._instances = p_instances
        self._ano_scores = p_ano_scores


## -------------------------------------------------------------------------------------------------
    def get_mean(self) -> float:
        """
        Method that returns the mean value of the anomaly.
        
        Returns
        -------
        float
            The mean value of the anomaly.
        """
        return self._mean


## -------------------------------------------------------------------------------------------------
    def get_mean_deviation(self) -> float:
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

        my_instances = self.get_instances()
    
        x1 = my_instances[0]
        x2 = my_instances[-1]

        x1 = x1.get_id()
        x2 = x2.get_id()
        a=[]
        b=[]

        for instance in my_instances:
            a.append(instance.get_feature_data().get_values())

        for x in a:
            b.extend(x)
            
        y1 = min(b)
        y2 = max(b)

        if self._rect is None:
            label = self.C_NAME[0]
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

