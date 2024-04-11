## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : point.py
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



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PointAnomaly (Anomaly):
    """
    Event class for anomaly events when point anomalies are detected.
    
    """

    C_NAME      = 'Point'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instance : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 p_deviation : float=None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instance, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)
        
        self.instance = p_instance
        self.ano_scores = p_ano_scores


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        return super()._init_plot_2d(p_figure, p_settings)


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        return super()._init_plot_3d(p_figure, p_settings)        


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        self._plot_line1 = None
        self._plot_line1_t1 : Text = None


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, p_axlimits_changed: bool, P_xlim, p_ylim, **p_kwargs):
        pass
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, p_axlimits_changed: bool, P_xlim, p_ylim, p_zlim, **p_kwargs):
        pass
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, p_axlimits_changed: bool, p_ylim, **p_kwargs):

        if ( self._plot_line1 is not None ) and not p_axlimits_changed: return
        
        inst_id = self.get_instance()[-1].get_id()
        xpos    = [inst_id, inst_id]
        
        if self._plot_line1 is None:
            label = 'PO(' + str(self.get_id()) + ')'
            self._plot_line1 = p_settings.axes.plot(xpos, p_ylim, color='r', linestyle='dashed', lw=1, label=label)[0]
            self._plot_line1_t1 = p_settings.axes.text(inst_id, p_ylim[1], label, color='r' )

        else:
            self._plot_line1.set_data( xpos, p_ylim )
            self._plot_line1_t1.set(position=(inst_id, p_ylim[1]))


## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        if self._plot_line1 is not None: self._plot_line1.remove()
        if self._plot_line1_t1 is not None: self._plot_line1_t1.remove()
