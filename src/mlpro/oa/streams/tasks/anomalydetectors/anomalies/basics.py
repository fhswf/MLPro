## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-04-11  1.3.0     DA       Class Anomaly: extensions on methods update_plot_*
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2024-04-11)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from matplotlib.figure import Figure
from mlpro.bf.various import Id
from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.events import Event
from mlpro.bf.streams import Instance



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Anomaly (Id, Event, Plottable):
    """
    This is the base class for anomaly events which can be raised by the anomaly detectors when an
    anomaly is detected.

    Parameters
    ----------
    p_id
        Optional external id.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_color : string
        Color of the anomaly during visualization.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_TYPE                  = 'Anomaly'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instance : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        Id.__init__( self, p_id = 0 )
        Event.__init__( self, p_raising_object=p_raising_object,
                        p_tstamp=p_det_time, **p_kwargs)
        Plottable.__init__( self, p_visualize = p_visualize )

        self.instance : Instance = p_instance
        self.ano_scores = p_ano_scores


## -------------------------------------------------------------------------------------------------
    def get_instance(self) -> Instance:
        return self.instance
    

## -------------------------------------------------------------------------------------------------
    def get_ano_scores(self):
        return self.ano_scores
    

## -------------------------------------------------------------------------------------------------
    def update_plot( self, 
                     p_axlimits_changed : bool = False,
                     p_xlim = None,
                     p_ylim = None,
                     p_zlim = None,
                     **p_kwargs):
        
        return super().update_plot( p_axlimits_changed = p_axlimits_changed,
                                    p_xlim = p_xlim,
                                    p_ylim = p_ylim,
                                    p_zlim = p_zlim,
                                    **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d( self, 
                         p_settings: PlotSettings, 
                         p_axlimits_changed : bool, 
                         P_xlim,
                         p_ylim,
                         **p_kwargs ):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d( self, 
                         p_settings: PlotSettings, 
                         p_axlimits_changed : bool, 
                         P_xlim,
                         p_ylim,
                         p_zlim,
                         **p_kwargs ):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd( self, 
                         p_settings: PlotSettings, 
                         p_axlimits_changed : bool, 
                         p_ylim,
                         **p_kwargs ):
        pass


