## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : point.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-04-10)


This module provides a template class for point anomaly event to be used in anomaly detection algorithms.
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
    p_deviation : float
        The deviation of the anomaly. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_NAME              = 'Point'

    C_PLOT_CH_SIZE      = 0.06           # Crosshair size in % of visible axes area
    C_PLOT_CH_OFFSET    = 0.4            # Crosshair distance from the center in [0,1]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id = 0,
                 p_instances : list[Instance] = None,
                 p_ano_scores : list = None,
                 p_det_time : str = None,
                 p_deviation : float = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 **p_kwargs):
        
        super().__init__( p_id=p_id,
                          p_instances=p_instances, 
                          p_ano_scores=p_ano_scores,
                          p_visualize=p_visualize, 
                          p_raising_object=p_raising_object,
                          p_det_time=p_det_time, 
                          **p_kwargs )
        
        self._deviation = p_deviation
        

## -------------------------------------------------------------------------------------------------
    def get_deviation(self) -> float:
        """
        Method that returns the deviation of anomaly from the normal distribution of data.
        
        Returns
        -------
        float
            The devaition of anomaly from the normal data distribution.
        """
        return self._instances

## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        self._plot_line_x1 = None
        self._plot_line_x2 = None
        self._plot_line_y1 = None
        self._plot_line_y2 = None
        self._plot_label : Text = None
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        self._init_plot_2d(p_figure = p_figure, p_settings = p_settings)    
        self._plot_line_z1 = None
        self._plot_line_z2 = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        self._plot_line = None
        self._plot_label : Text = None


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, p_axlimits_changed: bool, p_xlim, p_ylim, **p_kwargs):

        if ( self._plot_line_x1 is not None ) and not p_axlimits_changed: return

        inst = self.get_instances()[-1]
        feature_values = inst.get_feature_data().get_values()

        len_x          = ( p_xlim[1] - p_xlim[0] ) * self.C_PLOT_CH_SIZE / 2
        len_y          = ( p_ylim[1] - p_ylim[0] ) * self.C_PLOT_CH_SIZE / 2

        offset_x       = len_x * self.C_PLOT_CH_OFFSET
        offset_y       = len_y * self.C_PLOT_CH_OFFSET

        line_x1_xpos   = [ feature_values[0], feature_values[0] ]
        line_x1_ypos   = [ feature_values[1] + offset_y, feature_values[1] + len_y ]

        line_x2_xpos   = [ feature_values[0], feature_values[0] ]
        line_x2_ypos   = [ feature_values[1] - offset_y, feature_values[1] - len_y ]

        line_y1_xpos   = [ feature_values[0] + offset_x, feature_values[0] + len_x ]
        line_y1_ypos   = [ feature_values[1], feature_values[1] ]

        line_y2_xpos   = [ feature_values[0] - offset_x, feature_values[0] - len_x ]
        line_y2_ypos   = [ feature_values[1], feature_values[1] ]

        if self._plot_line_x1 is None:
            label = 'PO(' + str(self.get_id()) + ')'
            self._plot_line_x1 = p_settings.axes.plot(line_x1_xpos, line_x1_ypos, color='r', linestyle='dashed', lw=1)[0]
            self._plot_line_x2 = p_settings.axes.plot(line_x2_xpos, line_x2_ypos, color='r', linestyle='dashed', lw=1)[0]
            self._plot_line_y1 = p_settings.axes.plot(line_y1_xpos, line_y1_ypos, color='r', linestyle='dashed', lw=1)[0]
            self._plot_line_y2 = p_settings.axes.plot(line_y2_xpos, line_y2_ypos, color='r', linestyle='dashed', lw=1)[0]
            self._plot_label   = p_settings.axes.text(line_x1_xpos[0], line_x1_ypos[1], label, color='r' )
    
        else:
            self._plot_line_x1.set_data( line_x1_xpos, line_x1_ypos )
            self._plot_line_x2.set_data( line_x2_xpos, line_x2_ypos )
            self._plot_line_y1.set_data( line_y1_xpos, line_y1_ypos )
            self._plot_line_y2.set_data( line_y2_xpos, line_y2_ypos )
            self._plot_label.set( position= (line_x1_xpos[0], line_x1_ypos[1]) )


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, p_axlimits_changed: bool, p_xlim, p_ylim, p_zlim, **p_kwargs):

        if ( self._plot_line_x1 is not None ) and not p_axlimits_changed: return

        inst = self.get_instances()[-1]
        feature_values = inst.get_feature_data().get_values()

        len_x          = ( p_xlim[1] - p_xlim[0] ) * self.C_PLOT_CH_SIZE / 2
        len_y          = ( p_ylim[1] - p_ylim[0] ) * self.C_PLOT_CH_SIZE / 2
        len_z          = ( p_zlim[1] - p_zlim[0] ) * self.C_PLOT_CH_SIZE / 2

        offset_x       = len_x * self.C_PLOT_CH_OFFSET
        offset_y       = len_y * self.C_PLOT_CH_OFFSET
        offset_z       = len_z * self.C_PLOT_CH_OFFSET

        line_x1_xpos   = [ feature_values[0], feature_values[0] ]
        line_x1_ypos   = [ feature_values[1] + offset_y, feature_values[1] + len_y ]
        line_x1_zpos   = [ feature_values[2], feature_values[2] ]

        line_x2_xpos   = [ feature_values[0], feature_values[0] ]
        line_x2_ypos   = [ feature_values[1] - offset_y, feature_values[1] - len_y ]
        line_x2_zpos   = [ feature_values[2], feature_values[2] ]

        line_y1_xpos   = [ feature_values[0] + offset_x, feature_values[0] + len_x ]
        line_y1_ypos   = [ feature_values[1], feature_values[1] ]
        line_y1_zpos   = [ feature_values[2], feature_values[2] ]

        line_y2_xpos   = [ feature_values[0] - offset_x, feature_values[0] - len_x ]
        line_y2_ypos   = [ feature_values[1], feature_values[1] ]
        line_y2_zpos   = [ feature_values[2], feature_values[2] ]

        line_z1_xpos   = [ feature_values[0], feature_values[0] ]
        line_z1_ypos   = [ feature_values[1], feature_values[1] ]
        line_z1_zpos   = [ feature_values[2] + offset_z, feature_values[2] + len_z ]

        line_z2_xpos   = [ feature_values[0], feature_values[0] ]
        line_z2_ypos   = [ feature_values[1], feature_values[1] ]
        line_z2_zpos   = [ feature_values[2] - offset_z, feature_values[2] - len_z ]

        if self._plot_line_x1 is None:
            label = 'PO(' + str(self.get_id()) + ')'
            self._plot_line_x1 = p_settings.axes.plot( line_x1_xpos, line_x1_ypos, line_x1_zpos, color='r', linestyle='dashed', lw=1 )[0]
            self._plot_line_x2 = p_settings.axes.plot( line_x2_xpos, line_x2_ypos, line_x2_zpos, color='r', linestyle='dashed', lw=1 )[0]
            self._plot_line_y1 = p_settings.axes.plot( line_y1_xpos, line_y1_ypos, line_y1_zpos, color='r', linestyle='dashed', lw=1 )[0]
            self._plot_line_y2 = p_settings.axes.plot( line_y2_xpos, line_y2_ypos, line_y2_zpos, color='r', linestyle='dashed', lw=1 )[0]
            self._plot_line_z1 = p_settings.axes.plot( line_z1_xpos, line_z1_ypos, line_z1_zpos, color='r', linestyle='dashed', lw=1 )[0]
            self._plot_line_z2 = p_settings.axes.plot( line_z2_xpos, line_z2_ypos, line_z2_zpos, color='r', linestyle='dashed', lw=1 )[0]
            self._plot_label   = p_settings.axes.text( line_z1_xpos[0], line_z1_ypos[0], line_z1_zpos[1], label, color='r' )
    
        else:
            self._plot_line_x1.set_data_3d( line_x1_xpos, line_x1_ypos, line_x1_zpos )
            self._plot_line_x2.set_data_3d( line_x2_xpos, line_x2_ypos, line_x2_zpos )
            self._plot_line_y1.set_data_3d( line_y1_xpos, line_y1_ypos, line_y1_zpos )
            self._plot_line_y2.set_data_3d( line_y2_xpos, line_y2_ypos, line_y2_zpos )
            self._plot_line_z1.set_data_3d( line_z1_xpos, line_z1_ypos, line_z1_zpos )
            self._plot_line_z2.set_data_3d( line_z2_xpos, line_z2_ypos, line_z2_zpos )
            self._plot_label.set( position= ( line_z1_xpos[0], line_z1_ypos[0], line_z1_zpos[1] ) )

        
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
    def _remove_plot_2d(self):
        if self._plot_line_x1 is None: return
        self._plot_line_x1.remove()
        self._plot_line_x2.remove()
        self._plot_line_y1.remove()
        self._plot_line_y2.remove()
        self._plot_label.remove()


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        if self._plot_line_x1 is None: return
        self._remove_plot_2d()
        self._plot_line_z1.remove()
        self._plot_line_z2.remove()


## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        if self._plot_line is None: return
        self._plot_line.remove()
        self._plot_label.remove()
