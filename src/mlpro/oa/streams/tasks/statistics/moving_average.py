## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.statistics
## -- Module  : moving_average.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-07-02  1.0.0     DA       Creation
## -- 2025-07-05  1.0.1     DA       Correction of renormalization
## -- 2025-07-07  1.1.0     DA       Class MovingAverage: removal of crosshair in nD view
## -- 2025-07-11  1.2.0     DA       Class MovingAverage: new parameter p_renormalize_plot_data
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.2.0 (2025-07-11)

Ths module provides the class MovingAverage calculating the moving average of incomming new and
outdated instances. 

"""

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math import Element
from mlpro.bf.math.properties import Properties
from mlpro.bf.math.geometry import cprop_crosshair
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.streams import InstDict, InstTypeNew, InstTypeDel, Instance
from mlpro.oa.streams import OAStreamTask



# Export list for public API
__all__ = [ 'MovingAverage' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MovingAverage (OAStreamTask, Properties):
    """
    Online-adaptive stream task computing the moving average (MA) of incoming new and obsolete instances.
 
    Key features are:
    - Incremental update of internally stored MA
    - Real-time renormalization in combination with a prio normalizer
    - 2D, 3D, nD visualization using a moving crosshair property

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_buffer_size : int
        Initial size of internal data buffer. Defaut = 0 (no buffering).
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_remove_obs : bool = True
        Enables/disables the removal of obsolete instances from the internally stored MA. Default = True.
    **p_kwargs 
        Further optional keyword arguments
    """

    C_NAME              = 'Moving average'
    C_PROPERTIES        = [ cprop_crosshair ]   

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name = None, 
                  p_range_max = OAStreamTask.C_RANGE_THREAD, 
                  p_ada : bool = True, 
                  p_buffer_size : int = 0, 
                  p_duplicate_data : bool = False, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  p_remove_obs : bool = True,
                  p_renormalize_plot_data : bool = True,
                  **p_kwargs ):
        
        Properties.__init__( self, p_visualize = p_visualize )
       
        OAStreamTask.__init__( self, 
                               p_name = p_name, 
                               p_range_max = p_range_max, 
                               p_ada = p_ada, 
                               p_buffer_size = p_buffer_size, 
                               p_duplicate_data = p_duplicate_data, 
                               p_visualize = p_visualize, 
                               p_logging = p_logging, 
                               **p_kwargs )
                 
        self._moving_avg            = None
        self._num_inst              = 0
        self._remove_obs            = p_remove_obs
        self._renormalize_plot_data = p_renormalize_plot_data
        self.crosshair.color        = 'red'


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict ):

        # 0 Intro
        inst_avg_id     = -1
        inst_avg_tstamp = None

        
        # 1 Process all incoming new/obsolete stream instances
        for inst_id, (inst_type, inst) in p_instances.items():

            feature_data = inst.get_feature_data().get_values()

            if inst_type == InstTypeNew:
                if self._moving_avg is None:
                    self._moving_avg = feature_data.copy() 
                else:
                    self._moving_avg = ( self._moving_avg * self._num_inst + feature_data ) / ( self._num_inst + 1 )

                self._num_inst += 1

            elif ( inst_type == InstTypeDel ) and self._remove_obs:
                self._moving_avg = ( self._moving_avg * self._num_inst - feature_data ) / ( self._num_inst - 1 )
                self._num_inst  -= 1

            if inst_id > inst_avg_id:
                inst_avg_id     = inst_id
                inst_avg_tstamp = inst.tstamp
                feature_set     = inst.get_feature_data().get_related_set()

        if inst_avg_id == -1: return

            
        # 2 Clear all incoming stream instances
        p_instances.clear()


        # 3 Add a new stream instance containing the moving average 
        inst_avg_data       = Element( p_set = feature_set )
        inst_avg_data.set_values( p_values = self._moving_avg.copy() )
        inst_avg            = Instance( p_feature_data = inst_avg_data, p_tstamp = inst_avg_tstamp )
        inst_avg.id         = inst_avg_id

        p_instances[inst_avg.id] = ( InstTypeNew, inst_avg )

        self.crosshair.value = self._moving_avg
 

## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer: Normalizer):
        try:
            self._moving_avg = p_normalizer.renormalize( p_data = self._moving_avg.copy() )
            self.log(Log.C_LOG_TYPE_W, 'Moving avg renormalized')
        except:
            pass

        if self._renormalize_plot_data:
            self._update_plot_data( p_normalizer = p_normalizer )


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure = None, p_plot_settings = None):
        OAStreamTask.init_plot( self, p_figure = p_figure, p_plot_settings = p_plot_settings )

        if self.get_plot_settings().view != PlotSettings.C_VIEW_ND:
            self.crosshair.init_plot( p_figure = self._figure, 
                                      p_plot_settings = self.get_plot_settings() )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_instances : InstDict = None, **p_kwargs):
        OAStreamTask.update_plot( self, p_instances = p_instances, **p_kwargs )

        if self.get_plot_settings().view != PlotSettings.C_VIEW_ND:
            self.crosshair.update_plot( p_instances = p_instances, **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh = True):
        OAStreamTask.remove_plot(self, p_refresh)
        self.crosshair.remove_plot( p_refresh)


## -------------------------------------------------------------------------------------------------
    def _finalize_plot_view(self, p_inst_ref):
        ps_old = self.get_plot_settings().copy()
        OAStreamTask._finalize_plot_view(self,p_inst_ref)
        ps_new = self.get_plot_settings()

        if ps_new.view != ps_old.view:
            self.crosshair._plot_initialized = False
            self.crosshair.init_plot( p_figure = self._figure, p_plot_settings = ps_new )


## -------------------------------------------------------------------------------------------------
    def _update_plot_data(self, p_normalizer : Normalizer):
        
        if not self.get_visualization(): return
        view = self.get_plot_settings().view

        if view == PlotSettings.C_VIEW_2D:
            self._update_plot_data_2d( p_normalizer = p_normalizer )
        elif view == PlotSettings.C_VIEW_3D:
            self._update_plot_data_3d( p_normalizer = p_normalizer )
        elif view == PlotSettings.C_VIEW_ND:
            self._update_plot_data_nd( p_normalizer = p_normalizer )

        self._update_ax_limits = True
        self._recalc_ax_limits = True


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_2d(self, p_normalizer : Normalizer):
        """
        Updates the 2D plot data after parameter changes by renormalizing the existing points.
        """
        
        if not self._plot_2d_xdata: return

        p_normalizer.renormalize( p_data = self._plot_2d_xdata, p_dim = 0 )
        p_normalizer.renormalize( p_data = self._plot_2d_ydata, p_dim = 1 )


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_3d(self, p_normalizer : Normalizer):
        """
        Method to update the 3d plot for Normalizer. Extended to renormalize the obsolete data on change of parameters.
        """

        if not self._plot_3d_xdata: return

        p_normalizer.renormalize( p_data = self._plot_3d_xdata, p_dim = 0 )
        p_normalizer.renormalize( p_data = self._plot_3d_ydata, p_dim = 1 )
        p_normalizer.renormalize( p_data = self._plot_3d_zdata, p_dim = 2 )


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_nd(self, p_normalizer : Normalizer):
        """
        Method to update the nd plot for Normalizer. Extended to renormalize the obsolete data on change of parameters.
        """

        if not self._plot_nd_plots: return

        for dim, plot_data in enumerate(self._plot_nd_plots):
            p_normalizer.renormalize( p_data=plot_data[0], p_dim=dim )