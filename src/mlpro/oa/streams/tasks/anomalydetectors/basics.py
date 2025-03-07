## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-04-11  1.3.0     DA       Methods AnomalyDetector.init/update_plot: determination and
## --                                forwarding of changes on ax limits
## -- 2024-05-22  1.4.0     SK       Refactoring
## -- 2024-08-12  1.4.1     DA       Correction in AnomalyDetector.update_plot()
## -- 2024-12-11  1.4.2     DA       Pseudo classes if matplotlib is not installed
## -- 2025-02-14  1.5.0     DA       Review and refactoring
## -- 2025-03-03  1.5.1     DA       Corrections
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.1 (2025-03-03)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.streams import InstDict

from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.anomalydetectors.anomalies import Anomaly



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetector (OAStreamTask):
    """
    Base class for online anomaly detectors. It raises an event when an
    anomaly is detected.

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_anomaly_buffer_size : int = 100
        Size of the internal anomaly buffer self.anomalies. Default = 100.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE              = 'Anomaly Detector'
    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = False

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name:str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )
        
        self._ano_id : int          = 0
        self.anomalies              = {}
        self._ano_buffer_size : int = p_anomaly_buffer_size


## -------------------------------------------------------------------------------------------------
    def _get_next_anomaly_id(self):
        """
        Methd that returns the id of the next anomaly. 

        Returns
        -------
        _ano_id : int
        """

        self._ano_id +=1
        return self._ano_id


## -------------------------------------------------------------------------------------------------
    def _buffer_anomaly(self, p_anomaly:Anomaly):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be added.
        """

        # 1 Buffering turned on?
        if self._ano_buffer_size <= 0: return

        # 2 Buffer full?
        if len( self.anomalies ) >= self._ano_buffer_size:
            # 2.1 Remove oldest entry
            oldest_key     = next(iter(self.anomalies))
            oldest_anomaly = self.anomalies.pop(oldest_key)
            oldest_anomaly.remove_plot()

        # 3 Buffer new anomaly
        p_anomaly.id = self._get_next_anomaly_id() 
        self.anomalies[p_anomaly.id] = p_anomaly


## -------------------------------------------------------------------------------------------------
    def _remove_anomaly(self, p_anomaly:Anomaly):
        """
        Method to remove an existing anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be removed.
        """

        p_anomaly.remove_plot(p_refresh=True)
        del self.anomalies[p_anomaly.id]


## -------------------------------------------------------------------------------------------------
    def _raise_anomaly_event(self, p_anomaly:Anomaly, p_buffer: bool = True):
        """
        Method to raise an anomaly event.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be raised.
        p_buffer : bool
            Anomaly is buffered when set to True.
        """

        if p_buffer: self._buffer_anomaly( p_anomaly=p_anomaly )

        if self.get_visualization(): 
            p_anomaly.init_plot( p_figure=self._figure, 
                                 p_plot_settings=self.get_plot_settings() )

        self._raise_event( p_event_id = p_anomaly.event_id,
                           p_event_object = p_anomaly )

                 
## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        self._plot_ax_xlim = None
        self._plot_ax_ylim = None
        self._plot_ax_zlim = None

        super().init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)

        for anomaly in self.anomalies.values():
            anomaly.init_plot(p_figure=p_figure, p_plot_settings = p_plot_settings)
    

## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_inst : InstDict = None, **p_kwargs):
    
        if not self.get_visualization(): return

        # super().update_plot(p_inst, **p_kwargs)

        axes = self._plot_settings.axes

        ax_xlim_new = axes.get_xlim()
        if self._plot_settings.view != PlotSettings.C_VIEW_ND:
            axlimits_changed = ( self._plot_ax_xlim is None ) or ( self._plot_ax_xlim != ax_xlim_new )
        else:
            axlimits_changed = False

        ax_ylim_new = axes.get_ylim()
        axlimits_changed = axlimits_changed or ( self._plot_ax_ylim is None ) or ( self._plot_ax_ylim != ax_ylim_new )
        try:
            ax_zlim_new = axes.get_zlim()
            axlimits_changed = axlimits_changed or ( self._plot_ax_zlim is None ) or ( self._plot_ax_zlim != ax_zlim_new )
        except:
            ax_zlim_new = None
        
        self._plot_ax_xlim = ax_xlim_new
        self._plot_ax_ylim = ax_ylim_new
        self._plot_ax_zlim = ax_zlim_new

        for anomaly in self.anomalies.values():
            anomaly.update_plot( p_axlimits_changed = axlimits_changed,
                                 p_xlim = ax_xlim_new,
                                 p_ylim = ax_ylim_new,
                                 p_zlim = ax_zlim_new,
                                 **p_kwargs )
    

## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh: bool = True):

        if not self.get_visualization(): return

        # super().remove_plot(p_refresh=p_refresh)

        for anomaly in self.anomalies.values():
            anomaly.remove_plot(p_refresh=p_refresh)


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer):
        """
        Internal renormalization of all buffered anomalies. See method OATask.renormalize_on_event() 
        for further information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """

        for anomaly in self.anomalies.values():
           anomaly.renormalize( p_normalizer=p_normalizer )