## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.driftdetectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-12  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-02-12)

This module provides templates for drift detection to be used in the context of online adaptivity.
"""

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import InstDict
from mlpro.bf.various import Log
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.driftdetectors.drifts import Drift



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetector (OAStreamTask):
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
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE                  = 'Drift Detector'

    C_EVENT_DRIFT_ADDED     = 'DRIFT_ADDED'
    C_EVENT_DRIFT_REMOVED   = 'DRIFT_REMOVED'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name:str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )
        
        self._drift_id      = 0
        self._drifts        = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        pass


## -------------------------------------------------------------------------------------------------
    def _get_next_drift_id(self):
        """
        Methd that returns the id of the next drift. 

        Returns
        -------
        drift_id : int
        """

        self._drift_id +=1
        return self._drift_id


## -------------------------------------------------------------------------------------------------
    def get_drifts(self):
        """
        Method to return the current list of drifts. 

        Returns
        -------
        drifts : dict[Drift]
            Current dictionary of drifts.
        """

        return self._drifts
    

## -------------------------------------------------------------------------------------------------
    def _buffer_drift(self, p_drift:Drift):
        """
        Method to be used internally to add a new drift object. Please use as part of your algorithm.

        Parameters
        ----------
        p_drift : Drift
            Drift object to be added.

        Returns
        -------
        p_drift  Drift
            Enriched drift object
        """

        p_drift.id = self._get_next_drift_id()
        self._drifts[p_drift.id] = p_drift
        return p_drift


## -------------------------------------------------------------------------------------------------
    def remove_drift(self, p_drift:Drift):
        """
        Method to remove an existing drift object. Please use as part of your algorithm.

        Parameters
        ----------
        p_drift : Drift
            Drift object to be removed.
        """

        p_drift.remove_plot(p_refresh=True)
        del self._drifts[p_drift.id]


## -------------------------------------------------------------------------------------------------
    def _raise_drift_event( self, 
                            p_event_id : str, 
                            p_drift : Drift ):
        """
        Specialized method to raise drift events. 

        Parameters
        ----------
        p_event_id : str
            Event id. See class constants C_EVENT_DRIFT_ADDED/REMOVED.
        p_drift : Drift
            Drift event object to be raised.
        """

        self._buffer_drift( p_drift = p_drift )

        if self.get_visualization(): 
            p_drift.init_plot( p_figure=self._figure, p_plot_settings=self.get_plot_settings() )

        return super()._raise_event( p_event_id = p_event_id, p_event_object = p_drift )
    
                 
## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        self._plot_ax_xlim = None
        self._plot_ax_ylim = None
        self._plot_ax_zlim = None

        #super().init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)

        for drift in self._drifts.values():
            drift.init_plot(p_figure=p_figure, p_plot_settings = p_plot_settings)
    

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

        for drift in self._drifts.values():
            drift.update_plot( p_axlimits_changed = axlimits_changed,
                               p_xlim = ax_xlim_new,
                               p_ylim = ax_ylim_new,
                               p_zlim = ax_zlim_new,
                               **p_kwargs )
    

## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh: bool = True):

        if not self.get_visualization(): return

        # super().remove_plot( p_refresh = p_refresh )

        for drift in self._drifts.values():
            drift.remove_plot(p_refresh=p_refresh)