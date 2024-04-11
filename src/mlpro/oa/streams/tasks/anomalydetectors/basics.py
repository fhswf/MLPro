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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2024-04-11)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from typing import List
from matplotlib.figure import Figure
from mlpro.bf.plot import PlotSettings
from mlpro.bf.various import Log
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks import OATask
from mlpro.oa.streams.tasks.anomalydetectors.anomalies import Anomaly



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetector(OATask):
    """
    This is the base class for online anomaly detectors. It raises an event when an
    anomaly is detected.

    """

    C_TYPE          = 'Anomaly Detector'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = OATask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self._ano_id = 0
        self._anomalies = {}
        self._ano_scores = []


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass


## -------------------------------------------------------------------------------------------------
    def _get_next_anomaly_id(self):
        self._ano_id +=1
        return self._ano_id


## -------------------------------------------------------------------------------------------------
    def get_anomalies(self):
        """
        This method returns the current list of anomalies. 

        Returns
        -------
        dict_of_anomalies : dict[Anomaly]
            Current dictionary of anomalies.
        """

        return self._anomalies
    

## -------------------------------------------------------------------------------------------------
    def _buffer_anomaly(self, p_anomaly:Anomaly):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be added.
        """

        p_anomaly.set_id( p_id = self._get_next_anomaly_id() )
        self._anomalies[p_anomaly.get_id()] = p_anomaly
        return p_anomaly


## -------------------------------------------------------------------------------------------------
    def remove_anomaly(self, p_anomaly):
        """
        Method to remove an existing anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be removed.
        """

        p_anomaly.remove_plot(p_refresh=True)
        del self._anomalies[p_anomaly.get_id()]


## -------------------------------------------------------------------------------------------------
    def _raise_anomaly_event(self, p_anomaly : Anomaly ):

        p_anomaly = self._buffer_anomaly(p_anomaly=p_anomaly)

        if self.get_visualization(): 
            p_anomaly.init_plot( p_figure=self._figure, p_plot_settings=self.get_plot_settings() )

        self._raise_event(p_anomaly.C_NAME, p_anomaly)

                 
## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        self._plot_ax_xlim = None
        self._plot_ax_ylim = None
        self._plot_ax_zlim = None

        super().init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)

        for anomaly in self._anomalies.values():
            anomaly.init_plot(p_figure=p_figure, p_plot_settings = p_plot_settings)
    

## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_inst_new: List[Instance] = None, p_inst_del: List[Instance] = None, **p_kwargs):
    
        if not self.get_visualization(): return

        super().update_plot(p_inst_new, p_inst_del, **p_kwargs)

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

        for anomaly in self._anomalies.values():
            anomaly.update_plot( p_axlimits_changed = axlimits_changed,
                                 p_xlim = ax_xlim_new,
                                 p_ylim = ax_ylim_new,
                                 p_zlim = ax_zlim_new,
                                 **p_kwargs )
    

## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh: bool = True):

        if not self.get_visualization(): return

        super().remove_plot(p_refresh=p_refresh)

        for anomaly in self._anomalies.values():
            anomaly.remove_plot(p_refresh=p_refresh)


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer: Normalizer):
        """
        Internal renormalization of all anomaly instances. See method OATask.renormalize_on_event() for further
        information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """

        for anomaly in self._anomalies.values():
            anomaly.get_instance().renormalize( p_normalizer=p_normalizer)
