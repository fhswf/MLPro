## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.changedetectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-28  0.1.0     DA/DS    Creation
## -- 2025-05-30  1.0.0     DA/DS    Completion
## -- 2025-06-03  1.1.0     DA       Class ChangeDetector: new parameter p_thrs_inst
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-06-03)

This module provides templates for change detection to be used in the context of online adaptivity.
"""

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from datetime import datetime

from mlpro.bf.various import Id, Log
from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.events import Event
from mlpro.bf.math.normalizers import Renormalizable
from mlpro.bf.streams import Instance, InstDict, InstTypeNew

from mlpro.oa.streams import OAStreamTask




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Change (Id, Event, Plottable, Renormalizable):
    """
    This is the base class for change events which can be raised by the change detectors when an
    change is detected.

    Parameters
    ----------
    p_id : int
        Change ID. Default value = 0.
    p_status : bool = True
        Status of the change.
    p_tstamp : datetime
        Time of occurance of change. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_PLOT_ACTIVE           = False
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id : int = 0,
                  p_status : bool = True,
                  p_tstamp : datetime = None,
                  p_visualize : bool = False,
                  p_raising_object : object = None,
                  **p_kwargs ):
        
        Id.__init__( self, p_id = p_id )

        Event.__init__( self, 
                        p_raising_object=p_raising_object,
                        p_tstamp=p_tstamp, 
                        **p_kwargs )
        
        Plottable.__init__( self, p_visualize = p_visualize )

        self._status: bool = p_status
        
        if p_status:
            self._event_id = type(self).__name__ + '(ON)'
        else:
            self._event_id = type(self).__name__ + '(OFF)'


## -------------------------------------------------------------------------------------------------
    def _get_status(self) -> bool:
        return self._status
    

## -------------------------------------------------------------------------------------------------
    def _get_event_id(self) -> str:
        return self._event_id


## -------------------------------------------------------------------------------------------------
    status      = property( fget = _get_status )
    event_id    = property( fget = _get_event_id )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ChangeDetector (OAStreamTask):
    """
    Base class for online change detectors.

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
    p_change_buffer_size : int = 100
        Size of the internal change buffer self.changes. Default = 100.
    p_thrs_inst : int = 0
        The algorithm is only executed after this number of instances.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE              = 'Change Detector'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = False

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name:str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  p_change_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )
        
        self._change_id : int          = 0
        self.changes                   = {}
        self._change_buffer_size : int = p_change_buffer_size
        self._thrs_inst : int          = p_thrs_inst
        self._num_inst : int           = 0
        self._chk_num_inst : bool      = True


## -------------------------------------------------------------------------------------------------
    def _get_next_change_id(self):
        """
        Methd that returns the id of the next change. 

        Returns
        -------
        _change_id : int
        """

        self._change_id +=1
        return self._change_id


## -------------------------------------------------------------------------------------------------
    def _buffer_change(self, p_change:Change):
        """
        Method to be used to add a new change. Please use as part of your algorithm.

        Parameters
        ----------
        p_change : Change
            Change object to be added.
        """

        # 1 Buffering turned on?
        if self._change_buffer_size <= 0: return

        # 2 Buffer full?
        if len( self.changes ) >= self._change_buffer_size:
            # 2.1 Remove oldest entry
            oldest_key    = next(iter(self.changes))
            oldest_change = self.changes.pop(oldest_key)
            oldest_change.remove_plot()

        # 3 Buffer new change
        p_change.id = self._get_next_change_id() 
        self.changes[p_change.id] = p_change


## -------------------------------------------------------------------------------------------------
    def _remove_change(self, p_change:Change):
        """
        Method to remove an existing change. Please use as part of your algorithm.

        Parameters
        ----------
        p_change : change
            change object to be removed.
        """

        p_change.remove_plot(p_refresh=True)
        del self.changes[p_change.id]


## -------------------------------------------------------------------------------------------------
    def _raise_change_event( self, 
                             p_change: Change, 
                             p_inst : Instance = None,
                             p_buffer: bool = True ):
        """
        Method to raise an change event. 

        Parameters
        ----------
        p_change : Change
            Change object to be raised.
        p_inst : Instance = None
            Instance causing the change. If provided, the time stamp of the instance is taken over
            to the change.
        p_buffer : bool
            Change is buffered when set to True.
        """

        if p_change.tstamp is None:
            if p_inst is not None:
                p_change.tstamp = p_inst.tstamp
            else:
                p_change.tstamp = self.get_so().tstamp

        if p_buffer: self._buffer_change( p_change=p_change )

        if self.get_visualization(): 
            p_change.init_plot( p_figure=self._figure, 
                                 p_plot_settings=self.get_plot_settings() )

        self._raise_event( p_event_id = p_change.event_id,
                           p_event_object = p_change )


## -------------------------------------------------------------------------------------------------
    def _detect(self, p_inst: Instance) -> None:
        """
        Custom method for the main detection algorithm.

        Parameters
        ----------
        p_inst : Instance
            Instance that triggered the detection.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _triage( self, 
                 p_change: Change,
                 **p_kwargs ) -> bool:
        """
        Custom method for extended change triage.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        """
        This method is called by the stream task to process the incoming instance.

        Parameters
        ----------
        p_inst : InstDict
            The incoming instance to be processed.

        Returns
        -------
        None

        """

        # 0 Check whether the minimum number of instances has been reached
        if self._chk_num_inst:
            self._num_inst += len( p_inst )
            if self._num_inst < self._thrs_inst: return
            self._chk_num_inst = False


        # 1 Execution of the main detection algorithm        
        try:
            inst_type, inst = list(p_inst.values())[-1]
            if inst_type != InstTypeNew:
                inst = None
        except:
            inst = None

        self._detect( p_inst = inst )


        # 2 Clean-up loop ('triage')
        triage_list = []

        # 2.1 Collect changes to be deleted
        for change in self.changes.values():

            # 2.1.1 Apply custom triage method to each change
            if self._triage( p_change = change ):
                triage_list.append( change )

        # 2.2 Remove all obsolete changes from the triage list
        for change in triage_list:
            self._remove_change( p_change = change )

                 
## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        self._plot_ax_xlim = None
        self._plot_ax_ylim = None
        self._plot_ax_zlim = None

        super().init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)

        for change in self.changes.values():
            change.init_plot(p_figure=p_figure, p_plot_settings = p_plot_settings)
    

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

        for change in self.changes.values():
            change.update_plot( p_axlimits_changed = axlimits_changed,
                                 p_xlim = ax_xlim_new,
                                 p_ylim = ax_ylim_new,
                                 p_zlim = ax_zlim_new,
                                 **p_kwargs )
    

## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh: bool = True):

        if not self.get_visualization(): return

        # super().remove_plot(p_refresh=p_refresh)

        for change in self.changes.values():
            change.remove_plot(p_refresh=p_refresh)


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer):
        """
        Internal renormalization of all buffered changes. See method OATask.renormalize_on_event() 
        for further information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """

        for change in self.changes.values():
           change.renormalize( p_normalizer=p_normalizer )

