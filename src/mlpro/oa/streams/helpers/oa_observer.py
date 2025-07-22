## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.helper
## -- Module  : oa_observer.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-28  0.1.0     DA       New class OAObserver for adaptation observation
## -- 2025-06-04  0.2.0     DA       Refactoring
## -- 2025-07-16  0.3.0     DA       Refactoring: new parent class StreamTaskHelper
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-07-16)

This module provides the OAObserver class to be used for observation and visualization of stream
adaptation events.

"""


from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import StreamTaskHelper

from mlpro.oa.streams import OAStreamAdaptation, OAStreamAdaptationType, OAStreamTask



# Export list for public API
__all__ = [ 'OAObserver' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAObserver (StreamTaskHelper):
    """
    This class observes adaptations of particular oa stream tasks. Its event handler method can be 
    registered for adaptation events of a stream task.

    Parameters
    ----------
    p_related_task : StreamTask
        Related stream task that raises events.
    p_no_per_task : int = 0
        Helper number of the task. This is used to distinguish between multiple helpers for the same task.
    p_annotation : str = None
        Optional annotation for the helper.
    p_window_title: str = None
        Optional window title for the helper. If None, a default title is generated. 
    p_logarithmic_plot : bool = True
        If True, the y-axis of the plot is logarithmic.
    p_visualize : bool = True
        If True, the plot is visualized.
    p_logging : int = Log.C_LOG_ALL
        Logging level for this helper. Default is Log.C_LOG_ALL.
    p_kwargs : dict
        Further keyword arguments for the helper.
    """

    C_NAME              = 'OA Observer'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = True
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

    C_ADAPTATION_COLORS = { OAStreamAdaptationType.FORWARD : 'blue',
                            OAStreamAdaptationType.REVERSE : 'red',
                            OAStreamAdaptationType.EVENT   : 'green',
                            OAStreamAdaptationType.RENORM  : 'purple' }

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_related_task : OAStreamTask,
                  p_no_per_task : int = 0,
                  p_annotation : str = None,
                  p_window_title: str = None,
                  p_logarithmic_plot : bool = True,
                  p_filter_subtypes : list = [],
                  p_visualize : bool = True,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        

        self._logarithmic_plot  = p_logarithmic_plot
        self._filter_subtypes   = p_filter_subtypes

        self.stat_adaptation_events = {}
        self.stat_adaptation_inst   = {}
        
        super().__init__( p_related_task = p_related_task,
                          p_event_ids = [ OAStreamTask.C_EVENT_ADAPTED ],
                          p_no_per_task = p_no_per_task,
                          p_annotation = p_annotation,
                          p_window_title = p_window_title,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def _event_handler(self, p_event_id, p_event_object : OAStreamAdaptation ):

        # 0 Intro
        if ( len(self._filter_subtypes) > 0 ) and ( not p_event_object.subtype in self._filter_subtypes ): return
        self.log( Log.C_LOG_TYPE_W, 'Task "' + p_event_object.get_raising_object().get_name() + '" performed an adaptation of type "' + str(p_event_object.subtype) + '" on ' + str(p_event_object.num_inst) + ' instances' )


        # 1 Update statistics
        self._update_statistics( p_event_object = p_event_object )


        # 2 Update plot
        self.update_plot( p_event_object = p_event_object )


## -------------------------------------------------------------------------------------------------
    def _update_statistics( self, p_event_object : OAStreamAdaptation ):
        
        try:
            self.stat_adaptation_events[p_event_object.subtype] += 1
            self.stat_adaptation_inst[p_event_object.subtype]   += p_event_object.num_inst
        except:
            self.stat_adaptation_events[p_event_object.subtype] = 1
            self.stat_adaptation_inst[p_event_object.subtype]   = p_event_object.num_inst


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure = None,
                   p_plot_settings: PlotSettings = None,
                   p_window_title: str = None ) -> bool:
        
        super().init_plot( p_figure = p_figure,
                           p_plot_settings = p_plot_settings,
                           p_window_title = p_window_title )
        
        axes = self.get_plot_settings().axes
        axes.legend(title='Adaptations')
        axes.set_xlabel('Time index')
        axes.set_ylabel('Adapted instances')     

        if self._logarithmic_plot:
            axes.set_yscale('log')

        axes.figure.canvas.draw()    


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd( self, 
                       p_figure, 
                       p_settings : PlotSettings ):
        
        super()._init_plot_nd(p_figure, p_settings)
        self._vlines = {}
            

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd( self, 
                         p_settings : PlotSettings, 
                         p_event_object : OAStreamAdaptation,
                         **p_kwargs ) -> bool:

        try:
            vlines = self._vlines[p_event_object.subtype]
            label  = None
            update_legend = False
        except:
            vlines = []
            self._vlines[p_event_object.subtype] = vlines
            label = p_event_object.subtype
            update_legend = True


        color = self.C_ADAPTATION_COLORS[p_event_object.subtype]

        vlines.append( p_settings.axes.vlines( x = p_event_object.tstamp,
                                               ymin = 0,
                                               ymax = p_event_object.num_inst,
                                               colors = color,
                                               label = label ) )
        
        if update_legend:
            p_settings.axes.legend(title='Adaptations')

        return True