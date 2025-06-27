## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.helper
## -- Module  : change_observer.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-24  0.1.0     DA/DS    New class ChangeObserver for change observation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-06-24)

This module provides the ChangeObserver class to be used for observation and visualization of stream
adaptation events.

"""

from mlpro.bf.various import Log, KWArgs
from mlpro.bf.exceptions import ParamError
from mlpro.bf.plot import PlotSettings

from mlpro.oa.streams.tasks.changedetectors import Change 
from mlpro.oa.streams import OAStreamHelper, OAStreamTask



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ChangeObserver (OAStreamHelper, Log, KWArgs):
    """
    This class observes adaptations of particular oa stream tasks. Its event handler method can be 
    registered for adaptation events of a stream task.

    Parameters
    ----------
    
    """

    C_TYPE              = 'Helper'
    C_NAME              = 'Event Observer'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = True
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

    C_ANOMALY_COLORS = [ 'blue',
                         'red',
                         'green',
                         'purple', 
                         'yellow',
                         'orange' ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_related_task : OAStreamTask,
                  p_change_types : list = [],
                  p_logarithmic_plot : bool = True,
                  p_visualize : bool = True,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        OAStreamHelper.__init__(self, p_visualize = p_visualize )
        Log.__init__(self, p_logging = p_logging)
        KWArgs.__init__(self, **p_kwargs)

        self._related_task      = p_related_task
        self._logarithmic_plot  = p_logarithmic_plot
        self._change_types      = p_change_types
        self._color_max         = len( ChangeObserver.C_ANOMALY_COLORS )
        self._color_id          = 0
        self._change_colors     = {}

        self.stat_change_events = {}

        if len(self._change_types) == 1:
            self._related_task.register_event_handler( p_event_id = self._change_types[0].get_event_id( p_status = True ),
                                                       p_event_handler = self._event_handler )
            
            self._change_colors[self._change_types[0].get_event_id( p_status = True)] = self.C_ANOMALY_COLORS[0]
            
            self._related_task.register_event_handler( p_event_id = self._change_types[0].get_event_id( p_status = False ),
                                                       p_event_handler = self._event_handler )
            
            self._change_colors[self._change_types[0].get_event_id( p_status = False)] = self.C_ANOMALY_COLORS[1]

        elif len(self._change_types) > 1:
            for anomaly_type in self._change_types:
                self._related_task.register_event_handler( p_event_id = anomaly_type.get_event_id( p_status = True),
                                                           p_event_handler = self._event_handler )
                self._change_colors[anomaly_type.get_event_id( p_status = True)] = self.C_ANOMALY_COLORS[self._color_id]
                self._color_id = ( self._color_id + 1 ) & self._color_max
                
        else:
            raise ParamError( 'ChangeObserver: No change types given for observation.' )


## -------------------------------------------------------------------------------------------------
    def _event_handler(self, p_event_id, p_event_object : Change ):

        # 0 Intro
        self.log( Log.C_LOG_TYPE_W, 'Task "' + p_event_object.get_raising_object().get_name() + '" raised a change of type "' + str(p_event_id) + '"' )


        # 1 Update statistics
        self._update_statistics( p_event_object = p_event_object )


        # 2 Update plot
        self.update_plot( p_event_object = p_event_object )


## -------------------------------------------------------------------------------------------------
    def _update_statistics( self, p_event_object : Change ):
        
        try:
            self.stat_change_events[p_event_object.get_event_id()] += 1
        except:
            self.stat_change_events[p_event_object.get_event_id()] = 1


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure = None,
                   p_plot_settings: PlotSettings = None,
                   p_window_title: str = None ) -> bool:
        
        if p_window_title is None:
            window_title = 'Change Observer for Task "' + self._related_task.get_name() + '"'
        else:
            window_title = p_window_title

        super().init_plot( p_figure = p_figure,
                           p_plot_settings = p_plot_settings,
                           p_window_title = window_title )
        
        axes = self.get_plot_settings().axes
        axes.legend(title='Changes')
        axes.set_xlabel('Time index')
        # axes.set_ylabel('Adapted instances')     

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
                         p_event_object : Change,
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


        color = self._change_colors[p_event_object.get_event_id()]

        vlines.append( p_settings.axes.vlines( x = p_event_object.tstamp,
                                               ymin = 0,
                                               ymax = 1,
                                               colors = color,
                                               label = label ) )
        
        if update_legend:
            p_settings.axes.legend(title='Changes')

        return True