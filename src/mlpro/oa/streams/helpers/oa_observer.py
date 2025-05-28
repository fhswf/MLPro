## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.helper
## -- Module  : oa_observer.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-28  0.1.0     DA       New class OAObserver for adaptation observation
## -------------------------------------------------------------------------------------------------


from mlpro.bf.various import Log, KWArgs
from mlpro.bf.plot import PlotSettings, Plottable

from mlpro.oa.streams import OAStreamAdaptation




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAObserver (Log, Plottable, KWArgs):
    """
    This class observes adaptations of particular oa stream tasks. Its event handler method can be 
    registered for adaptation events of a stream task.

    Parameters
    ----------
    
    """

    C_TYPE              = 'Helper'
    C_NAME              = 'OA Observer'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = True
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_visualize : bool = True,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        Log.__init__(self, p_logging = p_logging)
        Plottable.__init__(self, p_visualize = p_visualize )
        KWArgs.__init__(self, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def event_handler(self, p_event_id, p_event_object : OAStreamAdaptation ):
        self.log( Log.C_LOG_TYPE_W, 'Task "' + p_event_object.get_raising_object().get_name() + '" performed an adaptation of type "' + str(p_event_object.subtype) + '" on ' + str(p_event_object.num_inst) + ' instances' )
        self.update_plot()


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure, p_settings):
        return super()._init_plot_nd(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings, **p_kwargs):
        return super()._update_plot_nd(p_settings, **p_kwargs)