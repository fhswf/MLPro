## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.normalizers
## -- Module  : minmax.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-07  1.0.0     LSB      Creation/Release
## -- 2022-12-13  1.0.1     LSB      Refactoring
## -- 2022-12-20  1.0.2     DA       Refactoring
## -- 2022-12-20  1.0.3     LSB      Bugfix
## -- 2022-12-30  1.0.4     LSB      Bugfix
## -- 2023-01-12  1.1.0     LSB      Renormalizing plot data
## -- 2023-01-24  1.1.1     LSB      Bugfix
## -- 2023-02-13  1.1.2     LSB      Bugfix: Setting the default parameter update flag ot false
## -- 2023-04-09  1.2.0     DA       Class NormalizerZTransform: new methods _adapt(), _adapt_reverse()
## -- 2023-05-03  1.2.1     DA       Bugfix in NormalizerMinMax._update_plot_2d/3d/nd
## -- 2023-05-22  1.2.2     SY       Refactoring
## -- 2024-05-22  1.3.0     DA       Refactoring and splitting
## -- 2024-06-13  1.3.1     LSB      Bug Fix: Handling plot data with IDs
## -- 2024-07-12  1.3.2     LSB      Renormalization error
## -- 2024-10-29  1.3.3     DA       - Refactoring of NormalizerMinMax._adapt_on_event()
## --                                - Bugfix in NormalizerMinMax._update_plot_data_3d()
## -- 2024-12-16  1.4.0     DA       Method NormalizerMinMax._run(): little code tuning
## -- 2025-06-05  1.5.0     DA       Refactoring
## -- 2025-06-25  1.6.0     DA       Refactoring: p_inst -> p_instance/s
## -- 2025-06-30  2.0.0     DA       Refactoring: new parent OAStreamNormalizer
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.0 (2025-06-30)

This module provides implementation for adaptive normalizers for MinMax Normalization.
"""


from mlpro.bf.various import Log
from mlpro.bf.events import Event
from mlpro.bf.exceptions import Error
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math import normalizers as Norm
from mlpro.bf.streams import InstDict
from mlpro.oa.streams.basics import OAStreamTask
from mlpro.oa.streams.tasks.normalizers import OAStreamNormalizer



# Export list for public API
__all__ = [ 'NormalizerMinMax' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax (Norm.NormalizerMinMax, OAStreamNormalizer):
    """
    Class with functionality for adaptive normalization of instances using MinMax Normalization.

    Parameters
    ----------
    p_name: str = None,
        Optional name of the task.
    p_range_max:
        Processing range of the task, default is a Thread.
    p_ada : bool
        True if the task has adaptivity, default is true.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize:
        True for visualization, false by default.
    p_logging:
        Logging level of the task. Default is Log.C_LOG_ALL
    p_dst_boundaries : list = [-1,1]
        Explicit list of (low, high) destination boundaries. Default is [-1, 1].
    **p_kwargs:
        Additional task parameters
    """

    C_NAME = 'MinMax' 

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name: str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize:bool = False,
                  p_logging = Log.C_LOG_ALL,
                  p_dst_boundaries : list = [-1, 1],
                  **p_kwargs ):

        OAStreamNormalizer.__init__( self,
                                     p_name = p_name,
                                     p_range_max = p_range_max,
                                     p_ada = p_ada,
                                     p_duplicate_data = p_duplicate_data,
                                     p_visualize = p_visualize,
                                     p_logging=p_logging,
                                     **p_kwargs )

        Norm.NormalizerMinMax.__init__( self, 
                                        p_input_set = None,
                                        p_output_set = None,
                                        p_output_elem_cls = None,
                                        p_autocreate_elements = False,
                                        p_dst_boundaries = p_dst_boundaries,
                                        **p_kwargs )
        
        if p_visualize:
            self._plot_data_2d = None
            self._plot_data_3d = None
            self._plot_data_nd = None


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict):
        """
        Runs MinMax Normalizer task for normalizing stream instances.

        Parameters
        ----------
        p_instances : InstDict
            Instances to be processed
        """
        
        # Normalization of all incoming stream instances
        for (inst_type, inst) in p_instances.values():
            feature_data = inst.get_feature_data()    
            normalized_element = self.normalize( p_data = feature_data )
            feature_data.set_values(normalized_element.get_values())


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event( self, 
                         p_event_id:str, 
                         p_event_object:Event ) -> bool:
        """
        Custom method to adapt the MinMax normalizer parameters based on event raised by Boundary object for changed
        boundaries.

        Parameters
        ----------
        p_event_id: str
            Event id of the raised event

        p_event_obj: Event
            Event object that raises the corresponding event

        Returns
        -------
        bool
            Returns True, if the task has adapted. False otherwise.
        """

        adapted = self.update_parameters( p_boundaries = p_event_object.get_raising_object().get_boundaries() )
        if not adapted: return False

        if self._visualize:
            if self._plot_settings.view == PlotSettings.C_VIEW_2D:
                self._update_plot_data_2d()
            elif self._plot_settings.view == PlotSettings.C_VIEW_3D:
                self._update_plot_data_3d()
            elif self._plot_settings.view == PlotSettings.C_VIEW_ND:
                self._update_plot_data_nd()
            else:
                raise Error
            
            self._update_ax_limits = True
            self._recalc_ax_limits = True
            
        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_2d(self):
        """
        Updates the 2D plot data after parameter changes by renormalizing the existing points.
        """
        
        if not self._plot_2d_xdata: return

        self.renormalize( p_data = self._plot_2d_xdata, p_dim = 0 )
        self.renormalize( p_data = self._plot_2d_ydata, p_dim = 1 )


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_3d(self):
        """
        Method to update the 3d plot for Normalizer. Extended to renormalize the obsolete data on change of parameters.
        """

        if not self._plot_3d_xdata: return

        self.renormalize( p_data = self._plot_3d_xdata, p_dim = 0 )
        self.renormalize( p_data = self._plot_3d_ydata, p_dim = 1 )
        self.renormalize( p_data = self._plot_3d_zdata, p_dim = 2 )


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_nd(self):
        """
        Method to update the nd plot for Normalizer. Extended to renormalize the obsolete data on change of parameters.
        """

        if not self._plot_nd_plots: return

        for dim, plot_data in enumerate(self._plot_nd_plots):
            self.renormalize(p_data=plot_data[0], p_dim=dim)