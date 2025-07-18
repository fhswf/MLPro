## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.normalizers
## -- Module  : ztrans.py
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
## -- 2024-05-23  1.3.1     DA       Bugfix
## -- 2024-05-27  1.3.2     LSB      Fixed Plotting
## -- 2024-05-28  1.3.3     LSB      Fixing the plotting bugs
## -- 2024-05-28  1.3.4     LSB      Fixed the denormalizing method when zero std
## -- 2024-12-05  1.3.5     DA       Bugfix in method NormalizersZTransform._run()
## -- 2024-12-06  1.3.6     DA       Fixes and optimization 
## -- 2025-06-06  1.4.0     DA       Refactoring: p_inst -> p_instance/s
## -- 2025-06-25  2.0.0     DA       Refactoring, simplification, correction
## -- 2025-07-05  2.1.0     DA       Refactoring: new parent OAStreamNormalizer
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.1.0 (2025-07-05)

This module provides implementation for adaptive normalizers for ZTransformation
"""

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math import normalizers as Norm
from mlpro.bf.streams import Instance, InstDict

from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.normalizers import OAStreamNormalizer



# Export list for public API
__all__ = [ 'NormalizerZTransform' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTransform (Norm.NormalizerZTrans, OAStreamNormalizer):
    """
    Online-adaptive normalization of instances with Z-transformation.

    Parameters
    ----------
    p_name: str, optional
        Name of the task.
    p_range_max:
        Processing range of the task, default is a Thread.
    p_ada:
        True if the task has adaptivity, default is true.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize:
        True for visualization, false by default.
    p_logging:
        Logging level of the task. Default is Log.C_LOG_ALL
    **p_kwargs:
        Additional task parameters
    """

    C_NAME = 'ZTrans'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name: str = None,
                 p_range_max=OAStreamTask.C_RANGE_THREAD,
                 p_ada: bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        OAStreamNormalizer.__init__( self,
                                     p_name = p_name,
                                     p_range_max = p_range_max,
                                     p_ada = p_ada,
                                     p_duplicate_data = p_duplicate_data,
                                     p_visualize = p_visualize,
                                     p_logging=p_logging,
                                     **p_kwargs )

        Norm.NormalizerZTrans.__init__( self, 
                                        p_output_elem_cls = None,
                                        p_autocreate_elements = False,
                                        **p_kwargs )
        
        if p_visualize:
            self._plot_data_2d = None
            self._plot_data_3d = None
            self._plot_data_nd = None


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict):
        """
        Custom method to for run Z-transform task for normalizing new instances and denormalizing deleted instances.

        Parameters
        ----------
        p_instances : InstDict
            Stream instances to be processed

        """

        for inst_id, (inst_type, inst) in p_instances.items():

            feature_data = inst.get_feature_data()
            self.adapt( p_instances = { inst_id : (inst_type, inst) } )
            feature_data.set_values( p_values = self.normalize(feature_data).get_values() )
            
            # Udpdate of plot data
            self._update_plot_data()


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_instance_new : Instance) -> bool:
        """
        Custom method to for adapting of Z-transform parameters on new instances.

        Parameters
        ----------
        p_instance_new: Instance
            Instance to be adapted on.

        Returns
        -------
        adapted : bool
            Returns True, if task has adapted.

        """

        return self.update_parameters( p_data_new = p_instance_new.get_feature_data() )


## -------------------------------------------------------------------------------------------------
    def _adapt_reverse(self, p_instance_del : Instance) -> bool:
        """
        Custom method to for adapting of Z-transform parameters on deleted instances.

        Parameters
        ----------
        p_instance_del: Instance
            Instance to be adapted on.

        Returns
        -------
        adapted : bool
            Returns True, if task has adapted.

        """

        self.update_parameters( p_data_del = p_instance_del.get_feature_data() )

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


## -------------------------------------------------------------------------------------------------
    def _update_plot_data(self):
        """
        Updates the plot data.
        """

        if not self.get_visualization(): return
        view = self.get_plot_settings().view

        if view == PlotSettings.C_VIEW_2D:
            self._update_plot_data_2d()
        elif view == PlotSettings.C_VIEW_3D:
            self._update_plot_data_3d()
        elif view == PlotSettings.C_VIEW_ND:
            self._update_plot_data_nd()

        self._update_ax_limits = True
        self._recalc_ax_limits = True
