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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.4 (2024-05-27)

This module provides implementation for adaptive normalizers for ZTransformation
"""


from mlpro.oa.streams.basics import *
from mlpro.bf.math import normalizers as Norm



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTransform (OAStreamTask, Norm.NormalizerZTrans):
    """
    Online adaptive normalization of instances with Z-Transformation

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
    p_kwargs:
        Additional task parameters
    """

    C_NAME = 'Normalizer Z Transform'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name: str = None,
                 p_range_max=StreamTask.C_RANGE_THREAD,
                 p_ada: bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        OAStreamTask.__init__(self,
            p_name=p_name,
            p_range_max=p_range_max,
            p_ada=p_ada,
            p_duplicate_data = p_duplicate_data,
            p_visualize = p_visualize,
            p_logging=p_logging,
            **p_kwargs)

        Norm.NormalizerZTrans.__init__(self)
        self._parameters_updated:bool = None
        self._test_data = None
        if p_visualize:
            self._plot_data_2d = None
            self._plot_data_3d = None
            self._plot_data_nd = None

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        """
        Custom method to for run Z-transform task for normalizing new instances and denormalizing deleted instances.

        Parameters
        ----------
        p_inst : InstDict
            Stream instances to be processed

        """

        # 1 Online update of transformation parameters
        self.adapt( p_inst = p_inst )
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):

            feature_data = inst.get_feature_data()

            if self._param is None:
                if inst_type == InstTypeNew:
                    self.update_parameters( p_data_new = feature_data )
                    self.update_plot_data()
                else:
                    self.update_parameters( p_data_del = feature_data )
                    self.update_plot_data()
            feature_data.set_values( p_values = self.normalize(feature_data).get_values() )


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new : Instance) -> bool:
        """
        Custom method to for adapting of Z-transform parameters on new instances.

        Parameters
        ----------
        p_inst_new: Instance
            Instance to be adapted on.

        Returns
        -------
        adapted : bool
            Returns True, if task has adapted.

        """

        self.update_parameters( p_data_new = p_inst_new.get_feature_data() )
        self.update_plot_data()
        self._parameters_updated = True

        return True


## -------------------------------------------------------------------------------------------------
    def _adapt_reverse(self, p_inst_del:Instance) -> bool:
        """
        Custom method to for adapting of Z-transform parameters on deleted instances.

        Parameters
        ----------
        p_inst_del: Instance
            Instance to be adapted on.

        Returns
        -------
        adapted : bool
            Returns True, if task has adapted.

        """

        self.update_parameters( p_data_del = p_inst_del.get_feature_data() )
        self.update_plot_data()
        self._parameters_updated = True

        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_2d(self):
        """
        Renormalizing the plot data.
        """
        if self._parameters_updated and (len(self._plot_2d_xdata) != 0):

            if self._parameters_updated and (len(self._plot_2d_xdata) != 0):

                if (self._plot_data_2d is None) or (len(self._plot_2d_xdata) > self._plot_data_2d.shape[0]):
                    self._plot_data_2d = np.zeros((len(self._plot_2d_xdata), 2))
                    self._parameters_updated = False
                    # return

            for i in range(len(self._plot_2d_xdata)):
                self._plot_data_2d[i][0] = self._plot_2d_xdata[i]
                self._plot_data_2d[i][1] = self._plot_2d_ydata[i]

            plot_data_renormalized = self.renormalize(self._plot_data_2d)

            for i, data_2d in enumerate(plot_data_renormalized):
                self._plot_2d_xdata[i] = data_2d[0]
                self._plot_2d_ydata[i] = data_2d[1]

            self._parameters_updated = False


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d( self,
                         p_settings : PlotSettings,
                         p_inst : InstDict,
                         **p_kwargs ):
        """
        Updates the 2d plot for Normalizer. Extended to renormalize the obsolete data on change of parameters.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.
        """

        self.update_plot_data()

        OAStreamTask._update_plot_2d( self,
                                p_settings = p_settings,
                                p_inst = p_inst,
                                **p_kwargs )

## -------------------------------------------------------------------------------------------------
    def _update_plot_data_3d(self):
        if self._parameters_updated and (len(self._plot_3d_xdata) != 0):

            if (self._plot_data_3d is None) or (len(self._plot_3d_xdata) > self._plot_data_3d.shape[0]):
                self._plot_data_3d = np.zeros((len(self._plot_3d_xdata), 3))

            for i in range(len(self._plot_3d_xdata)):
                self._plot_data_3d[i][0] = self._plot_3d_xdata[i]
                self._plot_data_3d[i][1] = self._plot_3d_ydata[i]
                self._plot_data_3d[i][2] = self._plot_3d_zdata[i]

            plot_data_renormalized = self.renormalize(self._plot_data_3d)

            self._plot_3d_xdata = {}
            self._plot_3d_ydata = {}
            self._plot_3d_zdata = {}

            for i, data_3d in enumerate(plot_data_renormalized):
                self._plot_3d_xdata[i] = data_3d[0]
                self._plot_3d_ydata[i] = data_3d[1]
                self._plot_3d_zdata[i] = data_3d[2]

            self._parameters_updated = False

## -------------------------------------------------------------------------------------------------
    def _update_plot_3d( self,
                         p_settings : PlotSettings,
                         p_inst : InstDict,
                         **p_kwargs ):
        """
        Method to update the 3d plot for Normalizer. Extended to renormalize the obsolete data on change of parameters.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.

        """

        self._update_plot_data_3d()

        OAStreamTask._update_plot_3d( self,
                                p_settings = p_settings,
                                p_inst = p_inst,
                                **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_nd(self):
        if self._parameters_updated and self._plot_nd_plots:
            if (len(self._plot_nd_plots[0][0])) != 0:

                if (self._plot_data_nd is None) or (len(self._plot_nd_plots[0][0]) > self._plot_data_nd.shape[0]):
                    self._plot_data_nd = np.zeros((len(self._plot_nd_plots[0][0]), len(self._plot_nd_plots)))

                for j in range(len(self._plot_nd_plots)):
                    for i in range(len(self._plot_nd_plots[0][0])):
                        self._plot_data_nd[i][j] = self._plot_nd_plots[j][0][i]

                plot_data_renormalized = self.renormalize(self._plot_data_nd)

                for j in range(len(self._plot_nd_plots)):
                    self._plot_nd_plots[j][0] = list(k[j] for k in plot_data_renormalized)

                self._parameters_updated = False


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd( self,
                         p_settings : PlotSettings,
                         p_inst : InstDict,
                         **p_kwargs ):
        """

        Method to update the nd plot for Normalizer. Extended to renormalize the obsolete data on change of parameters.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.
        """

        self._update_plot_data_nd()

        OAStreamTask._update_plot_nd( self,
                                p_settings = p_settings,
                                p_inst = p_inst,
                                **p_kwargs )



## -------------------------------------------------------------------------------------------------
    def update_plot_data(self):
        """
        Updates the plot data.
        """
        try:
            self._update_plot_data_2d()
        except:
            pass
        try:
            self._update_plot_data_3d()
        except:
            pass
        try:
            self._update_plot_data_nd()
        except:
            pass