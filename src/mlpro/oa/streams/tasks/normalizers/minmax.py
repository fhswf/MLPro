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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.3 (2024-10-29)

This module provides implementation for adaptive normalizers for MinMax Normalization.
"""


from mlpro.oa.streams.basics import *
from mlpro.bf.math import normalizers as Norm



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax (OAStreamTask, Norm.NormalizerMinMax):
    """
    Class with functionality for adaptive normalization of instances using MinMax Normalization.

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

    C_NAME = 'Normalizer MinMax' 

## -------------------------------------------------------------------------------------------------
    def __init__(self,p_name: str = None,
                  p_range_max = StreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize:bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs):

        OAStreamTask.__init__(self,
                        p_name = p_name,
                        p_range_max = p_range_max,
                        p_ada = p_ada,
                        p_duplicate_data = p_duplicate_data,
                        p_visualize = p_visualize,
                        p_logging=p_logging,
                        **p_kwargs )


        Norm.NormalizerMinMax.__init__(self)
        self._parameters_updated:bool = None

        if p_visualize:
            self._plot_data_2d = None
            self._plot_data_3d = None
            self._plot_data_nd = None


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst:InstDict):
        """
        Runs MinMax Normalizer task for normalizing stream instances.

        Parameters
        ----------
        p_inst : InstDict
            Instances to be processed
        """
        
        # Normalization of all incoming stream instances (order doesn't matter)
        for ids, (inst_type, inst) in p_inst.items():
            if self._param is None:
                self.update_parameters( p_set = inst.get_feature_data().get_related_set() )
            normalized_element = self.normalize(inst.get_feature_data())
            inst.get_feature_data().set_values(normalized_element.get_values())


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
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
        adapted: bool
            Returns True, if the task has adapted. False otherwise.
        """

        set = p_event_object.get_raising_object().get_related_set()

        self.update_parameters(set)

        if self._visualize:
            if self._plot_settings.view == PlotSettings.C_VIEW_2D:
                self._update_plot_data_2d()
            elif self._plot_settings.view == PlotSettings.C_VIEW_3D:
                self._update_plot_data_3d()
            elif self._plot_settings.view == PlotSettings.C_VIEW_ND:
                self._update_plot_data_nd()
            else:
                raise Error

        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_2d(self):
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
        try:
            if len(self._plot_2d_xdata) != 0 and len(self._plot_2d_xdata):
                if ( self._plot_data_2d is None ) or ( len(self._plot_2d_xdata) > self._plot_data_2d.shape[0] ):
                    self._plot_data_2d = np.zeros((len(self._plot_2d_xdata),2))
                ids = []
                for i, (id, val) in enumerate(self._plot_2d_xdata.items()):
                    ids.extend([id])
                    self._plot_data_2d[i][0] = self._plot_2d_xdata[id]
                    self._plot_data_2d[i][1] = self._plot_2d_ydata[id]

                plot_data_renormalized = self.renormalize(self._plot_data_2d)

                self._plot_2d_xdata = {}
                self._plot_2d_ydata = {}

                for i, data_2d in enumerate(plot_data_renormalized):
                    self._plot_2d_xdata[ids[i]] = data_2d[0]
                    self._plot_2d_ydata[ids[i]] = data_2d[1]


                self._parameters_updated = False
        except:
            raise Error


## -------------------------------------------------------------------------------------------------
    def _update_plot_data_3d(self):
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
        try:
            if len(self._plot_3d_xdata) != 0:
                if ( self._plot_data_3d is None ) or ( len(self._plot_3d_xdata) > self._plot_data_3d.shape[0] ):
                    self._plot_data_3d = np.zeros((len(self._plot_3d_xdata),3))

                ids = []
                for i, (id,val) in enumerate(self._plot_3d_xdata.items()):
                    ids.extend([id])
                    self._plot_data_3d[i][0] = self._plot_3d_xdata[id]
                    self._plot_data_3d[i][1] = self._plot_3d_ydata[id]
                    self._plot_data_3d[i][2] = self._plot_3d_zdata[id]

                plot_data_renormalized = self.renormalize(self._plot_data_3d)

                self._plot_3d_xdata = {}
                self._plot_3d_ydata = {}
                self._plot_3d_zdata = {}

                for i, data_3d in enumerate(plot_data_renormalized):
                    self._plot_3d_xdata[ids[i]] = data_3d[0]
                    self._plot_3d_ydata[ids[i]] = data_3d[1]
                    self._plot_3d_zdata[ids[i]] = data_3d[2]
        except:
            pass



## -------------------------------------------------------------------------------------------------
    def _update_plot_data_nd(self):
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

        try:
            if self._plot_nd_plots:
                if ( self._plot_data_nd is None ) or ( len(self._plot_nd_plots[0][0]) > self._plot_data_nd.shape[0] ):
                        self._plot_data_nd = np.zeros((len(self._plot_nd_plots[0][0]),len(self._plot_nd_plots)))
                ids = []
                for j in range(len(self._plot_nd_plots)):
                    for i in range(len(self._plot_nd_plots[0][0])):
                        self._plot_data_nd[i][j] = self._plot_nd_plots[j][0][i]

                plot_data_renormalized = self.renormalize(self._plot_data_nd)

                for j in range(len(self._plot_nd_plots)):
                    self._plot_nd_plots[j][0] = list(k[j] for k in plot_data_renormalized)
        except:
            pass