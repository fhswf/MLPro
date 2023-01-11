## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks.normalizers
## -- Module  : normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-07  1.0.0     LSB      Creation/Release
## -- 2022-12-13  1.0.1     LSB      Refactoring
## -- 2022-12-20  1.0.2     DA       Refactoring
## -- 2022-12-20  1.0.3     LSB      Bug fix
## -- 2022-12-30  1.0.4     LSB      Bug fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2022-12-30)

This module provides implementation for adaptive normalizers for MinMax Normalization and ZTransformation
"""


from mlpro.oa.models import *
from mlpro.bf.math import normalizers as Norm



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax(OATask, Norm.NormalizerMinMax):
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

        OATask.__init__(self,
                        p_name = p_name,
                        p_range_max = p_range_max,
                        p_ada = p_ada,
                        p_duplicate_data = p_duplicate_data,
                        p_visualize = p_visualize,
                        p_logging=p_logging,
                        **p_kwargs )

        Norm.NormalizerMinMax.__init__(self)
        self._parameters_updated:bool = True


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Custom method to for run MinMax Normalizer task for normalizing new instances and denormalizing deleted
        instances.

        Parameters
        ----------
        p_inst_new: list
            List of new instances in the workflow
        p_inst_del: list
            List of deleted instances in the workflow
        """

        if ((p_inst_new is None) or (len(p_inst_new) == 0)
                and ((p_inst_del is None) or len(p_inst_del) ==0)) : return

        for i,inst in enumerate(p_inst_new):
            if self._param is None:
                self.update_parameters(inst.get_feature_data().get_related_set())
            normalized_element = self.normalize(inst.get_feature_data())
            inst.get_feature_data().set_values(normalized_element.get_values())

        for j, del_inst in enumerate(p_inst_del):
            if self._param is None:
                self.update_parameters(del_inst.get_feature_data().get_related_set())
            normalized_element = self.normalize(del_inst.get_feature_data())
            del_inst.get_feature_data().set_values(normalized_element.get_values())


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

        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d( self,
                         p_settings : PlotSettings,
                         p_inst_new : list,
                         p_inst_del : list,
                         **p_kwargs ):
        """

        Parameters
        ----------
        p_settings
        p_inst_new
        p_inst_del
        p_kwargs

        Returns
        -------

        """

        if self._parameters_updated:
            plot_data = np.zeros((2,len(self._plot_2d_xdata)))

            for i in range(len(self._plot_2d_xdata)):
                plot_data[0][i] = self._plot_2d_xdata[i]
                plot_data[0][i] = self._plot_2d_ydata[i]

            plot_data_renormalized = self.renormalize(plot_data)

            self._plot_2d_xdata = list(j for j in plot_data_renormalized[0])
            self._plot_2d_ydata = list(j for j in plot_data_renormalized[1])

        OATask._update_plot_2d(self, p_settings = p_settings,
                               p_inst_new = p_inst_new,
                               p_inst_del = p_inst_del,
                               **p_kwargs)






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTransform(OATask, Norm.NormalizerZTrans):
    """
    Class with functionality of adaptive normalization of instances with Z-Transformation

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

        OATask.__init__(self,
            p_name=p_name,
            p_range_max=p_range_max,
            p_ada=p_ada,
            p_duplicate_data = p_duplicate_data,
            p_visualize = p_visualize,
            p_logging=p_logging,
            **p_kwargs)

        Norm.NormalizerZTrans.__init__(self)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Custom method to for run Z-transform task for normalizing new instances and denormalizing deleted instances.

        Parameters
        ----------
        p_inst_new: list
            List of new instances in the workflow
        p_inst_del: list
            List of deleted instances in the workflow

        """
        self.adapt(p_inst_new=p_inst_new, p_inst_del=p_inst_del)

        for i, inst in enumerate(p_inst_new):
            if self._param is None:
                self.update_parameters(p_data_new=inst.get_feature_data())
            normalized_element = self.normalize(inst.get_feature_data())
            inst.get_feature_data().set_values(normalized_element.get_values())

        for i,del_inst in enumerate(p_inst_del):
            if self._param is None:
                self.update_parameters(p_data_del=del_inst.get_feature_data())
            normalized_element = self.normalize(del_inst.get_feature_data())
            del_inst.get_feature_data().set_values(normalized_element.get_values())


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:list, p_inst_del:list) -> bool:
        """
        Custom method to for adapting of Z-transform parameters on new and deleted instances.

        Parameters
        ----------
        p_inst_new: list
            List of new instances in the workflow
        p_inst_del: list
            List of deleted instances in the workflow

        Returns
        -------
        adapted : bool
            Returns True, if task has adapted.

        """

        adapted = False
        try:
            # 1. Update parameters based on new elements
            for inst in p_inst_new:
                self.update_parameters(p_data_new=inst.get_feature_data())

            # 2. Update parameters based on deleted elements
            for del_inst in p_inst_del:
                self.update_parameters(p_data_del=del_inst.get_feature_data())

            adapted = True

        except: pass

        return adapted
