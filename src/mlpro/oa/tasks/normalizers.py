## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks.normalizers
## -- Module  : normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-07  1.0.0     LSB      Creation/Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-12-07)
This module provides implementation for adaptive normalizers for MinMax Normalization and ZTransformation
"""


from mlpro.oa.models import *
from mlpro.bf.math import normalizers as Norm



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax(OATask, Norm.NormalizerMinMax):
    """
    Class with functionality for adaptive normalization of instances using MinMax Normalization
    """


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
        for i,inst in enumerate(p_inst_new):
            normalized_element = self.normalize(inst)
            p_inst_new[i] = normalized_element

        for j, del_inst in enumerate(p_inst_del):
            denormalized_element = self.denormalize(del_inst)
            p_inst_del[j] = denormalized_element



## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_obj:Event) -> bool:
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
        adapted = False
        try:
            self.update_parameters(p_event_obj.get_raising_object(),)
            adapted = True
        except:
            pass
        return adapted





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTransform(OATask, Norm.NormalizerZTrans):
    """
    Class with functionality of adaptive normalization of instances with Z-Transformation
    """


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
            normalized_element = self.normalize(inst)
            p_inst_new[i] = normalized_element

        for i,del_inst in enumerate(p_inst_del):
            denormalized_element = self.denormalize(del_inst)
            p_inst_del[i] = denormalized_element


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
                self.update_parameters(p_data_new=inst)

            # 2. Update parameters based on deleted elements
            for del_inst in p_inst_del:
                self.update_parameters(p_data_del=del_inst)

            adapted = True

        except: pass

        return adapted
