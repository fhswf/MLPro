## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.pool.tasks.boundarydetectors
## -- Module  : boundarydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-17  0.0.0     LSB      Creation
## -- 2022-10-21  1.0.0     LSB      Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-10-21)
This module provides pool of boundary detector object further used in the context of online adaptivity.
"""

from mlpro.oa.pool.tasks.windows import *
from mlpro.oa import *
from typing import Union, Iterable





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundaryDetector(OATask):
    """
    This is the base class for Boundary Detector object. It raises event when a change in the current boundaries is
    detected based on the new data instances
    """

    C_NAME = 'Boundary Detector'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scaler:Union[float, Iterable], p_logging = Log.C_LOG_ALL):

        self._scaler = p_scaler
        super().__init__(p_logging = p_logging)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:list, p_inst_del:list):
        """
        Method to check if the new instances exceed the current boundaries of the Set.

        Parameters
        ----------
            p_inst_new:list
                List of new instance/s added to the workflow

        Returns
        -------
            bool
                Returns true if there is a change of boundaries, false otherwise.
        """
        dim = []
        for id in p_inst_new[0].get_related_set().get_dim_ids():
            dim.append(p_inst_new[0].get_related_set().get_dim(id))
        for inst in p_inst_new:
            for i,value in enumerate(inst.get_values()):
                boundary = dim[i].get_boundaries()[0]
                if value < boundary[0]:
                    dim[i].set_boundaries([value, boundary[1]])
                    return True
                elif value > boundary[1]:
                    dim[i].set_boundaries([boundary[0],value])
                    return True
                else:
                    return False


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Method to run the boundary detector task

        Parameters
        ----------
            p_inst_new:list
                List of new instance/s added to the workflow
            p_inst_del:list
                List of old obsolete instance/s removed from the workflow
        """
        if p_inst_new and not p_inst_del:
            self.adapt(p_inst_new, p_inst_del)


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_obj:Event):
        """
        Event handler for Boundary Detector that adapts if the related event is raised
        Parameters
        ----------
            p_event_id
                The event id related to the adaptation.
            p_event_obj:Event
                The event object related to the raised event.

        Returns
        -------
            bool
                Returns true if adapted, false otherwise.
        """
        adapted = False
        try:
            boundaries = self._scaler*p_event_obj.get_raising_object().get_boundaries()
            dims = [p_event_obj.get_data()["p_set"].get_dim(i) for i in p_event_obj.get_data()["p_set"].get_dim_ids()]
            for i,dim in enumerate(dims):
                if any(dim.get_boundaries() != boundaries[i]):
                    dim.set_boundaries([boundaries[i]])
                    adapted = True
        except: raise ImplementationError("Event not raised by a window")
        return adapted