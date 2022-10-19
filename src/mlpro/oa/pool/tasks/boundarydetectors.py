## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.pool.tasks.boundarydetectors
## -- Module  : boundarydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-17  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-10-17)
This module provides pool of boundary detector object further used in the context of online adaptivity.
"""

from mlpro.oa.pool.tasks.windows import *
from mlpro.oa import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundaryDetector(OATask):
    """
    This is the base class for Boundary Detector object. It raises event when a change in the current boundaries is
    detected based on the new data instances
    """

    C_NAME = 'Boundary Detector'


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
        boundaries = []
        current_set = p_inst_new[0].get_related_set()
        for i in current_set.get_dim_ids():
            boundaries.append(current_set.get_dim(i).get_boundaries())
        for inst in p_inst_new:
           for i,value in enumerate(inst.get_values()):
               if value < boundaries[i][0] or value > boundaries[i][1]:
                    return True
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
        self.adapt(p_inst_new, p_inst_del)


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_obj:Event):
        """
        Event handler for Boundary Detector that adapts if the related event is raised
        Parameters
        ----------
            p_event_id
                The event id related to the adaptation.
            p_event_obj
                The event object related to the raised event.
        """
        self.log(self.C_LOG_TYPE_I, 'Event"'+p_event_id+'"raised by', p_event_obj)
        data = p_event_obj.get_data()
        p_inst_new = data['p_inst_new']
        p_inst_del = data['p_inst_del']
        return self._adapt(p_inst_new, p_inst_del)
