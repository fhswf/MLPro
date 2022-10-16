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





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundaryDetector(OATask):
    """
    This is the base class for Boundary Detector object. It raises event when a change in the current boundaries is
    detected based on the new data instances
    """
    pass

    C_NAME = 'Boundary Detector'


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:list):
        # if p_inst_new >
        pass


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):

        self.adapt(p_inst_new, p_inst_del)


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id, p_event_obj:Event):

        pass
