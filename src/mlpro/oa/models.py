## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa
## -- Module  : models.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-04  0.0.0     DA       Creation
## -- 2022-10-09  0.1.0     DA       Initial class definitions
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2022-10-09)

Template classes for online machine learning.
"""


from mlpro.bf.various import Log
from mlpro.bf.streams import *
import mlpro.bf.mt as mt
import mlpro.bf.ml as ml
import mlpro.sl.models as sl




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAShared (mt.Shared):
    """
    Template class for a shared memory. 
    """ 
    
    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OATask (ml.MLTask):
    """
    ...
    """

    C_TYPE      = 'OA-Task'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max=ml.MLTask.C_RANGE_THREAD, 
                  p_ada=True, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        super().__init__( p_name=p_name, 
                          p_range_max=p_range_max, 
                          p_autorun=ml.MLTask.C_AUTORUN_NONE, 
                          p_class_shared=None, 
                          p_buffer_size=0, 
                          p_ada=p_ada, 
                          p_logging=p_logging, 
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def run(self, p_inst_new:list, p_inst_del:list, p_range:int = None, p_wait: bool = False):
        """
        Executes the task specific actions implemented in custom method _run(). At the end event
        C_EVENT_FINISHED is raised to start subsequent actions (p_wait=True).

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be processed.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that the maximum
            range defined during instantiation is taken. Oterwise the minimum range of both is taken.
        p_wait : bool
            If True, the method waits until all (a)synchronous tasks are finished.
        p_kwargs : dict
            Further parameters handed over to custom method _run().
        """

        super().run(p_range=p_range, p_wait=p_wait, p_inst_new=p_inst_new, p_inst_del=p_inst_del)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Custom method that is called by method run(). 

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be processed.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAWorkflow (ml.MLWorkflow):
    """
    ...
    """

    C_TYPE      = 'OA-Workflow'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max=ml.MLWorkflow.C_RANGE_THREAD, 
                  p_class_shared=OAShared, 
                  p_ada=True, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        super().__init__( p_name=p_name, 
                          p_range_max=p_range_max, 
                          p_class_shared=p_class_shared, 
                          p_ada=p_ada, 
                          p_logging=p_logging, 
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def run( self, p_inst:Instance, p_range: int = None, p_wait: bool = False ):
        super().run(p_range=p_range, p_wait=p_wait, p_inst=p_inst)                          





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OAFunction (sl.AdaptiveFunction):
    """
    ...
    """

    C_TYPE      = 'OA-Function'






# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OAScenario (ml.Scenario): 
    """
    ...
    """
    
    C_TYPE      = 'OA-Scenario'





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OATrainingResults (ml.TrainingResults): 
    """
    ...
    """
    
    pass





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OATraining (ml.Training): 
    """
    ...
    """
    
    C_NAME      = 'OA'
