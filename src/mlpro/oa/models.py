## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa
## -- Module  : models.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-04  0.0.0     DA       Creation
## -- 2022-10-09  0.1.0     DA       Initial class definitions
## -- 2022-10-26  0.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2022-10-26)

Core classes for online machine learning.
"""


from mlpro.bf.various import Log
from mlpro.bf.streams import *
from mlpro.bf.mt import Shared
from mlpro.bf.ml import Model, Scenario, Training, TrainingResults
import mlpro.sl.models as sl




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAShared (Shared):
    """
    Template class for a shared memory. 
    """ 
    
    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OATask (StreamTask, Model):
    """
    Template class for online adaptive ML tasks.

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_duplicate_data : bool     
        If True the incoming data are copied before processing. Otherwise the origin incoming data
        are modified.        
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE              = 'OA-Task'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = True
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_2D, PlotSettings.C_VIEW_3D, PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max=StreamTask.C_RANGE_THREAD, 
                  p_ada=True, 
                  p_duplicate_data:bool=False,
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        StreamTask.__init__( self,
                             p_name=p_name, 
                             p_range_max=p_range_max, 
                             p_autorun=StreamTask.C_AUTORUN_NONE, 
                             p_class_shared=None, 
                             p_buffer_size=0, 
                             p_ada=p_ada, 
                             p_logging=p_logging, 
                             **p_kwargs )

        Model.__init__( self, 
                        p_buffer_size=0, 
                        p_ada=p_ada, 
                        p_logging=p_logging,
                        **p_kwargs )  

        self._duplicate_data = p_duplicate_data


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

        if self._duplicate_data:
            inst_new = [ inst.copy() for inst in p_inst_new ] 
            inst_del = [ inst.copy() for inst in p_inst_del ]
        else:
            inst_new = p_inst_new
            inst_del = p_inst_del

        super().run(p_range=p_range, p_wait=p_wait, p_inst_new=inst_new, p_inst_del=inst_del)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Custom method that is called by method run(). If the task adapts during regular operation 
        please call method adapt() here and implement custom method _adapt().

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be processed.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:list, p_inst_del:list) -> bool:
        """
        Custom method for adaptations during regular operation. See method _run() for further informmation.

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be processed.
        p_inst_del : list
            List of obsolete stream instances to be removed.

        Returns
        -------
        adapted : bool
            True, if something has been adapted. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def adapt_on_event(self, p_event_id:str, p_event_obj:Event):
        """
        Method to be used as event handler for event-based adaptations. Calls custom method 
        _adapt_on_event() and updates the internal adaptation state.

        Parameters
        ----------
        p_event_id : str
            Event id.
        p_event_obj : Event
            Object with further context informations about the event.
        """

        self._set_adapted(p_adapted=self._adapt_on_event(p_event_id=p_event_id, p_event_obj=p_event_obj))        


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_obj:Event) -> bool:
        """
        Custom method to be used for event-based adaptation. See method adapt_on_event().

        Parameters
        ----------
        p_event_id : str
            Event id.
        p_event_obj : Event
            Object with further context informations about the event.

        Returns
        -------
        adapted : bool
            True, if something was adapted. False otherwise.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAWorkflow (StreamWorkflow, Model):
    """
    ...
    """

    C_TYPE      = 'OA-Workflow'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max=StreamWorkflow.C_RANGE_THREAD, 
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
class OAScenario (Scenario): 
    """
    ...
    """
    
    C_TYPE      = 'OA-Scenario'





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OATrainingResults (TrainingResults): 
    """
    ...
    """
    
    pass





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OATraining (Training): 
    """
    ...
    """
    
    C_NAME      = 'OA'
