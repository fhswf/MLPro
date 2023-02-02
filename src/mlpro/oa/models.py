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
## -- 2022-10-29  0.3.0     DA       Refactoring
## -- 2022-11-30  0.4.0     DA       Refactoring after changes on bf.streams design
## -- 2022-12-09  0.4.1     DA       Corrections
## -- 2022-12-20  0.5.0     DA       Refactoring
## -- 2023-01-01  0.6.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.6.0 (2023-01-01)

Core classes for online machine learning.
"""


from mlpro.bf.various import Log
from mlpro.bf.streams import *
from mlpro.bf.ml import *
import mlpro.sl.models as sl




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAShared (StreamShared):
    """
    Template class for shared objects in the context of online adaptive stream processing.
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
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
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
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_ada : bool = True, 
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        StreamTask.__init__( self,
                             p_name = p_name,
                             p_range_max = p_range_max,
                             p_duplicate_data = p_duplicate_data,
                             p_visualize = p_visualize,
                             p_logging = p_logging,
                             **p_kwargs )                             

        Model.__init__( self, 
                        p_buffer_size = 0, 
                        p_ada = p_ada, 
                        p_visualize = p_visualize,
                        p_logging = p_logging,
                        **p_kwargs )  


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
    def adapt_on_event(self, p_event_id:str, p_event_object:Event):
        """
        Method to be used as event handler for event-based adaptations. Calls custom method 
        _adapt_on_event() and updates the internal adaptation state.

        Parameters
        ----------
        p_event_id : str
            Event id.
        p_event_object : Event
            Object with further context informations about the event.
        """

        self._set_adapted(p_adapted=self._adapt_on_event(p_event_id=p_event_id, p_event_object=p_event_object))        


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """
        Custom method to be used for event-based adaptation. See method adapt_on_event().

        Parameters
        ----------
        p_event_id : str
            Event id.
        p_event_object : Event
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
    Online adaptive workflow based on a stream-workflow and an ml model.

    Parameters
    ----------
    p_name : str
        Optional name of the workflow. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_class_shared
        Optional class for a shared object (class OAShared or a child class of OAShared)
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.
     """

    C_TYPE      = 'OA-Workflow'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = StreamWorkflow.C_RANGE_THREAD, 
                  p_class_shared = OAShared, 
                  p_ada : bool = True, 
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        StreamWorkflow.__init__( self, 
                                 p_name = p_name,
                                 p_range_max = p_range_max,
                                 p_class_shared = p_class_shared,
                                 p_visualize = p_visualize,
                                 p_logging = p_logging,
                                 **p_kwargs )

        Model.__init__( self,
                        p_buffer_size = 0,
                        p_ada = p_ada,
                        p_visualize = p_visualize,
                        p_logging = p_logging,
                        **p_kwargs )                            


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task: StreamTask, p_pred_tasks: list = None):
        super().add_task(p_task=p_task, p_pred_tasks=p_pred_tasks)

        try:
            # Set adaptivity of new task
            p_task.switch_adaptivity(self._adaptivity)

            # Hyperparameter space of workflow is extended by dimensions of hyperparameter space of
            # the new task
            task_hp_set = p_task.get_hyperparam().get_related_set()

            for dim_id in task_hp_set.get_dim_ids():
                self._hyperparam_space.add_dim(p_dim=task_hp_set.get_dim(p_id=dim_id)) 

            # Hyperparameter tuple of workflow is extended by the hyperparameter tuple of the new task
            if self._hyperparam_tuple is None: 
                self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)

            task_hp_tuple = p_task.get_hyperparam()
            if task_hp_tuple is not None:
                self._hyperparam_tuple.add_hp_tuple(p_hpt=p_task.get_hyperparam())

        except:
            pass


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        for t in self._tasks:
            try:
                t.switch_adaptivity(p_ada=p_ada)
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        for t in self._tasks:
            try:
                t.set_random_seed(p_seed=p_seed)
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        adapted = False

        for t in self._tasks:
            try:
                adapted = adapted or t.get_adapted()
            except:
                pass

        return adapted


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """
        Explicit adaptation is disabled for OA-Workflows.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        for t in self._tasks:
            try:
                t.clear_buffer()
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def get_accuracy(self):
        accuracy       = 0
        adaptive_tasks = 0

        for t in self._tasks:
            try:
                accuracy       += t.get_accuracy()
                adaptive_tasks += 1
            except:
                pass

        if adaptive_tasks > 0: return accuracy / adaptive_tasks
        else: return 1





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OAFunction (sl.AdaptiveFunction):
    """
    ...
    """

    C_TYPE      = 'OA-Function'





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OAScenario (StreamScenario): 
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
