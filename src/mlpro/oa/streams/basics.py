## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.streams
## -- Module  : basics.py
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
## -- 2023-03-27  0.6.1     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.6.1 (2023-03-27)

Core classes for online machine learning.
"""


from mlpro.bf.various import Log
from mlpro.bf.streams import *
from mlpro.bf.ml import *
import mlpro.sl as sl




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

        Model.__init__( self,
                        p_ada = p_ada,
                        p_name = p_name,
                        p_range_max = p_range_max,
                        p_autorun = Task.C_AUTORUN_NONE,
                        p_class_shared = None,
                        p_buffer_size = 0,
                        p_visualize = p_visualize,
                        p_logging = p_logging )    

        StreamTask.__init__( self,
                             p_name = p_name,
                             p_range_max = p_range_max,
                             p_duplicate_data = p_duplicate_data,
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
## -------------------------------------------------------------------------------------------------
class OAWorkflow (StreamWorkflow, AWorkflow):
    """
    Online adaptive workflow based on a stream-workflow and an adaptive workflow.

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

        AWorkflow.__init__( self,
                            p_name = p_name,
                            p_range_max = p_range_max,
                            p_class_shared = p_class_shared,
                            p_ada = p_ada,
                            p_visualize = p_visualize,
                            p_logging = p_logging,
                            **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task : StreamTask, p_pred_tasks: list = None):
        AWorkflow.add_task( self, p_task=p_task, p_pred_tasks=p_pred_tasks )





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
