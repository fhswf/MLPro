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
## -- 2023-02-23  0.6.1     DA       Removed class OAFunction
## -- 2023-03-27  0.6.1     DA       Refactoring
## -- 2023-04-09  0.7.0     DA       Class OATask: new methods adapt(), _adapt(), adapt_reverse()
## -- 2023-05-15  0.7.1     DA       Class OATask: new parameter p_buffer_size
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.7.1 (2023-05-15)

Core classes for online adaptive stream processing.
"""


from mlpro.bf.various import Log
from mlpro.bf.streams import *
from mlpro.bf.ml import *
import mlpro.sl as sl

from typing import List




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
    p_buffer_size : int
        Initial size of internal data buffer. Defaut = 0 (no buffering).
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
                  p_buffer_size : int = 0,
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
                        p_buffer_size = p_buffer_size,
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
    def adapt(self, p_inst_new:List[Instance], p_inst_del:List[Instance]) -> bool:
        if not self._adaptivity: return False

        if len(p_inst_new) > 0:
            self.log(self.C_LOG_TYPE_S, 'Adaptation started')
            adapted = self._adapt( p_inst_new=p_inst_new )
        else:
            adapted = False

        if len(p_inst_del) > 0:
            self.log(self.C_LOG_TYPE_S, 'Reverse adaptation started')
            adapted = adapted or self._adapt_reverse( p_inst_del=p_inst_del )

        self._set_adapted( p_adapted = adapted )

        return adapted


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:List[Instance]) -> bool:
        """
        Obligatory custom method for adaptations during regular operation. 

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be processed.

        Returns
        -------
        adapted : bool
            True, if something has been adapted. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt_reverse(self, p_inst_del:List[Instance]) -> bool:
        """
        Optional custom method for reverse adaptations during regular operation. 

        Parameters
        ----------
        p_inst_del : list
            List of obsolete stream instances to be removed.

        Returns
        -------
        adapted : bool
            True, if something has been adapted. False otherwise.
        """

        return False





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
class OAScenario (StreamScenario): 
    """
    Template class for stream based scenarios with online adaptive workflows. 

    Parameters
    ----------
    p_mode
        Operation mode. See bf.ops.Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_cycle_limit : int
        Maximum number of cycles (0=no limit, -1=get from env). Default = 0.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_WE.
    """
    
    C_TYPE      = 'OA-Scenario'

# -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode = Mode.C_MODE_SIM,  
                  p_ada : bool = True,  
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):
        
        self._ada = p_ada

        super().__init__( p_mode = p_mode, 
                          p_cycle_limit = p_cycle_limit, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
    def setup(self):
        """
        Specialized method to set up an oa stream scenario. It is automatically called by the 
        constructor and calls in turn the custom method _setup().
        """

        self._stream, self._workflow = self._setup( p_mode = self.get_mode(),
                                                    p_ada = self._ada, 
                                                    p_visualize = self.get_visualization(),
                                                    p_logging = self.get_log_level() )


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging):
        """
        Custom method to set up a stream scenario consisting of a stream and a processing stream
        workflow.

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
        p_ada : bool
            Boolean switch for adaptivitiy. Default = True.
        p_visualize : bool
            Boolean switch for visualisation.
        p_logging
            Log level (see constants of class Log). Default: Log.C_LOG_ALL.  

        Returns
        -------
        stream : Stream
            A stream object.
        workflow : OAWorkflow
            An online adaptive stream workflow object.
        """

        raise NotImplementedError





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
