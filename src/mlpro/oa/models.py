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

Core classes for online machine learning.
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
                  p_range_max=ml.MLTask.C_RANGE_THREAD, 
                  p_ada=True, 
                  p_duplicate_data:bool=False,
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
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Specialized definition of method update_plot() of class mlpro.bf.plot.Plottable.

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        return super().update_plot(p_inst_new=p_inst_new, p_inst_del=p_inst_del, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_output: bool, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_output : bool
            If True, the plot output shall be carried out.  
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_output: bool, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_output : bool
            If True, the plot output shall be carried out.  
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_output: bool, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_output : bool
            If True, the plot output shall be carried out.  
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass





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
