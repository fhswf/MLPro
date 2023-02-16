## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.systems
## -- Module  : systems.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-mm-mm  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-02-16)

This module provides modules and template classes for adaptive systems and adaptive functions.
"""





from mlpro.bf.ml.systems.adaptive_systems import *
from mlpro.bf.systems import *
from mlpro.bf.ml import Model
from mlpro.bf.streams import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSTrans(AFctSTrans, Model):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
              p_name:str=None,
              p_range_max=Async.C_RANGE_THREAD,
              p_class_shared=None,
              p_visualize:bool=False,
              p_logging=Log.C_LOG_ALL,
              **p_kwargs):

        AFctSTrans.__init__(self,
              p_name=p_name,
              p_range_max=p_range_max,
              p_class_shared=p_class_shared,
              p_visualize=p_visualize,
              p_logging=p_logging,
              **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """

        Parameters
        ----------
        p_state
        p_action

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """

        Parameters
        ----------
        p_state
        p_action

        Returns
        -------

        """


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task:StreamTask):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSuccess(FctSuccess, Model):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
              p_name:str=None,
              p_range_max=Async.C_RANGE_THREAD,
              p_class_shared=None,
              p_visualize:bool=False,
              p_logging=Log.C_LOG_ALL,
              **p_kwargs):

        AFctSuccess.__init__(self,
              p_name=p_name,
              p_range_max=p_range_max,
              p_class_shared=p_class_shared,
              p_visualize=p_visualize,
              p_logging=p_logging,
              **p_kwargs)

        Model.__init__()


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _add_task(self, p_task:StreamTask):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFCtBroken(FctBroken, Model):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
              p_name:str=None,
              p_range_max=Async.C_RANGE_THREAD,
              p_class_shared=None,
              p_visualize:bool=False,
              p_logging=Log.C_LOG_ALL,
              **p_kwargs):

        AFctBroken.__init__(self,
              p_name=p_name,
              p_range_max=p_range_max,
              p_class_shared=p_class_shared,
              p_visualize=p_visualize,
              p_logging=p_logging,
              **p_kwargs)

        Model.__init__()


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task: StreamTask):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OASystem(ASystem):
    """

    Parameters
    ----------
    p_mode
    p_latency
    p_fct_strans
    p_fct_success
    p_fct_broken
    p_processing_wf
    p_visualize
    p_logging
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
               p_mode = Mode.C_MODE_SIM,
               p_latency = None,
               p_fct_strans : FctSTrans = None,
               p_fct_success : FctSuccess = None,
               p_fct_broken : FctBroken = None,
               p_processing_wf : StreamWorkflow = None,
               p_visualize : bool = False,
               p_logging = Log.C_LOG_ALL):

        ASystem.__init__(self,
               p_mode = p_mode,
               p_latency = p_latency,
               p_fct_strans = p_fct_strans,
               p_fct_success = p_fct_success,
               p_fct_broken = p_fct_broken,
               p_visualize = p_visualize,
               p_logging = p_logging)

        self._processing_wf = p_processing_wf



## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces() -> (MSpace, MSpace):

        return None, None


## -------------------------------------------------------------------------------------------------
    def _set_adapted(self, p_adapted:bool):
        """

        Parameters
        ----------
        p_adapted

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """

        Parameters
        ----------
        p_ada

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _run_processing_wf(self):
        """

        Returns
        -------

        """
        pass