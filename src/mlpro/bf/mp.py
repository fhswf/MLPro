## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : mp
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-27  0.0.0     DA       Creation 
## -- 2022-09-10  0.1.0     DA       Initial class definition
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2022-09-11)

This module provides classes for multiprocessing with optional interprocess communication (IPC) based
on shared objects.
"""


from mlpro.bf.various import Log
from mlpro.bf.events import EventManager




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Shared: 
    """
    Template class for shared objects. It is ready to use and the default class for IPC. It is also
    possible to inherit and enrich this class for special needs. It provides elementary mechanisms 
    for access control and messaging.
    """

    C_MSG_TYPE_DATA         = 0
    C_MSG_TYPE_TERM         = 1


## -------------------------------------------------------------------------------------------------
    def __init__(self):
       self._locked = False
       self._active_processes = 0
       self._messages = {}


## -------------------------------------------------------------------------------------------------
    def lock(self):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def unlock(self):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def checkin_process(self, p_tid):
        self._active_processes+=1


## -------------------------------------------------------------------------------------------------
    def checkout_process(self, p_tid):
        if self._active_processes > 0:
            self._active_processes-=1


## -------------------------------------------------------------------------------------------------
    def send_message ( self, p_msg_type, p_tid=None, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def receive_message(self, p_tid):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Async (Log):
    """
    Property class that enables child classes to run sub-tasks asynchronously. Depending on the
    given range a task can be executed as a separate thread in the same process or a separate
    process on the same machine.

    Parameters
    ----------
    p_cls_shared
        Optional class name for a shared object (class Shared or a child class of Shared)
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    
    """

    # Possible ranges for sub-tasks
    C_RANGE_THREAD          = 0         # as separate thread inside the same process
    C_RANGE_PROCESS         = 1         # as separate process inside the same machine     

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_range=C_RANGE_PROCESS,
                  p_cls_shared=None, 
                  p_logging=Log.C_LOG_ALL ):

        self._so = None
        self._async_tasks = []


## -------------------------------------------------------------------------------------------------
    def _get_so(self) -> Shared: 
        return self._so


## -------------------------------------------------------------------------------------------------
    def _run_async( self, 
                    p_method=None,
                    p_class=None,
                    p_wait:bool=False,
                    **p_kwargs ):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run_myself_async(self, p_wait:bool=False, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _wait_async_runs(self):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Task (Async, EventManager): 
    """
    ...

    Parameters
    ----------
    
    """

    C_TYPE          = 'Task'

    C_AUTORUN_NONE  = 0
    C_AUTURUN_RUN   = 1
    C_AUTORUN_LOOP  = 2

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_range=Async.C_RANGE_PROCESS, 
                  p_autorun=C_AUTORUN_NONE,
                  p_cls_shared=None, 
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        Async.__init__(self, p_range=p_range, p_cls_shared=p_cls_shared, p_logging=p_logging)
        EventManager.__init__(self, p_logging=p_logging)
        self._autorun(p_autorun=p_autorun, p_kwargs=p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _autorun(self, p_autorun, **p_kwargs):
        if p_autorun == self.C_AUTURUN_RUN:
            self.run(p_kwargs=p_kwargs)
        elif p_autorun == self.C_AUTORUN_LOOP:
            self.run_loop(p_kwargs=p_kwargs)


## -------------------------------------------------------------------------------------------------
    def run(self, **p_kwargs):
        self._run(p_kwargs=p_kwargs)
        

## -------------------------------------------------------------------------------------------------
    def _run(self, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def run_loop(self, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run_myself_async(self, p_wait: bool = False, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def terminate(self):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Workflow (Task): 
    """
    ...

    Parameters
    ----------

    """
    
    C_TYPE          = 'Workflow'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_range=Async.C_RANGE_PROCESS, 
                  p_autorun=Task.C_AUTORUN_NONE, 
                  p_cls_shared=None, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run(self, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task:Task, p_pred_tasks:list=None):
        """
        Adds a task to the workflow.

        Parameters
        ----------
        p_task : Task
            Task object to be added.
        p_pred_tasks : list
            Optional list of predecessor task objects

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def do_recursively(self, p_method, **p_kwargs):
        raise NotImplementedError        


## -------------------------------------------------------------------------------------------------
    def terminate(self):
        raise NotImplemented