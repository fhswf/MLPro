## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : mp
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-27  0.0.0     DA       Creation 
## -- 2022-09-dd  1.0.0     DA       Initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-09-dd)

This module provides classes for multiprocessing with optional interprocess communication (IPC) based
on shared objects.
"""


from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Shared: 
    """
    Template class for shared objects. It is ready to use and the default class for IPC. It is also
    possible to inherit and enrich this class for special needs.
    """

    C_MSG_TYPE_DATA         = 0
    C_MSG_TYPE_TERM         = 1

## -------------------------------------------------------------------------------------------------
    def lock(self):
        pass


## -------------------------------------------------------------------------------------------------
    def unlock(self):
        pass


## -------------------------------------------------------------------------------------------------
    def checkin_process( p_pid ):
        pass


## -------------------------------------------------------------------------------------------------
    def checkout_process( p_pid ):
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MP (Log):
    """
    Property class that enables child classes to run sub-tasks asynchronously. Depending on the
    given range a task can be executed as a separate thread in the same process or a separate
    process on the same machine.

    Parameters
    ----------
    p_cls_shared
        Class name for a shared object
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    
    """

    # Possible ranges for sub-tasks
    C_RANGE_THREAD          = 0         # as separate thread inside the same process
    C_RANGE_PROCESS         = 1         # as separate process inside the same machine     

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_range=C_RANGE_PROCESS,
                  p_cls_shared=Shared, 
                  p_logging=Log.C_LOG_ALL ):

        self._so = None


## -------------------------------------------------------------------------------------------------
    def _get_so(self) -> Shared: 
        return self._so


## -------------------------------------------------------------------------------------------------
    def _run_async_task( self, 
                         p_fct,
                         p_wait:bool=False,
                         **p_kwargs ):
        pass


## -------------------------------------------------------------------------------------------------
    def _wait_async_tasks(self):
        pass