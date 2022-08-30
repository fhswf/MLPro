## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : mp
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-27  0.0.0     DA       Creation 
## -- 2022-08-27  0.1.0     DA       Implementation of process type C_PROCESS_TYPE_LOCAL
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2022-08-27)

This module provides classes for multiprocessing.
"""


from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SharedMemory: 
    """
    Shared memory class to be used by processes to exchange data.
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ProcessBase (Log):
    """
    Root class for all classes of MLPro's process management.

    Parameters:
    -----------
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_PROCESS_TYPE_LOCAL    = 0     # Runs a process locally 
    C_PROCESS_TYPE_GLOBAL   = 1     # Runs a process by spawning a separate os process
    C_VALID_PROCESS_TYPES   = [C_PROCESS_TYPE_LOCAL]

## -------------------------------------------------------------------------------------------------
    def process(self, p_process_type=C_PROCESS_TYPE_LOCAL, **p_kwargs):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Process (Log):
    """
    Template class for a single process step.

    Parameters:
    -----------
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_TYPE                  = 'Process'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        self._smem = None
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_pid(self):
        return self


## -------------------------------------------------------------------------------------------------
    def set_shared_memory(self, p_smem:SharedMemory):
        self._smem = p_smem


## -------------------------------------------------------------------------------------------------
    def process(self, p_process_type=ProcessBase.C_PROCESS_TYPE_LOCAL, **p_kwargs):
        self._process(p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _process(self, **p_kwargs):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Processor (Log):
    """
    ...

    Parameters:
    -----------
    p_cls_shared_mem
        Class used for shared memory object.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_TYPE                  = 'Processor'
    C_NAME                  = ''

    C_PARENT_NONE           = None

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_cls_shared_mem=SharedMemory,
                  p_logging=Log.C_LOG_ALL ):
        self._processes = {}
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def process(self, p_process_type=ProcessBase.C_PROCESS_TYPE_LOCAL, **p_kwargs):
        pass


## -------------------------------------------------------------------------------------------------
    def add_process(self, p_process:Process, p_parent_pid=C_PARENT_NONE):
        pass


## -------------------------------------------------------------------------------------------------
    def _do_recursively(p_parent, p_fct):
        pass


    