## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : mt
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-27  0.0.0     DA       Creation 
## -- 2022-09-10  0.1.0     DA       Initial class definition
## -- 2022-09-xx  1.0.0     DA       Initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-09-xx)

This module provides classes for multitasking with optional interprocess communication (IPC) based
on shared objects.
"""


import threading as mt
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from mlpro.bf.exceptions import *
from mlpro.bf.various import Log
from mlpro.bf.events import EventManager




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Range:
    """
    Property class that adds the range of asynchronicity to a child class.

    Parameters
    ----------
    p_range : int
        Range of asynchonicity 
    """

    # Possible ranges for child classes
    C_RANGE_THREAD          = 0         # separate thread inside the same process
    C_RANGE_PROCESS         = 1         # separate process inside the same machine    

    C_VALID_RANGES          = [ C_RANGE_THREAD, C_RANGE_PROCESS ] 

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_range=C_RANGE_PROCESS):
        if p_range not in self.C_VALID_RANGES: raise ParamError
        self._range = p_range


## -------------------------------------------------------------------------------------------------
    def get_range(self):
        return self._range





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Shared (Range): 
    """
    Template class for shared objects. It is ready to use and the default class for IPC. It is also
    possible to inherit and enrich this class for special needs. It provides elementary mechanisms 
    for access control and messaging.

    Parameters
    ----------
    p_range : int
        Range of asynchonicity 
    """

#    C_MSG_TYPE_DATA         = 0
#    C_MSG_TYPE_TERM         = 1

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_range=Range.C_RANGE_PROCESS):

        super().__init__(p_range)

        if p_range == self.C_RANGE_THREAD:
            self._lock_obj  = mt.Lock()
        elif p_range == self.C_RANGE_PROCESS:
            self._lock_obj  = mp.Lock()
        else:
            raise ParamError

        self._locking_task  = None
        self._messages      = {}


# -------------------------------------------------------------------------------------------------
    def lock(self, p_tid=None, p_timeout:float=None) -> bool: 
        """
        Locks the shared object for a specific process.

        Parameters
        ----------
        p_tid
            Unique task id. If None then the internal locking mechanism is disabled.
        p_timeout : float
            Optional timeout in seconds. If None, timeout is infinite.

        Returns
        True, if shared object was locked successfully. False otherwise.
        """

        if p_tid == self._locking_task: return True
        if p_timeout is None:
            return self._lock_obj.acquire()
        else:
            return self._lock_obj.acquire(timeout=p_timeout)


## -------------------------------------------------------------------------------------------------
    def unlock(self):
        """
        Unlocks the shared object.
        """

        if self._locking_task is None: return
        self._locking_task = None
        self._lock.release()


### -------------------------------------------------------------------------------------------------
    def checkin(self):
        self.lock()
        self._active_tasks +=1
        self.unlock()


## -------------------------------------------------------------------------------------------------
    def checkout(self):
        self.lock()
        if self._active_tasks > 0: self._active_tasks-=1
        self.unlock()


## -------------------------------------------------------------------------------------------------
    def send_message ( self, p_msg_type, p_tid=None, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def receive_message(self, p_tid):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Async (Range, Log):
    """
    Property class that enables child classes to run sub-tasks asynchronously. Depending on the
    given range a task can be executed as a separate thread in the same process or a separate
    process on the same machine.

    Parameters
    ----------
    p_range : int
        Range of asynchonicity. See class Range. 
    p_class_shared
        Optional class name for a shared object (class Shared or a child class of Shared)
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_range=Range.C_RANGE_PROCESS,
                  p_class_shared=None, 
                  p_logging=Log.C_LOG_ALL ):

        Log.__init__(p_logging=p_logging)
        Range.__init__(self, p_range=p_range)

        if p_class_shared is not None:
            # Instantiation of shared object
            if p_range == self.C_RANGE_THREAD:
                self._so = p_class_shared(p_range)
            else:
                BaseManager.register('Shared', p_class_shared)
                self._mpmanager = BaseManager()
                self._mpmanager.start()
                self._so = self._mpmanager.Shared(p_range=p_range)
        else:
            self._so = None

        self._async_tasks   = []


## -------------------------------------------------------------------------------------------------
    def _get_so(self) -> Shared: 
        """
        Returns the associated shared object.

        Returns
        -------
        so : Shared
            Shared object of type Shared (or inherited)
        """

        return self._so


## -------------------------------------------------------------------------------------------------
    def _run_async( self, 
                    p_method=None,
                    p_class=None,
                    **p_kwargs ):
        """
        Runs a method or a new instance of a given class asynchronously. If neither a method nor a
        class is specified, a new instance of the current class is created asynchronously.

        Parameters
        ----------
        p_method
            Optional method to be called asynchronously
        p_class
            Optional class to be instantiated asynchronously
        p_kwargs : dictionary
            Parameters to be handed over to asynchonous method/instance
        """

        if p_method is not None:
            # 1 Prepares a new task for a single method 
            if self._range == self.C_RANGE_THREAD:
                # 1.1 ... as a thread
                task = mt.Thread(target=p_method, kwargs=p_kwargs, group=None)

            else:
                # 1.2 ... as a process
                task = mp.Process(target=p_method, kwargs=p_kwargs, group=None)

        else:
            # 2 Prepares a new task for a new object of a given class
            if p_class is not None: 
                c = p_class
            else:
                c = self.__class__

            kwargs = p_kwargs
            kwargs['p_class'] = c

            if self._range == self.C_RANGE_THREAD:
                # 2.1 ... as a thread
                task = mt.Thread(target=self._run_object_async, kwargs=kwargs, group=None)

            else:
                # 2.2 ... as a process
                task = mp.Process(target=self._run_object_async, kwargs=kwargs, group=None)


        # 3 Registers and runs the new task
        self._async_tasks.append(task)
        task.start()


## -------------------------------------------------------------------------------------------------
    def _run_object_async(self, p_class, **p_kwargs):
        p_class(p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _wait_async_tasks(self):
        for task in self._async_tasks: task.join()
        self._async_tasks.clear()





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
                  p_class_shared=None, 
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        Async.__init__(self, p_range=p_range, p_class_shared=p_class_shared, p_logging=p_logging)
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
                  p_class_shared=None, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        self._tasks = []
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        return super().switch_logging(p_logging)


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
    def _do_recursively(self, p_method, **p_kwargs):
        raise NotImplementedError        


## -------------------------------------------------------------------------------------------------
    def terminate(self):
        raise NotImplemented