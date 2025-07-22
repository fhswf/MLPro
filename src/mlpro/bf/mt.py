## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf
## -- Module  : mt
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-27  0.0.0     DA       Creation 
## -- 2022-09-10  0.1.0     DA       Initial class definition
## -- 2022-09-30  0.5.0     DA       Implementation of classes Range, Shared, Async
## -- 2022-10-04  1.0.0     DA       Implementation of classes Task, Workflow
## -- 2022-10-06  1.0.1     DA       Class Task: event definition as string
## -- 2022-10-09  1.1.0     DA       Class Shared: systematics for results
## -- 2022-10-12  1.1.1     DA       Replaced package multiprocessing (pickle) by multiprocess (dill)
## -- 2022-10-31  1.2.0     DA       Class Task, Workflow: plot functionality added
## -- 2022-11-04  1.2.1     DA       Class Workflow: corrections
## -- 2022-11-07  1.2.2     DA       Classes Async, Task, Workflow: corrections/refactoring
## -- 2022-11-12  1.2.3     DA       Bugfix in method Task.run()
## -- 2022-11-17  1.3.0     DA       - Class Task: extensions on plotting
## --                                - Bugfix in method Workflow.init_plot()
## -- 2022-11-18  1.3.1     DA       Method Workflow._init_figure: support of different backend types
## -- 2022-11-22  1.3.2     DA       Class Async, Task, Workflow: corrections on plotting
## -- 2022-12-08  1.4.0     DA       - Classes Task, Workflow: bugfixes in plot methods
## --                                - Class Task: replaced method set_num_predecessors() by 
## --                                  set_predecessors()
## -- 2022-12-10  1.4.1     DA       - Moved method _init_figure from class Workflow to Task
## --                                - Method Task._init_figure: added support of backend TkAgg
## -- 2022-12-16  1.5.0     DA       Class Task: new method _get_custom_run_method()
## -- 2022-12-28  1.6.0     DA       Refactoring of plot settings
## -- 2022-12-29  1.7.0     DA       Refactoring of plot settings
## -- 2022-12-30  1.7.1     DA       Bugfix in method Task._get_plot_host_tag()
## -- 2023-01-01  1.8.0     DA       Refactoring of plot settings
## -- 2023-02-15  1.8.1     DA       Class Task: changed default range to C_RANGE_THREAD
## -- 2023-03-27  1.9.0     DA       Class Task: added parent class Persistent
## -- 2024-01-05  1.9.1     DA       Class Task: bugfix in __init__() regarding name generation
## -- 2024-05-31  1.9.2     DA       Class Task: new exception rule for MacOs in meth. init_plot()
## -- 2024-06-17  2.0.0     DA       Class Workflow: new method get_tasks()
## -- 2024-06-18  2.1.0     DA       Class Task: new parent class KWArgs
## -- 2024-10-07  2.2.0     DA       Classes Task, Workflow: new method reset()
## -- 2024-11-10  2.2.0     DA       Refactoring of class Workflow regarding plotting
## -- 2024-11-11  2.3.0     DA       Class Task:
## --                                - new method _on_finished()
## --                                - redefinition of method _raise_event()
## -- 2024-12-10  2.3.1     DA       - Method Task.init_plot(): refactoring
## --                                - Method Workflow.init_plot(): Bugfix and optimization
## -- 2024-12-11  2.4.0     DA       New method Workflow.remove_plot()
## -- 2025-07-18  2.5.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.5.0 (2025-07-18)

This module provides classes for multitasking with optional interprocess communication (IPC) based
on shared objects. Multitasking in MLPro combines multrithreading and multiprocessing and simplifies
parallel programming.

Annotation to multitasking: Standard Python package multiprocessing uses pickle for serialization.
This leads to problems with more complex objects. That was the reason to opt for the more flexible 
package multiprocess, which is a fork of multiprocessing and uses dill for serialization.

See also: https://stackoverflow.com/questions/40234771/replace-pickle-in-python-multiprocessing-lib
"""


import threading as mt
import multiprocess as mp
from multiprocess.managers import BaseManager

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.exceptions import *
from mlpro.bf.various import *
from mlpro.bf.events import EventManager, Event
from mlpro.bf.plot import PlotSettings, Plottable



# Export list for public API
__all__ = [ 'Range',
            'Shared',
            'Async',
            'Task',
            'Workflow' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Range:
    """
    Property class that defines the possible ranges of asynchronous execution supported by MLPro.

    Parameters
    ----------
    p_range : int
        Range of asynchonicity 

    Attributes
    ----------
    C_RANGE_NONE : int
        Synchronous execution only.
    C_RANGE_THREAD : int
        Asynchronous execution as separate thread within the current process.
    C_RANGE_PROCESS : int
        Asynchronous execution as separate process within the current machine.
    C_VALID_RANGES : list
        List of valid ranges.        
    """

    # Possible ranges
    C_RANGE_NONE        = 0  
    C_RANGE_THREAD      = 1  
    C_RANGE_PROCESS     = 2  

    C_VALID_RANGES      = [ C_RANGE_NONE, C_RANGE_THREAD, C_RANGE_PROCESS ] 

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_range:int=C_RANGE_PROCESS):
        if p_range not in self.C_VALID_RANGES: raise ParamError
        self._range:int = p_range


## -------------------------------------------------------------------------------------------------
    def get_range(self) -> int:
        return self._range





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Shared (Range): 
    """
    Template class for shared objects. It is ready to use and the default class for IPC. It provides 
    elementary mechanisms for access control and messaging.

    It is also possible to inherit and enrich a child class for special needs but please beware that
    at least in multiprocessing mode (p_range=Range.C_RANGE_PROCESS) a direct access to attributes
    is not possible. Child classes should generally provide suitable methods for access to attribues.

    Parameters
    ----------
    p_range : int
        Range of asynchonicity 
    """

    C_MSG_TYPE_DATA     = 0
    C_MSG_TYPE_TERM     = 1

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_range:int=Range.C_RANGE_PROCESS):

        Range.__init__(self, p_range=p_range)

        if p_range in [ self.C_RANGE_NONE, self.C_RANGE_THREAD ]:
            self._lock_obj  = mt.Lock()

        elif p_range == self.C_RANGE_PROCESS:
            self._lock_obj  = mp.Lock()

        else:
            raise ParamError

        self._locking_task  = None
        self._active_tasks  = []
        self._results       = {}


## -------------------------------------------------------------------------------------------------
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
            if not self._lock_obj.acquire(): return False
        else:
            if not self._lock_obj.acquire(timeout=p_timeout): return False

        self._locking_task = p_tid
        return True


## -------------------------------------------------------------------------------------------------
    def unlock(self):
        """
        Unlocks the shared object.
        """

        if self._locking_task is None: return
        self._locking_task = None
        self._lock_obj.release()


## -------------------------------------------------------------------------------------------------
    def checkin(self, p_tid):
        """
        Registers a task.

        Parameters
        ----------
        p_tid
            Task id.
        """

        self.lock(p_tid=p_tid)
        self._active_tasks.append(p_tid)
        self.unlock()


## -------------------------------------------------------------------------------------------------
    def checkout(self, p_tid):
        """
        Unregisters a task.

        Parameters
        ----------
        p_tid
            Task id.
        """

        self.lock(p_tid=p_tid)

        if p_tid in self._active_tasks:
            self._active_tasks.remove(p_tid)

        self.unlock()


## -------------------------------------------------------------------------------------------------
    def add_result(self, p_tid, p_result):
        """
        Adds a result for a task.
 
        Parameters
        ----------
        p_tid
            Task id.
        p_result
            Any kind of result data.
        """

        self.lock(p_tid=p_tid)
        self._results[p_tid] = p_result
        self.unlock()


## -------------------------------------------------------------------------------------------------
    def get_result(self, p_tid):
        """
        Returns the result data of a task.

        Parameters
        ----------
        p_tid
            Task id.

        Returns
        -------
        task_results
            Result data of a task.
        """

        return self._results.get(p_tid)


## -------------------------------------------------------------------------------------------------
    def get_results(self):
        """
        Returns reference to internal dictionary of results

        Returns
        -------
        results : dict
            Dictionary of results
        """

        return self._results


## -------------------------------------------------------------------------------------------------
    def clear_results(self):
        """
        Clears internal dictionary of results
        """

        self._results.clear()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Async (Range, Log):
    """
    Property class that enables child classes to run sub-tasks asynchronously. Depending on the
    given range a task can be executed as a separate thread in the same process or a separate
    process on the same machine.

    Parameters
    ----------
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_class_shared
        Optional class for a shared object (class Shared or a child class of Shared)
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL   
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_range_max:int=Range.C_RANGE_PROCESS,
                  p_class_shared=None, 
                  p_logging=Log.C_LOG_ALL ):

        Log.__init__(self, p_logging=p_logging)
        Range.__init__(self, p_range=p_range_max)

        self._async_tasks   = []
        self._mpmanager     = None
        self._class_shared  = p_class_shared

        self._so : Shared   = self._create_so(p_range=p_range_max, p_class_shared=p_class_shared)


## -------------------------------------------------------------------------------------------------
    def _create_so(self, p_range:int, p_class_shared) -> Shared:
        """
        Internal use. Creates a suitable shared object for the given range.
    
        Parameters
        ----------
        p_range : int
            Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
        p_class_shared
            Class for a shared object (class Shared or a child class of Shared)

        Returns
        -------
        so : Shared
            A new shared object
        """

        if p_class_shared is not None:

            # Instantiation of shared object
            if p_range in [ self.C_RANGE_NONE, self.C_RANGE_THREAD ]:
               so = p_class_shared(p_range)

            elif p_range == self.C_RANGE_PROCESS:
                if self._mpmanager is None:
                    BaseManager.register('Shared', p_class_shared)
                    self._mpmanager = BaseManager()
                    self._mpmanager.start()

                so = self._mpmanager.Shared(p_range=p_range)

            else:
                raise NotImplementedError

            self._range = min( self._range, so.get_range() )
            return so

        else:
            return None


## -------------------------------------------------------------------------------------------------
    def get_so(self) -> Shared: 
        """
        Returns the associated shared object.

        Returns
        -------
        so : Shared
            Shared object of type Shared (or inherited)
        """

        return self._so


## -------------------------------------------------------------------------------------------------
    def assign_so(self, p_so:Shared):
        """
        Assigns an existing shared object to the task. The task takes over the range of asynchronicity
        of the shared object if it is less than the current one of the task.

        Parameters
        ----------
        p_so : Shared
            Shared object.
        """

        self._so    = p_so
        self._range = min( self._range, self._so.get_range() )


## -------------------------------------------------------------------------------------------------
    def _start_async( self, 
                      p_target,
                      p_range:int=None,
                      **p_kwargs ) -> int:
        """
        Starts a method or a new instance of a given class asynchronously. If neither a method nor a
        class is specified, a new instance of the current class is created asynchronously.

        Parameters
        ----------
        p_target
            A class, method or function to be executed (a)synchronously depending on the actual range
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that the maximum
            range defined during instantiation is taken. Oterwise the minimum range of both is taken.
        p_kwargs : dictionary
            Parameters to be handed over to asynchonous method/instance

        Returns
        -------
        range : int
            Actual range of asynchronicity
        """

        # 1 Determination of range of asynchronity
        if p_range is None:
            range_run = self._range
        else:
            range_run = min( self._range, p_range )


        # 2 Execution depending on range of asynchronity
        if range_run == self.C_RANGE_NONE: 
            # 2.1 Synchronous execution
            p_target(**p_kwargs)

        elif range_run in [ self.C_RANGE_THREAD, self.C_RANGE_PROCESS ]:
            # 2.2 Asynchronous execution as separate thread or process
            if range_run == self.C_RANGE_THREAD:
                # 2.2.1 Preparation of a new thread
                task = mt.Thread(target=p_target, kwargs=p_kwargs, group=None)

            else:
                # 2.2.2 Preparation of a new process
                task = mp.Process(target=p_target, kwargs=p_kwargs, group=None)

            # 2.2.3 Registration and start of new thread/process
            self._async_tasks.append(task)
            task.start()

        else:
            raise NotImplementedError   


        # 3 Returns actual range of asynchonicity
        return range_run         


## -------------------------------------------------------------------------------------------------
    def wait_async_tasks(self):
        """
        Waits until all internal asynchonous tasks are finished.
        """

        for task in self._async_tasks: task.join()
        self._async_tasks.clear()


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        try:
            self._mpmanager.shutdown()       
        except:
            pass 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Task (Async, EventManager, Plottable, Persistent, KWArgs): 
    """
    Template class for a task, that can run things - and even itself - asynchronously in a thread
    or process. Tasks can run standalone or as part of a workflow (see class Workflow). The integrated
    event manager allows callbacks on specific events inside the same process(!).

    Parameters
    ----------
    p_id
        Optional external id
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_THREAD.
    p_autorun : int
        On value C_AUTORUN_RUN method run() is called imediately during instantiation.
        On vaule C_AUTORUN_LOOP method run_loop() is called.
        Value C_AUTORUN_NONE (default) causes an object instantiation without starting further
        actions.    
    p_class_shared
        Optional class for a shared object (class Shared or a child class of Shared)
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional keyword arguments.
    """

    C_TYPE              = 'Task'

    C_AUTORUN_NONE      = 0
    C_AUTURUN_RUN       = 1
    C_AUTORUN_LOOP      = 2

    C_EVENT_FINISHED    = 'FINISHED'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id = None, 
                  p_name : str = None,
                  p_range_max : int = Async.C_RANGE_THREAD, 
                  p_autorun = C_AUTORUN_NONE,
                  p_class_shared = None, 
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        KWArgs.__init__(self, **p_kwargs)

        self._predecessor_tasks = []
        self._predecessor_ids   = []
        self._num_predecessors  = 0
        self._ctr_predecessors  = 0

        Id.__init__(self, p_id=p_id)

        if p_name is not None:
            self.set_name(p_name)
        else:
            self.set_name(str(self.get_id()))
            
        Async.__init__(self, p_range_max=p_range_max, p_class_shared=p_class_shared, p_logging=p_logging)
        EventManager.__init__(self, p_logging=p_logging)
        Plottable.__init__(self, p_visualize=p_visualize)
        Persistent.__init__(self, p_id=p_id, p_logging=p_logging)

        self._custom_run_method = self._get_custom_run_method()
        self._autorun(p_autorun=p_autorun, p_kwargs=self._kwargs)


## -------------------------------------------------------------------------------------------------
    def _get_custom_run_method(self):
        return self._run


## -------------------------------------------------------------------------------------------------
    def get_tid(self):
        """
        Returns unique task id.
        """

        return self.get_id()
    

## -------------------------------------------------------------------------------------------------
    def reset(self, **p_kwargs):
        pass


## -------------------------------------------------------------------------------------------------
    def _autorun(self, p_autorun, **p_kwargs):
        """
        Internal method to automate a single or looped run.

        Parameters
        ----------
        p_autorun : int
            On value C_AUTORUN_RUN method run() is called imediately during instantiation.
            On vaule C_AUTORUN_LOOP method run_loop() is called.
            Value C_AUTORUN_NONE (default) causes an object instantiation without starting further
            actions.    
        p_kwargs : dict
            Further parameters handed over to method run().
        """

        if p_autorun == self.C_AUTURUN_RUN:
            self.run(p_kwargs=p_kwargs)
        elif p_autorun == self.C_AUTORUN_LOOP:
            self.run_loop(p_kwargs=p_kwargs)


## -------------------------------------------------------------------------------------------------
    def run(self, p_range:int=None, p_wait:bool=False, **p_kwargs):
        """
        Executes the task specific actions implemented in custom method _run(). At the end event
        C_EVENT_FINISHED is raised to start subsequent actions (p_wait=True).

        Parameters
        ----------
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that the maximum
            range defined during instantiation is taken. Oterwise the minimum range of both is taken.
        p_wait : bool
            If True, the method waits until all (a)synchronous tasks are finished.
        p_kwargs : dict
            Further parameters handed over to custom method _run().
        """

        if p_range is None:
            self._range_run = self._range
        else:
            self._range_run = min( p_range, self._range )

        if self._range_run == self.C_RANGE_NONE:
            self.log(Log.C_LOG_TYPE_S, 'Started synchronously')
        elif self._range_run == self.C_RANGE_THREAD:
            self.log(Log.C_LOG_TYPE_S, 'Started as new thread')
        else:
            self.log(Log.C_LOG_TYPE_S, 'Started as new process')

        self._start_async( p_target=self._run_async, p_range=self._range_run, **p_kwargs )

        if p_wait: self.wait_async_tasks()
        

## -------------------------------------------------------------------------------------------------
    def _run_async(self, **p_kwargs):
        """
        Internally used by method run(). It runs the custom method _run() and raises event C_EVENT_FINISHED.

        Parameters
        ----------
        p_kwargs : dict
            Custom parameters.
        """

        if self._so is not None: 
            self._so.checkin(p_tid=self._id)
            self.log(Log.C_LOG_TYPE_I, 'Checked in to shared object')

        self._custom_run_method(**p_kwargs)

        if self._so is not None: 
            self._so.checkout(p_tid=self.get_tid())
            self.log(Log.C_LOG_TYPE_I, 'Checked out from shared object')

        self.update_plot(**p_kwargs)

        self.log(Log.C_LOG_TYPE_S, 'Stopped')

        self._raise_event( self.C_EVENT_FINISHED, Event(p_raising_object=self, p_range=self._range_run, p_wait=False) )


## -------------------------------------------------------------------------------------------------
    def _run(self, **p_kwargs):
        """
        Custom method that is called (asynchronously) by method run(). 

        Parameters
        ----------
        p_kwargs : dict
            Custom parameters.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def run_loop(self, **p_kwargs):
        """
        Executes method run() in a loop, until a message of type Shared.C_MSG_TYPE_TERM is sent to
        the task.

        Parameters
        ----------
        p_kwargs : dict
            Parameters for method run()
        """
        
        while True:
            self.run(p_kwargs=p_kwargs)

            if self._so is not None:
                msg_type, = self._so.receive_message( p_tid=self.get_tid(), p_msg_type=Shared.C_MSG_TYPE_TERM )
                if msg_type is not None: break


## -------------------------------------------------------------------------------------------------
    def get_predecessors(self) -> list:
        return self._predecessor_tasks


## -------------------------------------------------------------------------------------------------
    def set_predecessors(self, p_predecessor_tasks:list ):
        """
        Used by class Workflow to inform a task about it's number of predecessor tasks. See method
        run_on_event().

        Parameters
        ----------
        p_predecessor_ids : list
            List of ids of predecessor tasks in a workflow.
        """

        self._predecessor_tasks = p_predecessor_tasks
        self._predecessor_ids   = []

        for task in self._predecessor_tasks:
            self._predecessor_ids.append(task.get_tid())

        self._num_predecessors = self._ctr_predecessors = len(self._predecessor_ids)


## -------------------------------------------------------------------------------------------------
    def _raise_event(self, p_event_id: str, p_event_object: Event):
        if p_event_id == self.C_EVENT_FINISHED: self._on_finished()
        EventManager._raise_event(self, p_event_id, p_event_object)


## -------------------------------------------------------------------------------------------------
    def _on_finished(self):
        """
        Custom method that is called before an event C_EVENT_FINISHED is raised.
        """
        pass


## -------------------------------------------------------------------------------------------------
    def run_on_event(self, p_event_id, p_event_object:Event):
        """
        Can be used as event handler - in particular for other tasks in a workflow in combination 
        with event C_EVENT_FINISHED. Method self.run() is called if the last predecessor task in a
        workflow has raised event C_EVENT_FINISHED.

        Parameters
        ----------
        p_event_id 
            Event id.
        p_event_object : Event
            Event object with further context informations.
        """

        if p_event_id == self.C_EVENT_FINISHED:
            if self._ctr_predecessors > 1: 
                # Execution of method run() is delayed until the last predecessor task in a workflow
                # has finished
                self._ctr_predecessors -= 1
                return

            else:
                self._ctr_predecessors = self._num_predecessors

        self.run(**p_event_object.get_data())


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure: Figure = None, 
                   p_plot_settings : PlotSettings = None,
                   p_window_title: str = None ):
        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return

        try:
            if self._plot_initialized: return
        except:
            pass

        self.log(Log.C_LOG_TYPE_S, 'Init plot')

        try:
            view = p_plot_settings.view
        except:
            view = self.C_PLOT_DEFAULT_VIEW

        if p_window_title is not None:
            title = p_window_title
        else:
            title = 'MLPro: ' + self.C_TYPE + ' ' + self.get_name() + ' (' + view + ')'

        Plottable.init_plot( self,
                             p_figure=p_figure, 
                             p_plot_settings=p_plot_settings,
                             p_window_title=title )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return

        self.log(Log.C_LOG_TYPE_S, 'Update plot')
        Plottable.update_plot(self, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Workflow (Task): 
    """
    Ready-to-use container class for task groups. Objects of type Task (or inherited) can be added and
    chained to sequences or hierarchies of tasks. 

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Range of asynchonicity. See class Range. Default is Range.C_RANGE_THREAD.
    p_class_shared
        Optional class for a shared object (class Shared or a child class of Shared)
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters handed over to every task within.
    """
    
    C_TYPE          = 'Workflow'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name:str=None,
                  p_range_max=Async.C_RANGE_THREAD, 
                  p_class_shared=None, 
                  p_visualize:bool=False,
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        self._tasks             = []
        self._entry_tasks       = []
        self._final_tasks       = []
        self._first_run         = True

        self._finished          = mt.Event()
        self._finished.clear()

        Task.__init__( self, 
                       p_name=p_name,
                       p_range_max=p_range_max,
                       p_autorun=self.C_AUTORUN_NONE,
                       p_class_shared=p_class_shared,
                       p_visualize=p_visualize,
                       p_logging=p_logging,
                       **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        """
        Sets log level for the workflow and all tasks inside.

        Parameters
        ----------
        p_logging
            Log level (see constants of class Log).
        """

        Task.switch_logging(self, p_logging=p_logging)
        for task in self._tasks: task.switch_logging(p_logging=p_logging)


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

        # 1 Info log
        self.log(Log.C_LOG_TYPE_I, 'Adding task "' + p_task.get_name() + '"')


        # 2 Plausibility checks

        # 2.1 Check: task already added?
        if p_task in self._tasks:
            self.log(Log.C_LOG_TYPE_E, 'Please do not add the same task more than once')
            raise ParamError

        # 2.2 Check: is the task added after the first run?
        if not self._first_run:
            self.log(Log.C_LOG_TYPE_E, 'Please do not add further tasks after the first run')
            raise ParamError


        # 3 New task is prepared for workflow operation
        p_task.switch_logging(self._level)
        p_task.assign_so(self._so)


        # 4 Register task and its event handler to all predecessor tasks
        self._tasks.append(p_task)
        self._final_tasks.append(p_task)

        if ( p_pred_tasks is None ) or ( len(p_pred_tasks) == 0 ):
            self._entry_tasks.append(p_task)

        else:
            p_task.set_predecessors( p_predecessor_tasks=p_pred_tasks )

            if self._range > self.C_RANGE_THREAD:
                self.log(Log.C_LOG_TYPE_W, 'Predecessor relations are event-based and not yet supported beyond multithreading. Range is reduced')
                self._range = self.C_RANGE_THREAD
                
            for t_pred in p_pred_tasks: 
                t_pred.register_event_handler(p_event_id=self.C_EVENT_FINISHED, p_event_handler=p_task.run_on_event)
                if t_pred in self._final_tasks: self._final_tasks.remove(t_pred)


## -------------------------------------------------------------------------------------------------
    def get_tasks(self) -> list:
        return self._tasks


## -------------------------------------------------------------------------------------------------
    def reset(self, **p_kwargs):
        for task in self._tasks:
            task.reset(**p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _get_plot_host_task(self, p_task : Task) -> Task:
        plot_host = None

        for task in p_task.get_predecessors():
            if task.C_PLOT_STANDALONE and task.get_visualization():
                plot_host = task
                break

        if plot_host is None:
            return self
        else:
            return plot_host


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure:Figure=None, 
                   p_plot_settings : PlotSettings = None,
                   p_window_title: str = None ):
        """
        Initializes the plot of a workflow. The method creates a host figure for all tasks if no 
        external host figure is parameterized. The sub-plots of the tasks are autmatically arranged
        within the host figure.

        See method init_plot() of class mlpro.bf.plot.Plottable for further details.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure, optional
            Optional MatPlotLib host figure, where the plot shall be embedded. The default is None.
        p_plot_settings : PlotSettings
            Optional plot settings. If None, the default view is plotted (see attribute C_PLOT_DEFAULT_VIEW).
        p_window_title : str
            Optional window title.
        """

        # 1 Init plot output on workflow level (if activated)
        Task.init_plot( self,
                        p_figure = p_figure, 
                        p_plot_settings = p_plot_settings,
                        p_window_title = p_window_title )

      
        # 2 Init plot output on task level
        for task in self._tasks:

            task_figure        = None
            task_plot_settings = p_plot_settings

            if task.get_visualization():

                if not task.C_PLOT_STANDALONE:
                    plot_host = self._get_plot_host_task(p_task=task)
                else:
                    plot_host = self

                ps = plot_host.get_plot_settings()
                if ps is None: ps = p_plot_settings

                if task.C_PLOT_STANDALONE:
                    # Task plots in a separate figure (=window)
                    if ps is not None:
                        task_plot_settings = ps.copy()
                    else:
                        task_plot_settings = PlotSettings( p_view = self.C_PLOT_DEFAULT_VIEW )

                    task_plot_settings.axes  = None
                    task_plot_settings.pos_x = 1
                    task_plot_settings.pos_y = 1
                    task_plot_settings.id    = 1

                else:
                    # Task plots embedded in the predecessor/workflow figure/subplot
                    task_figure = plot_host._figure
                    task_plot_settings = ps
                
            task.init_plot( p_figure=task_figure,
                            p_plot_settings=task_plot_settings )


        # 3 Initial refresh of workflow window
        if self.get_visualization() and self._plot_own_figure:
            self._figure.canvas.draw()
            self._figure.canvas.flush_events()


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh = True):
        for task in self._tasks:
            task.remove_plot( p_refresh=p_refresh )

        return super().remove_plot(p_refresh)
    

## -------------------------------------------------------------------------------------------------
    def run(self, p_range:int=None, p_wait:bool=False, **p_kwargs):
        """
        Executes all tasks of the workflow. At the end event C_EVENT_FINISHED is raised to start 
        subsequent actions (p_wait=True).

        Parameters
        ----------
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that 
            the maximum range defined during instantiation is taken. Oterwise the minimum range of both 
            is taken.
        p_wait : bool
            If True, the method waits until all (a)synchronous tasks are finished.
        p_kwargs : dict
            Further parameters handed over to custom method _run().
        """

        # 1 Intro
        self._finished.clear()


        # 2 Determine the scope of asynchronicity
        if p_range is None:
            range_run = self._range
        else:
            range_run = min( p_range, self._range )

        if range_run == self.C_RANGE_NONE:
            self.log(Log.C_LOG_TYPE_S, 'Started synchronously')
        elif range_run == self.C_RANGE_THREAD:
            self.log(Log.C_LOG_TYPE_S, 'Started as new thread')
        else:
            self.log(Log.C_LOG_TYPE_S, 'Started as new process')


        # 3 Prepare inner task structure for first run
        if self._first_run:
            for t_final in self._final_tasks:
                t_final.register_event_handler(p_event_id=self.C_EVENT_FINISHED, p_event_handler=self.event_forwarder)

            self._first_run = False

        self._ctr_final_tasks = len(self._final_tasks)


        # 4 Update plot of workflow
        self.update_plot(**p_kwargs)


        # 5 Execution of all tasks within the workflow
        for task in self._entry_tasks: 
            task.run( p_range=range_run, **p_kwargs )

        if p_wait: self.wait_async_tasks()


## -------------------------------------------------------------------------------------------------
    def event_forwarder(self, p_event_id, p_event_object:Event):
        """
        Internally used to raise event C_EVENT_FINISHED on workflow level if all final tasks have
        been finished.

        Parameters
        ----------
        p_event_id 
            Event id.
        p_event_object : Event
            Event object with further context informations.
        """

        if self._ctr_final_tasks > 1:
            self._ctr_final_tasks -= 1
        else:
            self.log(Log.C_LOG_TYPE_S, 'Stopped')
            self._finished.set()
            self._raise_event( self.C_EVENT_FINISHED, Event(p_raising_object=self) )


## -------------------------------------------------------------------------------------------------
    def wait_async_tasks(self):
        """
        Waits until all tasks are finished.
        """

        self._finished.wait()