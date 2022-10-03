## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.mt
## -- Module  : howto_bf_mt_001_parallel_algorithms.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-01  1.0.0     DA       Creation/release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-10-01)

This module demonstrates the use of classes ASync and Shared as part of MLPro's multitasking concept.
Both classes are used to implement a simple parallel algorithm class MyParallelAlgorithm with a
method _async_subtask() for asynchronous execution collecting results in a shared object.

In three runs method _async_subtask() is executed several times a) synchronously, b) as threads and 
c) as processes. Depending on the number of cores per cpu and further factors multiprocessing outperforms 
multithreading more ore less drastically. Method MyParallelAlgoritm.execute() determines and logs the speed 
factor of multithreading and multiprocessing in comparison to serial/synchronous computation. Open the 
perfmeter of your system and play with number of tasks and their duration to observe the behavior.

All sub-tasks store dummy results in a shared object. It is not a surprise that the order of result
entries in multithreading and multiprocessing mode are random.

You will learn:

1) The meaning and basic properties of the classes Async and Shared.

2) How to set up an own class with parallel running sub-tasks inside.

3) How to collect results of the parallel sub-functions in a shared object.

"""


from time import sleep
from mlpro.bf.various import Log
import mlpro.bf.mt as mt
from datetime import datetime, timedelta

from cmath import pi, sin, cos, tan
import random




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyShared (mt.Shared):
    """
    This class is used for own shared objects with specific attributes and related methods for 
    access...
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_range: int = mt.Async.C_RANGE_PROCESS):
        super().__init__(p_range=p_range)
        self._results : list = []


## -------------------------------------------------------------------------------------------------
    def add_result(self, p_result):
        self._results.append(p_result)


## -------------------------------------------------------------------------------------------------
    def get_results(self) -> list:
        return self._results




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyParallelAlgorithm (mt.Async):
    """
    This class demonstrates how to run methods asynchronously and to collect results in a shared
    object.
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_TYPE      = 'Demo'
    C_NAME      = 'Parallel Algorithm'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_num_tasks:int,
                  p_duration:timedelta,
                  p_range_max=mt.Async.C_RANGE_PROCESS, 
                  p_class_shared=MyShared,
                  p_logging=Log.C_LOG_ALL ):

        super().__init__( p_range_max=p_range_max, 
                          p_class_shared=p_class_shared, 
                          p_logging=p_logging )

        self._num_tasks    = p_num_tasks
        self._duration     = p_duration
        self._duration_sec = self._duration.seconds + self._duration.microseconds / 1000000


## -------------------------------------------------------------------------------------------------
    def execute(self, p_pause:int):
        # Log at the beginning of a run
        if self._range == self.C_RANGE_NONE:
            self.log(Log.C_LOG_TYPE_S, 'Execution of', self._num_tasks, 'synchronous tasks started')
        elif self._range == self.C_RANGE_THREAD:
            self.log(Log.C_LOG_TYPE_S, 'Execution of', self._num_tasks, 'asynchronous tasks as threads started')
        else:
            self.log(Log.C_LOG_TYPE_S, 'Execution of', self._num_tasks, 'asynchronous tasks as processes started')

        # Start number of tasks (a)synchronously
        time_start = datetime.now()
        for t in range(self._num_tasks):
            self._run_async( p_target=self._async_subtask, p_tid=t)

        self._wait_async_tasks()
        time_end   = datetime.now()

        # Determination of speed factor (no parallelism = 1)
        duration_real       = time_end - time_start 
        duration_real_sec   = duration_real.seconds + duration_real.microseconds / 1000000
        speed_factor        = round( self._num_tasks * self._duration_sec / duration_real_sec, 2)

        # Log of speed factor 
        if self._range == self.C_RANGE_NONE:
            self.log(Log.C_LOG_TYPE_S, 'Execution of', self._num_tasks, 'synchronous tasks ended (speed factor =', speed_factor, ')')
        elif self._range == self.C_RANGE_THREAD:
            self.log(Log.C_LOG_TYPE_S, 'Execution of', self._num_tasks, 'asynchronous tasks as threads ended (speed factor =', speed_factor, ')')
        else:
            self.log(Log.C_LOG_TYPE_S, 'Execution of', self._num_tasks, 'asynchronous tasks as processes ended (speed factor =', speed_factor, ')')

        # Log of results collected in the shared object
        self.log(Log.C_LOG_TYPE_I, 'Results in shared object are:')
        for r in self._so.get_results(): self.log(Log.C_LOG_TYPE_I, r)

        # Short break for better observation of the CPU load in the perfmeter
        if p_pause > 0:
            self.log(Log.C_LOG_TYPE_W, 'Short break for better observation of CPU load in perfmeter')
            sleep(p_pause)


## -------------------------------------------------------------------------------------------------
    def _async_subtask(self, p_tid):

        self.log(Log.C_LOG_TYPE_I, 'Task', p_tid, 'started')

        # 1 Sub-task needs to check in on shared object        
        self._so.checkin( p_tid=p_tid )

        # 2 Dummy implementation to simulate a busy sub-task
        time_start = datetime.now()
        result     = 0

        while True:
            # do something meaningful
            for i in range(500): 
                result += sin(random.random()*pi) * cos(random.random()*pi) * tan(random.random()*pi)

            time_current = datetime.now()
            time_diff    = time_current - time_start
            if time_diff >= self._duration: break

        # 3 Sub-task can optionally store resuls in the shared object.
        self._so.lock()
        self._so.add_result( [p_tid, result] )
        self._so.unlock()

        # 4 Sub-task needs to check out from shared object
        self._so.checkout( p_tid=p_tid )

        self.log(Log.C_LOG_TYPE_I, 'Task', p_tid, 'stopped')





# 1 Preparation of execution
if __name__ == "__main__":
    # 1.1 Preparation for demo mode
    num_tasks   = 50
    duration    = timedelta(0,0,500000)
    pause_sec   = 5
    logging     = Log.C_LOG_ALL

else:
    # 1.2 Preparation for unit test mode
    num_tasks   = 2
    duration    = timedelta(0,0,10000)
    pause_sec   = 0
    logging     = Log.C_LOG_ALL



# 2 Execution of demo class (synchronous)
MyParallelAlgorithm( p_num_tasks = num_tasks, 
                     p_duration = duration, 
                     p_range_max = mt.Async.C_RANGE_NONE, 
                     p_logging = logging ).execute(p_pause=pause_sec)                     



# 3 Execution of demo class (asynchonous, multi-threading)
MyParallelAlgorithm( p_num_tasks = num_tasks, 
                     p_duration = duration, 
                     p_range_max = mt.Async.C_RANGE_THREAD, 
                     p_logging = logging ).execute(p_pause=pause_sec)                     



# 4 Execution of demo class (asynchronous, multi-processing)
MyParallelAlgorithm( p_num_tasks = num_tasks, 
                     p_duration = duration, 
                     p_range_max = mt.Async.C_RANGE_PROCESS, 
                     p_logging = logging ).execute(p_pause=0)                     
