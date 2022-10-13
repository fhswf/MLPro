## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_mt_001_parallel_algorithms.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-03  1.0.0     DA       Creation/release
## -- 2022-10-09  1.1.0     DA       Simplification
## -- 2022-10-12  1.2.0     DA       Restructuring of demo steps
## -- 2022-10-13  1.3.0     DA       Restructuring of demo steps
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2022-10-13)

This module demonstrates the use of classes ASync and Shared as part of MLPro's multitasking concept.
Both classes are used to implement a simple parallel algorithm class MyParallelAlgorithm with a
method _async_subtask() for asynchronous execution collecting results in a shared object.

In three runs method _async_subtask() is executed several times a) synchronously, b) as threads and 
c) as processes. Depending on the number of cores per cpu, the operating system, and further factors 
multiprocessing outperforms multithreading more ore less drastically. Method MyParallelAlgoritm.execute() 
determines and logs the speed factor of multithreading and multiprocessing in comparison to serial/synchronous 
computation. Open the perfmeter of your system and play with number of tasks and their duration to observe 
the behavior.

All sub-tasks store dummy results in a shared object. It is no surprise that the order of result entries
in multithreading and multiprocessing mode does not 100% match the order of sub-task starts.

You will learn:

1) The meaning and basic properties of the classes Async and Shared.

2) How to set up an own class with parallel running sub-tasks inside.

3) How to collect results of the parallel sub-functions in a shared object.

"""


from time import sleep
from mlpro.bf.various import Log
import multiprocess as mp
import mlpro.bf.mt as mt
from datetime import datetime, timedelta
from cmath import pi, sin, cos, tan
import random



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
                  p_class_shared=mt.Shared,
                  p_logging=Log.C_LOG_ALL ):

        super().__init__( p_range_max=p_range_max, 
                          p_class_shared=p_class_shared, 
                          p_logging=p_logging )

        self._num_tasks    = p_num_tasks
        self._duration     = p_duration
        self._duration_sec = self._duration.seconds + self._duration.microseconds / 1000000


## -------------------------------------------------------------------------------------------------
    def execute(self):
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
            self._start_async( p_target=self._async_subtask, p_tid=t)

        self.wait_async_tasks()
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
        self.log(Log.C_LOG_TYPE_I, 'Results in shared object are:\n',self._so.get_results())


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
            for i in range(300): 
                result += sin(random.random()*pi) * cos(random.random()*pi) * tan(random.random()*pi)

            time_current = datetime.now()
            time_diff    = time_current - time_start
            if time_diff >= self._duration: break

        # 3 Sub-task can optionally store resuls in the shared object.
        self._so.add_result(p_tid=p_tid, p_result=result)

        # 4 Sub-task needs to check out from shared object
        self._so.checkout( p_tid=p_tid )

        self.log(Log.C_LOG_TYPE_I, 'Task', p_tid, 'stopped')





if __name__ == "__main__":

    # 1 Preparation of execution

    # https://docs.python.org/3/library/multiprocessing.html?highlight=freeze_support#multiprocessing.freeze_support
    mp.freeze_support()

    num_tasks   = 20
    duration    = timedelta(0,1,0)
    pause_sec   = 5
    logging     = Log.C_LOG_ALL



    # 2 Execution of demo class (synchronously)
    a = MyParallelAlgorithm( p_num_tasks = num_tasks, 
                             p_duration = duration, 
                             p_range_max = mt.Async.C_RANGE_NONE, 
                             p_logging = logging )
                                
    a.execute()                     



    # 3 Execution of demo class (asynchonously, multithreading)
    a.log(Log.C_LOG_TYPE_W, 'Short break for better observation of CPU load in perfmeter')
    sleep(pause_sec)

    a = MyParallelAlgorithm( p_num_tasks = num_tasks, 
                             p_duration = duration, 
                             p_range_max = mt.Async.C_RANGE_THREAD, 
                             p_logging = logging )

    a.execute()



    # 4 Execution of demo class (asynchronously, multiprocessing)
    a.log(Log.C_LOG_TYPE_W, 'Short break for better observation of CPU load in perfmeter')
    sleep(pause_sec)

    a = MyParallelAlgorithm( p_num_tasks = num_tasks, 
                             p_duration = duration, 
                             p_range_max = mt.Async.C_RANGE_PROCESS, 
                             p_logging = logging )
                            
    a.execute()                     