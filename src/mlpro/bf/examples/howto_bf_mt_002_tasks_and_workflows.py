## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.mt
## -- Module  : howto_bf_mt_002_tasks_and_workflows.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-01  1.0.0     DA       Creation/release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-10-01)

This module demonstrates the use of tasks and workflows as part of MLPro's multiprocessing concept.
To this regard, we implement an own task class, instantiate 9 task objects based on it, and add
them to a workflow object in a way that...

- run single task in all ranges
- run a workflow in all ranges
- run a workflow as a separate process

You will learn:

1) ...

2) ...

3) ...

4) ...

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
class MyTask (mt.Task):
    """
    Demo implementation of a task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'My fancy task'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_duration:timedelta,
                  p_range_max: int = mt.Task.C_RANGE_PROCESS, 
                  p_autorun=mt.Task.C_AUTORUN_NONE,
                  p_class_shared=None, 
                  p_logging=Log.C_LOG_ALL ):

        super().__init__( p_range_max=p_range_max, 
                          p_autorun=p_autorun,
                          p_class_shared=p_class_shared, 
                          p_logging=p_logging )

        self._duration = p_duration

    
## -------------------------------------------------------------------------------------------------
    def _run(self, **p_kwargs):

        tid = self.get_tid()
        
        self.log(Log.C_LOG_TYPE_I, 'Task', tid, 'started')

        # 1 Dummy implementation to simulate a busy sub-task
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
        self._so.add_result( [tid, result] )
        self._so.unlock()

        self.log(Log.C_LOG_TYPE_I, 'Task', tid, 'stopped')





# 1 Preparation of execution
if __name__ == "__main__":
    # 1.1 Preparation for demo mode
    duration    = timedelta(0,0,500000)
    pause_sec   = 5
    logging     = Log.C_LOG_ALL

else:
    # 1.2 Preparation for unit test mode
    duration    = timedelta(0,0,10000)
    pause_sec   = 0
    logging     = Log.C_LOG_NOTHING



# 2 Create and run a single task
t   = MyTask( p_duration=duration, p_range_max=mt.Task.C_RANGE_PROCESS, p_class_shared=MyShared, p_logging=logging )
t.run(p_range=mt.Task.C_RANGE_NONE, p_wait=True)

exit()


# 3 Create a couple of tasks
t1a = MyTask( p_duration=duration, p_logging=logging )
t1b = MyTask( p_duration=duration, p_logging=logging )
t1c = MyTask( p_duration=duration, p_logging=logging )

t2a = MyTask( p_duration=duration, p_logging=logging )
t2b = MyTask( p_duration=duration, p_logging=logging )
t2c = MyTask( p_duration=duration, p_logging=logging )

t3a = MyTask( p_duration=duration, p_logging=logging )
t3b = MyTask( p_duration=duration, p_logging=logging )
t3c = MyTask( p_duration=duration, p_logging=logging )



# 4 Create a workflow and add the tasks
wf = mt.Workflow( p_range=mt.Workflow.C_RANGE_PROCESS, p_class_shared=MyShared, p_logging=logging )

# 4.1 At first we add three tasks that build the starting points of our workflow
wf.add_task( p_task=t1a )
wf.add_task( p_task=t1b )
wf.add_task( p_task=t1c )

# 4.2 Then, we add three further tasks that shall start when their predecessor tasks have finished
wf.add_task( p_task=t2a, p_pred_tasks=[t1a] )
wf.add_task( p_task=t2b, p_pred_tasks=[t1b] )
wf.add_task( p_task=t2c, p_pred_tasks=[t1c] )

# 4.3 Finally, we add three further tasks that build the end of our task chains
wf.add_task( p_task=t3a, p_pred_tasks=[t2a] )
wf.add_task( p_task=t3b, p_pred_tasks=[t2b] )
wf.add_task( p_task=t3c, p_pred_tasks=[t2c] )



# 5 Run the workflow

# 5.1 Synchronous
wf.run( p_range=mt.Workflow.C_RANGE_NONE, p_wait=True )

# 5.2 Multithreading
wf.run( p_range=mt.Workflow.C_RANGE_THREAD, p_wait=True )

# 5.3 Multiprocessing (This will fail. See log output.)
try:
    wf.run( p_range=mt.Workflow.C_RANGE_THREAD, p_wait=True )
except:
    pass
