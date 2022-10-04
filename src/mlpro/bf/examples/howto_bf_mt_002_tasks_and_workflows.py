## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.mt
## -- Module  : howto_bf_mt_002_tasks_and_workflows.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-04  1.0.0     DA       Creation/release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-10-04)

This module demonstrates the use of tasks and workflows as part of MLPro's multitasking concept.
To this regard, a demo custom task class is implemented. In the first experiment a single task 
instance is created and executed as a separate process. In the second experiment the task class is
instantiated 9 times and added to a workflow and chained by predecessor relations. The workflow is 
executed synchronously and in threading mode. In the latter case, the tasks are partly executed
parallel which increases the computation performance.

In all experiments pseudoo results are stored in a shared object.

You will learn:

1) How to implement an own custom task 

2) How to store results in a shared object

3) How to add tasks to a workflow 

4) How to run tasks and workflows in various ranges of asynchronicity

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
    def clear_results(self):
        self._results.clear()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyTask (mt.Task):
    """
    Demo implementation of a task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'My task'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_duration:timedelta,
                  p_name:str=None,
                  p_range_max: int = mt.Task.C_RANGE_PROCESS, 
                  p_autorun=mt.Task.C_AUTORUN_NONE,
                  p_class_shared=None, 
                  p_logging=Log.C_LOG_ALL ):

        super().__init__( p_name=p_name,
                          p_range_max=p_range_max, 
                          p_autorun=p_autorun,
                          p_class_shared=p_class_shared, 
                          p_logging=p_logging )

        self._duration = p_duration

    
## -------------------------------------------------------------------------------------------------
    def _run(self, **p_kwargs):

        tid = self.get_tid()
        
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
        self._so.add_result( [self.get_name(), result] )
        self._so.unlock()





# 1 Preparation of execution
if __name__ == "__main__":
    # 1.1 Preparation for demo mode
    duration    = timedelta(0,1,0)
    pause_sec   = 5
    logging     = Log.C_LOG_ALL

else:
    # 1.2 Preparation for unit test mode
    duration    = timedelta(0,0,10000)
    pause_sec   = 0
    logging     = Log.C_LOG_NOTHING



# 2 Create and run a single task as process
task = MyTask( p_duration=duration, 
               p_range_max=mt.Task.C_RANGE_PROCESS, 
               p_class_shared=MyShared, 
               p_logging=logging )

task.run(p_range=mt.Task.C_RANGE_PROCESS, p_wait=True)
task.log(Log.C_LOG_TYPE_I, 'Result in shared object:\n', task.get_so().get_results())

# 2.1 Wait for next run
task.log(Log.C_LOG_TYPE_W, 'Short break for better observation of CPU load in perfmeter')
sleep(pause_sec)



# 3 Create and run a workflow

# 3.1 Create a couple of tasks
t1a = MyTask( p_duration=duration, p_name='t1a', p_logging=logging )
t1b = MyTask( p_duration=duration, p_name='t1b', p_logging=logging )
t1c = MyTask( p_duration=duration, p_name='t1c', p_logging=logging )

t2a = MyTask( p_duration=duration, p_name='t2a', p_logging=logging )
t2b = MyTask( p_duration=duration, p_name='t2b', p_logging=logging )
t2c = MyTask( p_duration=duration, p_name='t2c', p_logging=logging )

t3a = MyTask( p_duration=duration, p_name='t3a', p_logging=logging )
t3b = MyTask( p_duration=duration, p_name='t3b', p_logging=logging )
t3c = MyTask( p_duration=duration, p_name='t3c', p_logging=logging )


# 3.2 Create a workflow and add the tasks
wf = mt.Workflow( p_name='wf1', 
                  p_range_max=mt.Workflow.C_RANGE_THREAD, 
                  p_class_shared=MyShared, 
                  p_logging=logging )

# 3.2.1 At first we add three tasks that build the starting points of our workflow
wf.add_task( p_task=t1a )
wf.add_task( p_task=t1b )
wf.add_task( p_task=t1c )

# 3.2.2 Then, we add three further tasks that shall start when their predecessor tasks have finished
wf.add_task( p_task=t2a, p_pred_tasks=[t1a] )
wf.add_task( p_task=t2b, p_pred_tasks=[t1b] )
wf.add_task( p_task=t2c, p_pred_tasks=[t1c] )

# 3.2.3 Finally, we add three further tasks that build the end of our task chains
wf.add_task( p_task=t3a, p_pred_tasks=[t2a, t2b, t2c] )
wf.add_task( p_task=t3b, p_pred_tasks=[t2a, t2b, t2c] )
wf.add_task( p_task=t3c, p_pred_tasks=[t2a, t2b, t2c] )


# 3.3 Run the workflow

# 3.3.1 Synchronous run
wf.run( p_range=mt.Workflow.C_RANGE_NONE, p_wait=True )
wf.log(Log.C_LOG_TYPE_I, 'Result in shared object:\n', wf.get_so().get_results())

# 3.3.2 Clear result list in shared object and wait for next run
wf.get_so().clear_results()
wf.log(Log.C_LOG_TYPE_W, 'Short break for better observation of CPU load in perfmeter')
sleep(pause_sec)

# 3.3.3 Multithreading run
wf.run( p_range=mt.Workflow.C_RANGE_THREAD, p_wait=True)
wf.log(Log.C_LOG_TYPE_I, 'Result in shared object:\n', wf.get_so().get_results())