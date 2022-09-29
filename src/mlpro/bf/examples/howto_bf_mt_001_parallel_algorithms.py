## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.mt
## -- Module  : howto_bf_mt_001_parallel_algorithms.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-dd  1.0.0     DA       Creation/release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-09-dd)

This module demonstrates the use of classes ASync and Shared as part of MLPro's multitasking concept.

You will learn:

1) The meaning and basic properties of the classes Async and Shared.

2) How to set up an own class with parallel running sub-functions inside.

2) How to collect results of the parallel sub-functions in a shared object.

"""


from mlpro.bf.various import Log
from mlpro.bf.mt import Range, Async, Shared



# 1 Definition of own class 
class MyParallelAlgorithm (Async):

    def __init__( self, 
                  p_num_tasks:int,
                  p_range=Range.C_RANGE_PROCESS, 
                  p_logging=Log.C_LOG_ALL ):

        super().__init__( p_range=p_range, 
                          p_class_shared=Shared, 
                          p_logging=p_logging )

        self._num_tasks = p_num_tasks


    def execute(self):
        for t in enumerate(self._num_tasks):
            self._run_async( p_method=self._async_subtask, p_tid=t)


    def _async_subtask(self, p_tid):
        
        self._so.checkin( p_tid=p_tid )

        # do something meaningful
        # ...

        self._so.checkout( p_tid=p_tid )




# 2 ...




if __name__ == "__main__":
    # 3.1 Interactive/Demo mode
    pass

else:
    # 3.2 Unit test mode
    pass


# 4 ...



# 5 ...
