## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.operators
## -- Module  : integrator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-05  0.1.0     AP       Creation and initial implementation
## -- 2024-11-10  0.2.0     DA       Turned off visualization
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-11-10)

This module provides an implementation of a converter that convertsdetermins the next control variable by
buffering and cumulating it.

"""

import numpy as np

from mlpro.bf.math.basics import Log
from mlpro.bf.mt import Log, Task
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.bf.control import ControlData, Operator, get_ctrl_data




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Converter (Operator):
    """
    The converter converts the source type into the destination type. It consumes (not to say: removes) 
    the current source instance variable and replaces it by destination instance variable.

    Parameters
    ----------
    p_src_type : type
    p_dst_type : type
    ...
    """

    C_NAME          = 'Converter'
    C_PLOT_ACTIVE   = False

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_src_type: type,
                 p_dst_type: type,
                 p_range_max=Task.C_RANGE_THREAD, 
                 p_visualize=False, 
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):
        
        super().__init__(p_range_max, p_visualize, p_logging, **p_kwargs)

        self._src_type = p_src_type
        self._dst_type = p_dst_type
        self._duplicate_data = True


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):

        # 1 Get source instance 
        src_instance : ControlData = get_ctrl_data( p_inst = p_inst, p_type = self._src_type, p_remove = True ) 

        if src_instance is None:       
  
            self.log(Log.C_LOG_TYPE_E, f'{self._src_type} missing!')
            return
        

        # 2 Create destination instance 
        dst_instance =self._dst_type( p_id = self.get_so().get_next_inst_id(),
                                      p_value_space = src_instance.value_space, 
                                      p_values = src_instance.values,
                                      p_tstamp = src_instance.get_tstamp() )


        # 3 Store destination instance      
        p_inst[dst_instance.id] = (InstTypeNew, dst_instance)