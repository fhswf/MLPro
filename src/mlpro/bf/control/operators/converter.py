## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.operators
<<<<<<< HEAD
## -- Module  : converter.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-04  0.1.0     ASP      Creation and initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.0 (2024-10-13)

This module provides an implementation of a comparator that determins the control error based on 
setpoint and controlled variable (system state).

"""
from mlpro.bf.various import Log
from mlpro.bf.mt import Task
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.bf.control import Operator,ControlData,ControlledVariable
=======
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

>>>>>>> origin/bf/oa/control



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Converter (Operator):
    """
<<<<<<< HEAD
    The converter converts the source type into the destination type.
    It consumes (not to say: removes) the current source instance variable and replaces it by destination instance variable.
    """

    C_NAME      = 'Converter'


    def __init__(self,
                 p_src_type: ControlData,
                 p_dst_type:ControlData,
=======
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
>>>>>>> origin/bf/oa/control
                 p_range_max=Task.C_RANGE_THREAD, 
                 p_visualize=False, 
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):
<<<<<<< HEAD
=======
        
>>>>>>> origin/bf/oa/control
        super().__init__(p_range_max, p_visualize, p_logging, **p_kwargs)

        self._src_type = p_src_type
        self._dst_type = p_dst_type
        self._duplicate_data = True

<<<<<<< HEAD
## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):

        remove = True if self._src_type !=ControlledVariable else False
        
        # 1 Get source instance 
        src_instance:ControlData = self._get_instance( p_inst = p_inst, p_type = self._src_type, p_remove = remove )

        if src_instance is None: 
            self.log(Log.C_LOG_TYPE_E, f'{self._src_type} missing!')
            return
        
        # 2 Create destination instance 
        dst_instance =self._dst_type(p_id = self.get_so().get_next_inst_id(),
                             p_value_space = src_instance.value_space, 
                             p_values = src_instance.values,
                             p_tstamp = self.get_so().get_tstamp() )


        # 3 Store destination instance      
        p_inst[dst_instance.id] = (InstTypeNew, dst_instance)


=======

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
>>>>>>> origin/bf/oa/control
