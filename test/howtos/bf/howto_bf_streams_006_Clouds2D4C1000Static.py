## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_006_Clouds2D4C1000Static.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-11  0.0.0     SP       Creation
## -- 2023-09-11  1.0.0     SP       First implementation
## -- 2023-11-10  1.0.1     SP       Bug Fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-28-23)

This module demonstrates how to use the StreamMLProClouds2D4C1000Static class from the clouds module.
This demonstrate and validate in dark mode the origin data and the buffered data.

"""


from mlpro.bf.streams import *
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.tasks import Window
from mlpro.bf.streams.models import StreamTask
from mlpro.bf.various import Log



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EmptyTask (StreamTask):
    """
    Implementation of an empty task with method _run().
    
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'My stream task'


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
