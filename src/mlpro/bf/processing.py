## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : processing
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-27  0.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-08-27)

This module provides classes for process management.
"""


from mlpro.bf.various import Log







## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Step (Log):
    """
    ...

    Parameters:
    -----------
    p_... : type
        ...

    """

    C_TYPE          = 'Process Step'

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Processor (Log):
    """
    ...

    Parameters:
    -----------
    p_... : type
        ...

    """

    C_TYPE          = 'Processor'
    C_NAME          = ''

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)


    