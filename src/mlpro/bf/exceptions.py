## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf
## -- Module  : exceptions
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-27  1.0.0     DA       Creation
## -- 2021-11-10  1.0.1     DA       Added new exception ImplementationError
## -- 2021-12-12  1.0.2     DA       Added new exception Error
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18)

This module provides exception classes.
"""


# Export list for public API
__all__ = [ 'ParamError',
            'ImplementationError',
            'Error' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ParamError(Exception):
    """
    To be raised on a parameter error...
    """
    pass


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ImplementationError(Exception):
    """
    To be raised on an implementation error in a child class of MLPro...
    """
    pass


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Error(Exception):
    """
    To be raised on an error...
    """
    pass
