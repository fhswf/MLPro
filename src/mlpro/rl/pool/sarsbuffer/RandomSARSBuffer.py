## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.pool.sarbuffer
## -- Module  : RandomSARSBuffer
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-24  0.0.0     MRD      Creation
## -- 2021-09-25  1.0.0     MRD      First Release
## -- 2025-07-17  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-17) 

This module provides implementation of SARBuffer with random sampling.
"""

from mlpro.bf.data import BufferRnd



# Export list for public API
__all__ = [ 'RandomSARSBuffer' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RandomSARSBuffer(BufferRnd):
    """
    Random Sampling SARBuffer.
    This is just renaming class from BufferRnd

    """
    pass