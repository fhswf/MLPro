## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.normalizers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-30  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-06-30)

This module provides provides the template for online-adaptive stream normalizers.
"""


from mlpro.bf.various import Log
from mlpro.bf.math import Element
from mlpro.bf.math.normalizers import Normalizer
from mlpro.oa.streams import OAStreamTask



# Export list for public API
__all__ = [ 'OAStreamNormalizer' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAStreamNormalizer( Normalizer, OAStreamTask ):
    """
    Template for online-adaptive stream normalizers.

    Parameters
    ----------
    p_name: str, optional
        Name of the task.
    p_range_max:
        Processing range of the task, default is a Thread.
    p_ada:
        True if the task has adaptivity, default is true.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize:
        True for visualization, false by default.
    p_logging:
        Logging level of the task. Default is Log.C_LOG_ALL
    **p_kwargs:
        Additional task parameters
    """

    C_TYPE  = 'OA Normalizer'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = OAStreamTask.C_RANGE_THREAD, 
                  p_ada : bool = True, 
                  p_buffer_size : int = 0,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        OAStreamTask.__init__( self, 
                               p_name = p_name,
                               p_range_max = p_range_max,
                               p_ada = p_ada,
                               p_buffer_size = p_buffer_size,
                               p_duplicate_data = p_duplicate_data,
                               p_visualize = p_visualize,
                               p_logging = p_logging, 
                               **p_kwargs ) 

        Normalizer.__init__( self,
                             p_input_set = None, 
                             p_output_set = None,
                             p_output_elem_cls = Element,
                             p_autocreate_elements = False,
                             **p_kwargs )    