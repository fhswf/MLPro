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


import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.math import Element
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.streams import Instance
from mlpro.oa.streams import OAStreamTask




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
    p_param_snapshots : bool = False
        If True, snapshots of the normalization parameters are stored for each instance to enable
        a proper normalization of outdated instances.
    **p_kwargs:
        Additional task parameters
    """

    C_TYPE      = 'OA Normalizer'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = OAStreamTask.C_RANGE_THREAD, 
                  p_ada : bool = True, 
                  p_buffer_size : int = 0,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  p_param_shapshots : bool = False,
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

        self._param_snapshots   = p_param_shapshots 
        self._param_snapshot    = None
        self._inst_param_buffer = {}


## -------------------------------------------------------------------------------------------------
    def _store_inst_param( self, p_instance : Instance ):

        if not self._param_snapshots: return self._param_new

        self._inst_param_buffer[p_instance.id] = self._param_snapshot
        return self._param_snapshot


## -------------------------------------------------------------------------------------------------
    def _restore_inst_param( self, p_instance : Instance ):

        if not self._param_snapshots: return self._param_new

        return self._inst_param_buffer.pop(p_instance.id)
    

## -------------------------------------------------------------------------------------------------
    def update_parameters(self, **p_kwargs) -> bool:

        if super().update_parameters(**p_kwargs):
            if self._param_snapshots: self._param_snapshot = self._param_new.copy()
            return True
        
        return False