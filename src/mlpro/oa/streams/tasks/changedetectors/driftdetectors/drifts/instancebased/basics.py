## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.instancebased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-04  1.0.0     DA       Creation
## -- 2025-06-08  1.0.1     DA       Review/refactoring
## -- 2025-06-13  1.1.0     DA       Class Change: param p_id is now initialized to -1
## -- 2025-07-18  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-07-18)

This module provides a template class for instance-based drifts to be used in instance-based drift 
detection algorithms.
"""


from mlpro.bf import TStampType
from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.basics import Drift



# Export list for public API
__all__ = [ 'DriftIB' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftIB (Drift):
    """
    This is the base class for instance-based drift events.

    Parameters
    ----------
    p_id : int = -1
        Drift ID. Default value = -1, indicating that the ID is not set. In that case, the id is
        automatically generated when raising the drift.
    p_status : bool = True
        Status of the drift. True marks the beginning of an drift, while False indicates its end.
    p_tstamp : TStampType = None
        Time stamp of occurance of drift. Default = None.
    p_visualize : bool = False
        Boolean switch for visualisation. Default = False.
    p_raising_object : object = None
        Reference of the object raised. Default = None.
    p_instances : list[Instances] = []
        List of related instances.
    **p_kwargs
        Further optional keyword arguments.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id = -1, 
                  p_status : bool = True,
                  p_tstamp : TStampType = None, 
                  p_visualize = False, 
                  p_raising_object = None, 
                  p_instances : list[Instance] = [],
                  **p_kwargs ):
        
        super().__init__( p_id = p_id, 
                          p_status = p_status,
                          p_tstamp = p_tstamp,
                          p_visualize = p_visualize,
                          p_raising_object = p_raising_object, 
                          **p_kwargs )
        
        self.instances = p_instances