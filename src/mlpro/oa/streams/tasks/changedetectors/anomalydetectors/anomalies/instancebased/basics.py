## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.instancebased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-28  1.0.0     DA       Creation
## -- 2025-06-08  1.0.1     DA       Review/refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2025-06-08)

This module provides a template class for instance-based anomalies to be used in instance-based
anomaly detection algorithms.
"""


from mlpro.bf.various import TStampType
from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.basics import Anomaly




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyIB ( Anomaly ):
    """
    This is the base class for instance-based anomaly events.

    Parameters
    ----------
    p_id : int
        Anomaly ID. Default value = 0.
    p_status : bool = True
        Status of the anomaly. True marks the beginning of an anomaly, while False indicates its end.
    p_tstamp : TStampType = None
        Time stamp of occurance of anomaly. Default = None.
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
                  p_id = 0, 
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