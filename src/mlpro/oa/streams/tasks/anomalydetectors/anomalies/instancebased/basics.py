## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.instancebased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-28  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-02-28)

This module provides a template class for instance-based anomalies to be used in anomaly detection algorithms.
"""


from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.basics import Anomaly




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyIB ( Anomaly ):
    """
    This is the base class for instance-based anomaly events.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_instances : list[Instance],
                  p_id = 0, 
                  p_tstamp = None, 
                  p_visualize = False, 
                  p_raising_object = None, 
                  **p_kwargs ):
        
        super().__init__(p_id, p_tstamp, p_visualize, p_raising_object, **p_kwargs)
        self.instances = p_instances