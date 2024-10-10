## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalypredictors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-04  0.0.0     DA/DS    Creation
## -- 2024-08-23  0.1.0     DA/DS    Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-08-23)

This module provides basic templates for online anomaly prediction in MLPro.
 
"""


from mlpro.bf.math import Function
from mlpro.bf.ml import Log
from mlpro.bf.streams import Log, StreamTask
from mlpro.bf.various import Log
from mlpro.oa.streams.tasks.anomalypredictors import AnomalyPredictor, AnomalyPrediction
from mlpro.sl import SLAdaptiveFunction
from mlpro.bf.streams import InstDict



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TimeSeriesForcaster (Function):
    """
    ...
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OATimeSeriesForcaster (TimeSeriesForcaster, SLAdaptiveFunction):
    """
    ...
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyPredictorTSF (AnomalyPredictor):
    """
    ...
    """
    
    C_TYPE = 'Anomaly Predictor TSF'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                p_cls_tsf,
                p_name: str = None, 
                p_range_max=StreamTask.C_RANGE_THREAD, 
                p_ada: bool = True, 
                p_buffer_size: int = 0, 
                p_duplicate_data: bool = False, 
                p_visualize: bool = False, 
                p_logging=Log.C_LOG_ALL, 
                **p_kwargs):
        super().__init__(p_name, p_range_max, p_ada, p_buffer_size, p_duplicate_data, p_visualize, p_logging, **p_kwargs)
    
        self.p_cls_tsf = p_cls_tsf

    ## -------------------------------------------------------------------------------------------------
    
    def _run(self, p_inst : InstDict):
        pass

