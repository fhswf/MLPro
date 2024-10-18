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


from typing import List
from mlpro.bf.math import Function
from mlpro.bf.math.basics import Element, MSpace
from mlpro.bf.ml import Async, Log, Task
from mlpro.bf.streams import Log, StreamTask
from mlpro.bf.various import Log
from mlpro.oa.streams.tasks.anomalypredictors import AnomalyPredictor, AnomalyPrediction
from mlpro.sl import SLAdaptiveFunction
from mlpro.bf.streams import InstDict
from mlpro.sl.models_eval import Metric



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
    def __init__(self, 
                 p_input_space: MSpace, 
                 p_output_space: MSpace, 
                 p_output_elem_cls=..., 
                 p_threshold=0, 
                 p_ada: bool = True, 
                 p_buffer_size: int = 0, 
                 p_metrics: List[Metric] = ..., 
                 p_score_metric=None, 
                 p_name: str = None, 
                 p_range_max: int = Async.C_RANGE_PROCESS, 
                 p_autorun=Task.C_AUTORUN_NONE, 
                 p_class_shared=None, 
                 p_visualize: bool = False, 
                 p_logging=Log.C_LOG_ALL, 
                 **p_par):
        super().__init__(p_input_space, 
                         p_output_space, 
                         p_output_elem_cls, 
                         p_threshold, 
                         p_ada, 
                         p_buffer_size, 
                         p_metrics, 
                         p_score_metric, 
                         p_name, 
                         p_range_max, 
                         p_autorun, 
                         p_class_shared, 
                         p_visualize, 
                         p_logging, 
                         **p_par)
        
## -------------------------------------------------------------------------------------------------
    
    def _adapt(self, p_input, p_timestamp, p_ano:bool):
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
        for inst_id, (inst_type, inst) in p_inst.entries():
            if inst_type not in self.known_types:
                self._adapt(p_inst=inst, p_ano_type='Unknown')
            else:
                self._adapt(p_inst=inst, p_ano_type=inst_type)
        pass

## -------------------------------------------------------------------------------------------------
    
    def _adapt(self, p_inst , p_ano_type):
        pass

## -------------------------------------------------------------------------------------------------
    
    def add_tsf(self, p_ano_type, p_tsf):
        pass