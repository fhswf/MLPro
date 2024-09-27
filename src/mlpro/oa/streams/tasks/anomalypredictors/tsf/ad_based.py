## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalypredictors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-04  0.0.0     DA/DS    Creation
## -- 2024-08-23  0.1.0     DA/DS    Creation
## -- 2024-09-27  0.2.0      DS      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-09-27)

This module provides basic templates for online anomaly prediction in MLPro.
 
"""


from mlpro.bf.ml import Log
from mlpro.bf.streams import Log, StreamTask
from mlpro.bf.various import Log
from mlpro.oa.streams.tasks.anomalypredictors.tsf.basics import AnomalyPredictorTSF
from mlpro.oa.streams.tasks.anomalydetectors.basics import Anomaly



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyPredictorAD (AnomalyPredictorTSF, Anomaly):
    """
    Parameters
    -----------
    p_name : str
         Optional name of the task. Default is None.
    p_range_max : int
       Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_buffer_size : int, optional
       
    p_duplicate_data : bool, optional
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool, optional
       Boolean switch for visualisation. Default = False.
    p_logging : int
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.
       
    """
    C_TYPE = 'Anomaly Predictor AD'


## -------------------------------------------------------------------------------------------------

    def __init__(self, 
                 p_name: str = None, 
                 p_range_max=StreamTask.C_RANGE_THREAD, 
                 p_ada: bool = True, 
                 p_buffer_size: int = 0, 
                 p_duplicate_data: bool = False, 
                 p_visualize: bool = False, 
                 p_logging=Log.C_LOG_ALL, 
                 **p_kwargs):
        
        super().__init__(p_name, 
                         p_range_max, 
                         p_ada, 
                         p_buffer_size, 
                         p_duplicate_data, 
                         p_visualize, 
                         p_logging, 
                         **p_kwargs)
        
        self.capture_anomalies = {}


## -------------------------------------------------------------------------------------------------

    def get_anomaly(self, ad_anomaly):
        """ 
        Process incoming anomaly data from the anomaly detector.

        parameters
        ----------
        ad_anomaly
            Anomaly data coming from the anomaly detector.

        """

        self.ad_anomaly = ad_anomaly
        self.captured_anomalies.append(ad_anomaly)
        
