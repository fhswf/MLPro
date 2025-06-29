## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalypredictors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-04  0.0.0     DA/DS    Creation
## -- 2024-08-23  0.1.0     DA/DS    Creation
## -- 2024-09-27  0.2.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-09-27)

This module provides basic templates for online anomaly prediction in MLPro.
 
"""


from mlpro.bf.ml import Log, Event
from mlpro.bf.math import Function
from mlpro.bf.streams import Log, StreamTask
from mlpro.bf.various import Log
from mlpro.bf.streams import Instance, InstDict
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
                 p_cls_tsf,
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
        
        self.p_cls_tsf = p_cls_tsf
        self.capture_anomalies = {}
        _findings:dict = {},
        


## -------------------------------------------------------------------------------------------------

    def _run(self, p_inst : InstDict):
        
        for inst_id,(inst_type, inst) in p_inst.entries():
                 if inst_type == self.InstTypeNew:
                      for ano_type, fl in self._findings.entries():
                           for finding in fl:
                                if inst in finding:
                                     self.adapt(p_inst=inst, p_ano_type = ano_type)
                                else:
                                      self.adapt(p_inst=inst, p_ano_type = None)

        #self.add_tsf(p_ano_type=p_inst.type, p_tsf= timeseriesforcaster)
## -------------------------------------------------------------------------------------------------

    def _adapt_on_event(self, p_event_id: str, p_event_object: Event) -> bool:
    
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------
        """
        t = type(p_event_object)

        try:
             self._findings[t].append(p_event_object.get_instances())
        except:
             self._findings[t] = (p_event_object.get_instances())


## -------------------------------------------------------------------------------------------------

    def get_anomaly(self, p_anomaly):
        """ 
        Process incoming anomaly data from the anomaly detector.

        parameters
        ----------
        ad_anomaly
            Anomaly data coming from the anomaly detector.

        """

        self.p_anomaly = p_anomaly
        self.captured_anomalies.append(p_anomaly)
        

    def is_posiitive_event(self):
         pass
    

