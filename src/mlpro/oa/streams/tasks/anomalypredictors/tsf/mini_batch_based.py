## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalypredictors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-04  0.0.0     DA/DS    Creation
## -- 2024-08-23  0.1.0     DA/DS    Creation
## -- 2024-09-27  0.2.0       DS     Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-09-27)

This module provides template for managing mini-batches for time series forcasting tasks in MLPro.
 
"""


from mlpro.oa.streams.tasks.anomalypredictors.tsf.basics import OATimeSeriesForcaster
from mlpro.oa.streams.tasks.anomalydetectors.basics import AnomalyDetector, Anomaly
from mlpro.bf.streams import Instance


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MiniBatchManager():
    """
    This module implements a template for managing mini-batches for time series forcasting tasks in MLPro.

    Parameters
    ----------
    data 
        Time series data to be split into batches.
    batch_size 
        Size of a mini-batch.

    """
## --------------------------------------------------------------------------------------------------    
    def __init__(self, p_batchno, p_batch_size: int) :

        self.batchno = p_batchno
        self.batch_size = p_batch_size
        self.data = []


## --------------------------------------------------------------------------------------------------    
    def add_data(self, **p_kwargs):
        pass


## --------------------------------------------------------------------------------------------------    
    def _add_data(self, p_anomaly: Anomaly) :
        
        self.data.append(p_anomaly)

## --------------------------------------------------------------------------------------------------
    def create_mini_batches(self):
        """
        Method to be used to create mini_batches from the data.

        Parameters
        ----------
        """
        n_data_points = len(self.data)

        mini_batches = [self.data[i:i + self.p_batch_size] for i in range(0, n_data_points, self.p_batch_size)]

        return mini_batches 


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMiniBatchManager ():
    """
    ...
    """
## --------------------------------------------------------------------------------------------------    
    def __init__(self) :
        pass

    
## --------------------------------------------------------------------------------------------------    
    def _add_data(self, p_inst : Instance ):
        pass


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OATimeSeriesForcasterMB (OATimeSeriesForcaster):
    """
    ...
    """

    pass