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
from mlpro.oa.streams.tasks.anomalydetectors.basics import AnomalyDetector


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MiniBatchManager(AnomalyDetector):
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
    def __init__(self, mb_data, mb_batch_size ): 
        
        self.mb_data = mb_data # dictionary contains anomaly data/ detector output
        self.mb_batch_size = mb_batch_size #good default for batch size is 32.
        self.mb_batches = self.create_mini_batches()
        self.mb_current_batch = 0

    
## --------------------------------------------------------------------------------------------------
    def create_mini_batches(self):
        """
        Method to be used to create mini_batches from the data.

        Parameters
        ----------
        """
        n_data_points = len(self.mb_data)

        mini_batches = [self.mb_data[i:i + self.mb_batch_size] for i in range(0, n_data_points, self.mb_batch_size)]

        return mini_batches 
    

## -------------------------------------------------------------------------------------------------
    def get_batch(self):
        """
        Method to be used to get the next mini batch for processing.
        
        Parameters
        ----------
        """
        if self.mb_current_batch < len(self.mb_batches):
            batch = self.mb_batches[self.mb_current_batch]
            self.mb_current_batch += 1
            return batch
        else:
            raise StopIteration


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OATimeSeriesForcasterMB (OATimeSeriesForcaster):
    """
    ...
    """

    pass