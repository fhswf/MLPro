## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : models_metrics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-18  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-13)

This module provides SL metrics classes for supervised learning tasks.
"""

from mlpro.bf.various import Log





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Metric(Log):


## -------------------------------------------------------------------------------------------------
    def __init__(self):
        Log.__init__(self)
        self._value = None


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed):

        self._value = self._reset(p_seed)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def compute(self, p_model):

        self._value = self._compute(p_model)
        return self._value

## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model):

        raise NotImplementedError






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MetricAccuracy(Metric):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_threshold = 0,
                 p_buffer_size = 1,
                 p_logging = Log.C_LOG_ALL):

        Metric.__init__(p_logging)
        self._threshold = p_threshold
        self._mappings_good = 0
        self._mappings_total = 0


## -------------------------------------------------------------------------------------------------
    def compute(self, p_model):

        input, target, output = p_model.get_last_mapping()

        distance = output.get_related_set().distance(target, output)
        self._mappings_total += 1

        if distance > self._threshold:
            self._mappings_good += 1

        return self._mappings_good/self._mappings_good


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        self._mappings_total = 0
        self._mappings_good = 0