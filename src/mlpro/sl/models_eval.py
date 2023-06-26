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
import numpy as np

from mlpro.bf.math import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MetricValue(Element):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_space):

        Element.__init__(self, p_set = p_space)
        self._epoch = None
        self._cycle = None


## -------------------------------------------------------------------------------------------------
    def set_epoch(self, p_epoch):
        self._epoch = p_epoch


## -------------------------------------------------------------------------------------------------
    def cycle(self, p_cycle):
        self._cycle = p_cycle






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Metric(Log):


## -------------------------------------------------------------------------------------------------
    def __init__(self):
        Log.__init__(self)
        self._value = None
        self._metric_space = self._setup_metric_space()


## -------------------------------------------------------------------------------------------------
    def _setup_metric_space(self) -> ESpace:

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_output_space(self) -> ESpace:

        return self._metric_space


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed):

        self._value = self._reset(p_seed)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        return 0


## -------------------------------------------------------------------------------------------------
    def compute(self, p_model, p_data):

        self._value = self._compute(p_model, p_data)
        metric = MetricValue(self._metric_space)
        metric.set_values(p_values=self._value)
        return metric


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):

        raise NotImplementedError






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MetricAccuracy(Metric):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_threshold = 0,
                 p_logging = Log.C_LOG_ALL):

        Metric.__init__(p_logging)
        self._threshold = p_threshold
        self._mappings_good = 0
        self._mappings_total = 0


## -------------------------------------------------------------------------------------------------
    def _setup_metric_space(self) -> ESpace:

        self._metric_space = ESpace()
        self._metric_space.add_dim(Dimension(p_name_short="acc", p_name_long="Accuracy"))
        return self._metric_space


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):

        input, target = p_data

        output = p_model(input)

        distance = output.get_related_set().distance(target, output)
        self._mappings_total += 1

        if distance > self._threshold:
            self._mappings_good += 1

        acc =  self._mappings_good/self._mappings_total

        return acc


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        self._mappings_total = 0
        self._mappings_good = 0





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MSEMetric(Metric):


## -------------------------------------------------------------------------------------------------
    def __init__(self):
        Metric.__init__(self)


## -------------------------------------------------------------------------------------------------
    def _setup_metric_space(self) -> ESpace:

        space = ESpace()
        space.add_dim(Dimension(p_name_short='MSE', p_name_long='Mean Squared Error', p_base_set=Dimension.C_BASE_SET_R))

        return space


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):

        inputs, targets = p_data[0].get_values(), p_data[1].get_values()

        outputs = p_model(inputs).get_values()

        mse = np.mean([np.square(np.array(outputs[i]) - np.array(inputs[i])) for i in range(len(inputs))])

        return mse

