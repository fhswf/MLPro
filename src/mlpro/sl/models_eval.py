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
import warnings





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MetricValue(Element):
    """

    Parameters
    ----------
    p_space
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_space):

        Element.__init__(self, p_set = p_space)
        self._epoch = None
        self._cycle = None


## -------------------------------------------------------------------------------------------------
    def set_epoch(self, p_epoch):
        """

        Parameters
        ----------
        p_epoch

        """
        self._epoch = p_epoch


## -------------------------------------------------------------------------------------------------
    def cycle(self, p_cycle):
        """

        Parameters
        ----------
        p_cycle

        """
        self._cycle = p_cycle






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Metric(Log):
    """

    Parameters
    ----------
    p_logging
    """
    C_OBJECTIVE_MINIMIZE = -1
    C_OBJECTIVE_MAXIMIZE = 1

    C_OBJECTIVE = C_OBJECTIVE_MAXIMIZE

## -------------------------------------------------------------------------------------------------
    def __init__(self,p_logging):

        Log.__init__(self, p_logging)

        self._metric_space = self._setup_metric_space()

        if self.C_OBJECTIVE == self.C_OBJECTIVE_MINIMIZE:
            self._score = np.inf
        else:
            self._score = np.inf*(-1)
        self._highscore = -(np.inf)


## -------------------------------------------------------------------------------------------------
    def _setup_metric_space(self) -> ESpace:
        """

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_output_space(self) -> ESpace:
        """

        Returns
        -------

        """
        return self._metric_space


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed):
        """

        Parameters
        ----------
        p_seed

        """
        try:
            self._score = self._reset(p_seed)
            self._highscore = (-1)*(np.inf)
        except:
            warnings.warn("Could not reset " + self.C_NAME + ".")


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """

        Parameters
        ----------
        p_seed

        Returns
        -------

        """
        if self.C_OBJECTIVE == self.C_OBJECTIVE_MINIMIZE:
            self._score = np.inf
        else:
            self._score = np.inf * (-1)

        return self._score


## -------------------------------------------------------------------------------------------------
    def compute(self, p_model, p_data):
        """

        Parameters
        ----------
        p_model
        p_data

        Returns
        -------

        """
        value = self._compute(p_model, p_data)
        metric = MetricValue(self._metric_space)
        metric.set_values(p_values=value)

        if self.C_OBJECTIVE == self.C_OBJECTIVE_MINIMIZE:
            self._score = -(self._update_score(p_value=value))

        elif self.C_OBJECTIVE == self.C_OBJECTIVE_MAXIMIZE:
            self._score = self._update_score(p_value=value)

        return metric


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):
        """

        Parameters
        ----------
        p_model
        p_data

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _update_score(self, p_value) -> float:
        """

        Parameters
        ----------
        p_value

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_current_score(self):
        """

        Returns
        -------

        """
        return self._score


## -------------------------------------------------------------------------------------------------
    def get_current_highscore(self):
        """

        Returns
        -------

        """

        if self._highscore<self._score:
            self._highscore = self._score

        return self._highscore






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MetricAccuracy(Metric):
    """

    Parameters
    ----------
    p_threshold
    p_logging
    """

    C_NAME = 'ACC'

    C_OBJECTIVE = Metric.C_OBJECTIVE_MAXIMIZE


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_threshold = 0,
                 p_logging = Log.C_LOG_ALL):

        Metric.__init__(self, p_logging)
        self._threshold = p_threshold
        self._mappings_good = 0
        self._mappings_total = 0
        self._num_instances = 0
        self._sum = 0



## -------------------------------------------------------------------------------------------------
    def _setup_metric_space(self) -> ESpace:
        """

        Returns
        -------

        """
        self._metric_space = ESpace()
        self._metric_space.add_dim(Dimension(p_name_short="acc", p_name_long="Accuracy"))
        return self._metric_space


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):
        """

        Parameters
        ----------
        p_model
        p_data

        Returns
        -------

        """
        input = p_data[0]
        target = p_data[1]


        output = p_model(input)



        for i in range(0, len(target.get_values()) if isinstance(input, BatchElement) else 1):
            distance = output.get_related_set().distance(target, output)
            self._mappings_total += 1

            if distance < self._threshold:
                self._mappings_good += 1

        acc =  self._mappings_good/self._mappings_total

        return acc


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """

        Parameters
        ----------
        p_seed

        Returns
        -------

        """
        self._num_instances = 0
        self._sum = 0
        self._mappings_total = 0
        self._mappings_good = 0
        return 0


## -------------------------------------------------------------------------------------------------
    def _update_score(self, p_value) -> float:

        """

        Parameters
        ----------
        p_value

        Returns
        -------

        """
        self._sum += p_value
        self._num_instances += 1

        return self._sum / self._num_instances






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MSEMetric(Metric):
    """

    Parameters
    ----------
    p_logging
    """
    C_NAME = 'MSE'

    C_OBJECTIVE = Metric.C_OBJECTIVE_MINIMIZE

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging):

        Metric.__init__(self, p_logging)
        self._num_instances = 0
        self._sum = 0


## -------------------------------------------------------------------------------------------------
    def _setup_metric_space(self) -> ESpace:
        """

        Returns
        -------

        """
        space = ESpace()
        space.add_dim(Dimension(p_name_short='MSE', p_name_long='Mean Squared Error', p_base_set=Dimension.C_BASE_SET_R))

        return space


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        self._num_instances = 0
        self._sum = 0
        return -np.inf


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):
        """

        Parameters
        ----------
        p_model
        p_data

        Returns
        -------

        """
        inputs, targets = p_data[0], p_data[1].get_values()

        outputs = p_model(inputs).get_values()

        mse = np.mean([np.square(np.array(outputs[i]) - np.array(targets[i])) for i in range(len(targets))])

        return mse



## -------------------------------------------------------------------------------------------------
    def _update_score(self, p_value) -> float:
        """

        Parameters
        ----------
        p_value

        Returns
        -------

        """
        self._sum += p_value
        self._num_instances += 1

        return self._sum/self._num_instances