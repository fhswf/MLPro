## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.sl
## -- Module  : models_metrics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-18  0.0.0     LSB      Creation
## -- 2023-07-15  1.0.0     LSB      Release
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18)

This module provides SL metrics classes for supervised learning tasks.
"""


import warnings

import numpy as np

from mlpro.bf import Log
from mlpro.bf.math import *



# Export list for public API
__all__ = [ 'MetricValue',
            'Metric',
            'MetricAccuracy',
            'MSEMetric' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MetricValue(Element):
    """
    This is a custom element class to Store the Metric Values of a Model, with
    additional information about the Epoch and the Cylcle.

    Parameters
    ----------
    p_space: The output space of the Metric.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_space):

        Element.__init__(self, p_set = p_space)
        self._epoch = None
        self._cycle = None


## -------------------------------------------------------------------------------------------------
    def set_epoch(self, p_epoch):
        """
        Set the epoch value to the Metric Element.

        Parameters
        ----------
        p_epoch:int
            The epoch number.

        """

        self._epoch = p_epoch


## -------------------------------------------------------------------------------------------------
    def set_cycle(self, p_cycle):
        """
        Set the cycle id of the metric element.

        Parameters
        ----------
        p_cycle:int
            The cycle id to be set.

        """

        self._cycle = p_cycle





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Metric(Log):
    """
    This class serves as the base class for metric computation for a supervised learning model.

    Parameters
    ----------
    p_logging:
        Log level for the metric.
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
    @staticmethod
    def _setup_metric_space() -> ESpace:
        """
        Setup the metric Space.

        Returns
        -------
        ESpace
            The output space of the metric.
        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_output_space(self) -> ESpace:
        """
        Get the output space of the metric.

        Returns
        -------
        ESpace
            The output space of the metric.
        """

        return self._metric_space


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed):
        """
        Reset the score and highscore of the metric to default value based on the objective. Calls custom _reset
        method for custom applications.

        Parameters
        ----------
        p_seed: Seed for the purpose of reproducibility.

        """
        try:
            self._score = self._reset(p_seed)
            # Shifted out to constructor
            # self._highscore = (-1)*(np.inf)
        except:
            warnings.warn("Could not reset " + self.C_NAME + ".")


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Custom reset method. Please set the current score of the metric in this method, in case reimplemented for
        custom applications.

        Parameters
        ----------
        p_seed: int
            Seed for the purpose of reproducibility.

        Returns
        -------
        score: float
            The score after reset.

        """
        if self.C_OBJECTIVE == self.C_OBJECTIVE_MINIMIZE:
            self._score = np.inf
        else:
            self._score = np.inf * (-1)

        return self._score


## -------------------------------------------------------------------------------------------------
    def compute(self, p_model, p_data):
        """
        Compute the current metric of the model based on the data provided. Calls the custom _compute method for
        custom applications.

        Parameters
        ----------
        p_model:
            model for which the metric is to be calculated.
        p_data:
            Data, based on which the metric shall be calculated.

        Returns
        -------
        Metric: MetricElement
            The current metric value as a Metric Element.
        """
        value = self._compute(p_model, p_data)
        metric = MetricValue(self._metric_space)
        metric.set_values(p_values=value)

        self._score = self._update_score(p_value=value)

        return metric


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):
        """
        Custom method to compute the metric.

        Parameters
        ----------
        p_model:
            Model for which the metric is to be calculated.
        p_data:
            Data, based on which the metric is to be calculated.

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _update_score(self, p_value) -> float:
        """
        Update the current score of the metric.

        Parameters
        ----------
        p_value: the current value of the metric.

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_current_score(self):
        """
        Gets the current score of the metric.

        Returns
        -------
        score
            The current score, since last reset.
        """
        return self._score


## -------------------------------------------------------------------------------------------------
    def get_current_highscore(self):
        """
        Get the current highscore of the metric.

        Returns
        -------
        float
            Current highscore of the metric.
        """
        if self.C_OBJECTIVE == self.C_OBJECTIVE_MINIMIZE:
            if self._highscore<(-(self._score)):
                self._highscore = -self._score

        elif self.C_OBJECTIVE == self.C_OBJECTIVE_MAXIMIZE:
            if self._highscore<self._score:
                self._highscore = self._score


        return self._highscore





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MetricAccuracy(Metric):
    """
    This is Accuracy Metric Object. Calculates the accuracy of Supervised Learning model based on the number of
    correct mappings to total mappings.

    Parameters
    ----------
    p_threshold:
        Threshold for categorizing a good mapping. Default is zero.
    p_logging:
        Log level for the Metric.
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
    @staticmethod
    def _setup_metric_space() -> ESpace:
        """
        Setup the output space of the metric.

        Returns
        -------
        metric_space:ESpace
            The output space of the metric.
        """
        metric_space = ESpace()
        metric_space.add_dim(Dimension(p_name_short="ACC", p_name_long="Accuracy"))
        return metric_space


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):
        """
        Custom compute method for the accuracy metric. Calculates the accuracy of the model.

        Parameters
        ----------
        p_model:
            The model for which the accuracy is to be calculated.

        p_data:
            The data based on which the accuracy is to be calculated.

        Returns
        -------
        acc: MetricValue
            The calculated accuracy of the model, based on the given data, as an MetricValue element.

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
        Custom reset method for the accuracy metric object.

        Parameters
        ----------
        p_seed:int
            Seed for the purposes of reproducibility.

        Returns
        -------
        0
            Zero as the default score of the metric.

        """

        self._num_instances = 0
        self._sum = 0
        self._mappings_total = 0
        self._mappings_good = 0
        return 0


## -------------------------------------------------------------------------------------------------
    def _update_score(self, p_value) -> float:

        """
        Update the score of the accuracy metric based on moving mean.

        Parameters
        ----------
        p_value:
            current metric value.

        Returns
        -------
            Returns the current score of the metric.
        """
        self._sum += p_value
        self._num_instances += 1

        return self._sum / self._num_instances





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MSEMetric(Metric):
    """
    This is a pool metric class for calculating the Mean Squared Error. Mean Squared error is as the name suggests
    mean of the squared sum of difference of model predictions from the target output.

    Parameters
    ----------
    p_logging:
        Log level of the Metric.
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
        Setup the output space of the metric.

        Returns
        -------
        metric_space:ESpace
            The output space of the metric.
        """

        space = ESpace()
        space.add_dim(Dimension(p_name_short='MSE', p_name_long='Mean Squared Error', p_base_set=Dimension.C_BASE_SET_R))

        return space


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Custom reset method for the MSE metric object.

        Parameters
        ----------
        p_seed:int
            Seed for the purposes of reproducibility.

        Returns
        -------
        0
            Zero as the default score of the metric.

        """
        self._num_instances = 0
        self._sum = 0
        return -np.inf


## -------------------------------------------------------------------------------------------------
    def _compute(self, p_model, p_data):
        """
        Custom compute method for the MSE metric. Calculates the Mean squared error based on the current mapping of
        the model.

        Parameters
        ----------
        p_model:
            The model for which the MSE is to be calculated.

        p_data:
            The data based on which the MSE is to be calculated.

        Returns
        -------
        acc: MetricValue
            The calculated MSE of the model, based on the given data, as an MetricValue element.

        """
        inputs, targets = p_data[0], p_data[1].get_values()

        outputs = p_model(inputs).get_values()

        mse = np.mean([np.square(np.array(outputs[i]) - np.array(targets[i])) for i in range(len(targets))])

        return mse


## -------------------------------------------------------------------------------------------------
    def _update_score(self, p_value) -> float:
        """
        Update the score of the MSE metric based on moving mean.

        Parameters
        ----------
        p_value:
            current metric value.

        Returns
        -------
            Returns the current score of the metric.
        """
        self._sum += p_value
        self._num_instances += 1

        return self._sum/self._num_instances