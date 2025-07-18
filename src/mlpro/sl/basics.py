## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.sl
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-08  0.0.0     DA       Creation 
## -- 2021-12-10  0.1.0     DA       Took over class AdaptiveFunction from bf.ml
## -- 2022-08-15  0.1.1     SY       Renaming maturity to accuracy
## -- 2022-11-02  0.2.0     DA       Refactoring: methods adapt(), _adapt()
## -- 2022-11-15  0.3.0     DA       Class SLAdaptiveFunction: new parent class AdaptiveFunction
## -- 2023-02-21  0.4.0     SY       - Introduce Class SLNetwork
## --                                - Update Class SLAdaptiveFunction
## -- 2023-02-22  0.4.1     SY       Update Class SLAdaptiveFunction
## -- 2023-03-01  0.4.2     SY       - Renaming module
## --                                - Remove SLNetwork
## -- 2023-03-10  0.4.3     DA       Class SLAdaptiveFunction: refactoring of constructor parameters
## -- 2023-06-20  0.4.4     LSB      Moved the quality check to the adapt online method
## -- 2023-06-20  0.5.0     LSB      New methods: adapt_offline and adapt_online
## -- 2023-07-24  0.5.1     LSB      Merged the new methods back to _adapt method
## -- 2025-07-18  0.6.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.6.0 (2025-07-18)

This module provides model classes for supervised learning tasks. 
"""

from typing import List

from mlpro.bf import Log
from mlpro.bf.data import Buffer
from mlpro.bf.mt import Async, Task
from mlpro.bf.math import Set, MSpace, Element, ESpace
from mlpro.bf.ml import *

from mlpro.sl.models_eval import Metric



# Export list for public API
__all__ = [ 'SLAdaptiveFunction' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLAdaptiveFunction (AdaptiveFunction):
    """
    Template class for an adaptive bi-multivariate mathematical function that adapts by supervised
    learning.

    Parameters
    ----------
    p_input_space : MSpace
        Input space of function
    p_output_space : MSpace
        Output space of function
    p_output_elem_cls 
        Output element class (compatible to/inherited from class Element)
    p_threshold : float
        Threshold for the difference between a setpoint and a computed output. Computed outputs with 
        a difference less than this threshold will be assessed as 'good' outputs. Default = 0.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_buffer_size : int
        Initial size of internal data buffer. Defaut = 0 (no buffering).
    p_name : str
        Optional name of the model. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_autorun : int
        On value C_AUTORUN_RUN method run() is called imediately during instantiation.
        On vaule C_AUTORUN_LOOP method run_loop() is called.
        Value C_AUTORUN_NONE (default) causes an object instantiation without starting further
        actions.    
    p_class_shared
        Optional class for a shared object (class Shared or a child class of it)
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_par : Dict
        Further model specific hyperparameters (to be defined in chhild class).
    """

    C_TYPE = 'Adaptive Function (SL)'
    C_NAME = '????'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_input_space : MSpace,
                  p_output_space : MSpace,
                  p_output_elem_cls=Element,
                  p_threshold=0,
                  p_ada : bool = True,
                  p_buffer_size : int = 0,
                  p_metrics : List[Metric] = [],
                  p_score_metric = None,
                  p_name: str = None,
                  p_range_max: int = Async.C_RANGE_PROCESS,
                  p_autorun = Task.C_AUTORUN_NONE,
                  p_class_shared=None,
                  p_visualize: bool = False, 
                  p_logging=Log.C_LOG_ALL,
                  **p_par ):

        super().__init__( p_input_space = p_input_space,
                          p_output_space = p_output_space,
                          p_output_elem_cls = p_output_elem_cls,
                          p_ada = p_ada,
                          p_buffer_size = p_buffer_size,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_autorun = p_autorun,
                          p_class_shared = p_class_shared,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_par )                  

        self._threshold      = p_threshold
        self._mappings_total = 0  # Number of mappings since last adaptation
        self._mappings_good  = 0  # Number of 'good' mappings since last adaptation
        self._metrics        = p_metrics

        self._score_metric   = p_score_metric or p_metrics[0] if len(p_metrics) != 0 else None
        if (self._score_metric is not None) and (self._score_metric not in self._metrics):
            self._metrics.insert(0, self._score_metric)

        # self.metric_variables = []
        self._metric_space = ESpace()

        for metric in self._metrics:
            dims = metric.get_output_space().get_dims()
            for dim in dims:
                self._metric_space.add_dim(dim.copy())

        self._metric_values = Element(self._metric_space)

        self._sl_model       = self._setup_model()
        self._logging_set    = self._setup_logging_set()


## -------------------------------------------------------------------------------------------------
    def _setup_model(self):
        """
        A method to set up a supervised learning network.
        Please redefine this method according to the type of network, if not provided yet.
        
        Returns
        ----------
            A set up supervised learning model
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_input_space(self):
        return self._input_space


## -------------------------------------------------------------------------------------------------
    def get_output_space(self):
        return self._output_space


## -------------------------------------------------------------------------------------------------
    def _setup_logging_set(self) -> Set :

        return Set()


## -------------------------------------------------------------------------------------------------
    def get_logging_set(self) -> Set:

        return self._logging_set


## -------------------------------------------------------------------------------------------------
    def get_logging_data(self) -> list :

        return []


## -------------------------------------------------------------------------------------------------
    def get_model(self):
        """
        A method to get the supervised learning network.
        
        Returns
        ----------
            A set up supervised learning model
        """

        return self._sl_model


## -------------------------------------------------------------------------------------------------
    def adapt(self, p_input:Element=None, p_output:Element=None, p_dataset=None) -> bool:
        """
        Adaption by supervised learning.
        
        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)
        p_dataset
            A set of data for offline learning
        """

        # if self._adaptivity:
        #     self._set_adapted(self._adapt(p_input, p_output, p_dataset))
        # else:
        #     self._set_adapted(False)
        adapted = Model.adapt(self, p_input=p_input, p_output = p_output, p_dataset=p_dataset)


        return adapted


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_input:Element, p_output:Element, p_dataset:Buffer) -> bool:
        """
        Adaptation algorithm that is called by public adaptation method. This covers online and
        offline supervised learning. Online learning means that the data set is dynamic according to
        the input data, meanwhile offline learning means that the learning procedure is based on a
        static data set.

        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element) for online learning
        p_output : Element
            Setpoint ordinate/output element (type Element) for online learning
        p_dataset : Buffer
            A set of data for offline learning

        Returns
        ----------
            bool
        """

        if ( p_input is not None ) and ( p_output is not None ):
            if not self._adaptivity:
                return False

            self.log(self.C_LOG_TYPE_I, 'Adaptation started')

            adapted = False

            if self._output_space.distance(p_output, self.map(p_input=p_input)) <= self._threshold:
                # Quality of function ok. No need to adapt.
                self._mappings_total += 1
                self._mappings_good += 1

            elif (p_input is not None) and (p_output is not None):
                adapted = self._adapt_online(p_input, p_output)

            if adapted:
                self._mappings_total = 1

                # Second quality check after adaptation
                if self._output_space.distance(p_output, self.map(p_input=p_input)) <= self._threshold:
                    self._mappings_good = 1
                else:
                    self._mappings_good = 0

            else:
                self._mappings_total += 1

        else:
            adapted = False

            self.log(self.C_LOG_TYPE_I, 'Adaptation started')

            if self._adaptivity:
                adapted = self._adapt_offline(p_dataset)

            return adapted

        return adapted


## -------------------------------------------------------------------------------------------------
    def _adapt_online(self, p_input:Element, p_output:Element) -> bool:
        """
        Custom adaptation algorithm for online learning. Please redefine.

        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)

        Returns
        ----------
            bool
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt_offline(self, p_dataset:Buffer) -> bool:
        """
        Custom adaptation algorithm for offline learning. Please redefine.

        Parameters
        ----------
        p_dataset : Buffer
            A set of data for offline learning

        Returns
        ----------
            bool
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_accuracy(self):
        """
        Returns the accuracy of the adaptive function. The accuracy is defined as the relation 
        between the number of successful mapped inputs and the total number of mappings since the 
        last adaptation.
        """

        if self._mappings_total == 0:
            return 0
        return self._mappings_good / self._mappings_total


## -------------------------------------------------------------------------------------------------
    def get_metrics(self) -> list:

        return self._metrics


## -------------------------------------------------------------------------------------------------
    def get_score_metric(self):

        return self._score_metric


## -------------------------------------------------------------------------------------------------
    def get_metric_space(self) -> ESpace:

        return self._metric_space


## -------------------------------------------------------------------------------------------------
    def calculate_metrics(self, p_data) -> Element:

        val = []
        for metric in self._metrics:
            val.append(metric.compute(self, p_data))

        self._metric_values = Element(self._metric_space)
        self._metric_values.set_values(val)
        return self._metric_values


## -------------------------------------------------------------------------------------------------
    def get_current_metrics(self):
        return self._metric_values




