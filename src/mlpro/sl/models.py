## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : models.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-08  0.0.0     DA       Creation 
## -- 2021-12-10  0.1.0     DA       Took over class AdaptiveFunction from bf.ml
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2021-12-10)

This module provides model classes for supervised learning tasks. 
"""

from mlpro.bf.ml import *


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdaptiveFunction(Model, Function):
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
    p_buffer_size : int
        Initial size of internal data buffer. Default = 0 (no buffering).
    p_ada : bool
        Boolean switch for adaptivity. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_par : Dict
        Further model specific parameters (to be specified in child class).

    """

    C_TYPE = 'Adaptive Function'
    C_NAME = '????'

    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_input_space: MSpace,
                 p_output_space: MSpace,
                 p_output_elem_cls=Element,
                 p_threshold=0,
                 p_buffer_size=0,
                 p_ada=True,
                 p_logging=Log.C_LOG_ALL,
                 **p_par):

        Model.__init__(self, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging, **p_par)
        Function.__init__(self, p_input_space=p_input_space, p_output_space=p_output_space,
                          p_output_elem_cls=p_output_elem_cls)
        self._threshold = p_threshold
        self._mappings_total = 0  # Number of mappings since last adaptation
        self._mappings_good = 0  # Number of 'good' mappings since last adaptation

    ## -------------------------------------------------------------------------------------------------
    def adapt(self, p_input: Element, p_output: Element) -> bool:
        """
        Adaption by supervised learning.
        
        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)

        """

        if not self._adaptivity:
            return False
        self.log(self.C_LOG_TYPE_I, 'Adaptation started')

        # Quality check
        if self._output_space.distance(p_output, self.map(p_input)) <= self._threshold:
            # Quality of function ok. No need to adapt.
            self._mappings_total += 1
            self._mappings_good += 1

        else:
            # Quality of function not ok. Adaptation is to be triggered.
            self._set_adapted(self._adapt(p_input, p_output))
            if self.get_adapted():
                self._mappings_total = 1

                # Second quality check after adaptation
                if self._output_space.distance(p_output, self.map(p_input)) <= self._threshold:
                    self._mappings_good = 1
                else:
                    self._mappings_good = 0

            else:
                self._mappings_total += 1

        return self.get_adapted()

    ## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_input: Element, p_output: Element) -> bool:
        """
        Custom adaptation algorithm that is called by public adaptation method. Please redefine.

        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)

        """

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        """
        Returns the maturity of the adaptive function. The maturity is defined as the relation 
        between the number of successful mapped inputs and the total number of mappings since the 
        last adaptation.
        """

        if self._mappings_total == 0:
            return 0
        return self._mappings_good / self._mappings_total


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLScenario(Scenario):
    """
    To be designed.
    """

    C_TYPE = 'SL-Scenario'


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTraining(Training):
    """
    To be designed.
    """

    C_NAME = 'SL'
