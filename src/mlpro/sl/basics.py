## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.3 (2023-03-10)

This module provides model classes for supervised learning tasks. 
"""


from mlpro.bf.ml import *





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
        self._sl_model       = self._setup_model()


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
            self._set_adapted(self._adapt(p_input, p_output, p_dataset))
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
            adapted = self._adapt_online(p_input, p_output)
        else:
            adapted = self._adapt_offline(p_dataset)

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
## -------------------------------------------------------------------------------------------------
class SLScenario (Scenario):
    """
    To be designed.
    """

    C_TYPE = 'SL-Scenario'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_ada: bool = True,
                 p_cycle_limit: int = 0,
                 p_visualize: bool = True, 
                 _logging=Log.C_LOG_ALL):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTraining (Training):
    """
    To be designed.
    """

    C_NAME = 'SL'


## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):
        raise NotImplementedError
