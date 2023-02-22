## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : models.py
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.1 (2023-02-22)

This module provides model classes for supervised learning tasks. 
"""


from mlpro.bf.ml import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLNetwork:
    """
    This class provides the base class of a supervised learning network.

    Parameters
    ----------
    p_input_size : int
        Input size of the network. Default = None
    p_output_size : int
        Output size of the network. Default = None
    p_hyperparameter : HyperParamTuple
        Related hyperparameter tuple of the network. Default = None
    p_kwargs : Dict
        Further model specific parameters.
     """

    C_TYPE = 'SLNetwork'
    C_NAME = '????'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_input_size:int=None,
                 p_output_size:int=None,
                 p_hyperparameter:HyperParamTuple=None,
                 **p_kwargs):
        
        self._input_size            = p_input_size
        self._output_size           = p_output_size
        self._hyperparameter_tuple  = p_hyperparameter
        if p_kwargs is not None:
            self._parameters        = p_kwargs
        else:
            self._parameters        = {}

        hp = self._hyperparameter_tuple.get_hyperparam()
        for idx in hp.get_dim_ids():
            par_name = hp.get_related_set().get_dim(idx).get_name_short()
            par_val  = hp.get_value(idx)
            self._parameters[par_name] = par_val

        if ( self._input_size is None ) or ( self._output_size is None ):
            raise ParamError('Input size and/or output size of the network are not defined.')
        
        self._sl_model = self._setup_model()


## -------------------------------------------------------------------------------------------------
    def _setup_model(self):
        """
        A method to set up a network. Additionally, the hyperparameters are stored in self._parameters.
        Please redefine this method.

        What is mandatory to be set up?
        1) The structure of the model
        2) Set up optimizer
        3) Set up loss functions

        Optionally, more advanced settings related to the network can be added, e.g. weight
        initialization, gradient monitoring, noises incorporation, etc.

        Returns
        ----------
        output : Element
            Output data
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def forward(self, p_input:Element) -> Element:
        """
        Custom forward propagation in neural networks to generate some output that can be called by
        an external method. Please redefine.

        Parameters
        ----------
        p_input : Element
            Input data

        Returns
        ----------
        output : Element
            Output data
        """

        raise NotImplementedError





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
    p_buffer_size : int
        Initial size of internal data buffer. Default = 0 (no buffering).
    p_ada : bool
        Boolean switch for adaptivity. Default = True.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : Dict
        Further model specific parameters (to be specified in child class).
    """

    C_TYPE = 'Adaptive Function (SL)'
    C_NAME = '????'


## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_input_space: MSpace,
                  p_output_space:MSpace,
                  p_output_elem_cls=Element,
                  p_net_model:SLNetwork=None,
                  p_threshold=0,
                  p_buffer_size=0,
                  p_ada:bool=True,
                  p_visualize:bool=False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        super().__init__( p_input_space=p_input_space,
                          p_output_space=p_output_space,
                          p_output_elem_cls=p_output_elem_cls,
                          p_buffer_size=p_buffer_size,
                          p_ada=p_ada,
                          p_visualize=p_visualize,
                          p_logging=p_logging,
                          **p_kwargs )                  
        
        self._threshold      = p_threshold
        self._mappings_total = 0  # Number of mappings since last adaptation
        self._mappings_good  = 0  # Number of 'good' mappings since last adaptation
        
        if p_net_model is None:
            self._net_model  = self._setup_model()
        else:
            self._net_model  = p_net_model

        if self._net_model is None:
            raise ParamError("Please assign your network model to self._net_model")


## -------------------------------------------------------------------------------------------------
    def _setup_model(self) -> SLNetwork:
        """
        A method to set up a supervised learning network.
        Please redefine this method according to the type of network.
        
        Returns
        ----------
        Set up model under SLNetwork type
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def adapt(self, p_input:Element, p_output:Element) -> bool:
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
