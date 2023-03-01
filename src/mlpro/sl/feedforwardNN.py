## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : feedforwardNN.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-01  0.0.0     SY       Creation 
## -- 2023-03-01  0.1.0     SY       Initial design of FNN for MLPro v1.0.0
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2023-03-01)

This module provides model classes of feedforward neural networks for supervised learning tasks. 
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
            Network model
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
