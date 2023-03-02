## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.FNN
## -- Module  : basics.py
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


from mlpro.sl import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FNN(SLAdaptiveFunction):
    """
    This class provides the base class of feedforward neural networks.
    """

    C_TYPE = 'Feedforward NN'


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
    def _map(self, p_input: Element, p_output: Element):
        """
        Maps a multivariate abscissa/input element to a multivariate ordinate/output element. 

        Parameters
        ----------
        p_input : Element
            Abscissa/input element object (type Element)
        p_output : Element
            Setpoint ordinate/output element (type Element)
        """
        
        output = self.forward(input)
        p_output.set_values(output)


## -------------------------------------------------------------------------------------------------
    def _optimize(self):
        """
        This method provides provide a funtionality to call the optimizer of the feedforward network.
        """
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _calc_loss(self, p_act_output:Element, p_pred_output:Element):
        """
        This method provides provide a funtionality to call the loss function of the feedforward
        network.

        Parameters
        ----------
        p_act_output : Element
            Actual output from the buffer.
        p_pred_output : Element
            Predicted output by the SL model.
        """
        
        return self._parameters['loss_fct'](p_act_output.get_values(), p_pred_output.get_values())
