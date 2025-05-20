## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.controllers
## -- Module  : wrapper_fct.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-19  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-09-19)

This module provides a wrapper class for MLPro's adaptive functions.

"""


from mlpro.oa.control import OAController




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControllerFct (OAController):
    """
    Wrapper class for controllers based on an online-adaptive function mapping an error to an action.

    Parameters
    ----------
    p_fct : Function
        Function object mapping a control error to an action

    See class Controller for further parameters.
    """

    C_TYPE          = 'OA Controller Fct'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        try:
            self._fct.switch_adaptivity(p_ada)
        except:
            pass
