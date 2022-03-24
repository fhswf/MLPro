## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : optuna.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-mm-dd  0.0.0     FN       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2021-mm-dd)
This module provides ...
"""


# import sys
# from mlpro.bf.various import *
# from mlpro.bf.math import *
# from mlpro.bf.data import Buffer
# from mlpro.bf.plot import *
# import random




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Example:
    """
    The __init__ method should be documented in the class level docstring and the docstring itself
    should not go beyond 100 characters length (within the dash separator). Sections inside the 
    docstring can be seperated like the reStructuredText format.
    
    Parameters are documented in the Parameters section.
    Public attributes of classes are documented inisde Attributes section.
    Returns attronites are documented in the Returns section.
    
    A few examples of data types: int / str / bool / list of str / tuple of int / float / None / dict.
    If the parameters have default values, you should add "TYPE, optional" as part of the type
    and "The default is ...." as part of the description.
    
    .. _Further_formatting_information: 
        https://numpydoc.readthedocs.io/en/latest/format.html
    
    Notes
    -----
        The content inside the section should be indented. 
    
    Parameters
    ----------
    p_arg1 : TYPE
        explanation of the first parameter.
    p_arg2 : TYPE, optional
        explanation of the second parameter. The default is True.
        
    Attributes
    ----------
    attr1: TYPE
        explanation of the public attribute attr1.
    """
    
    attr1 = None

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_arg1, p_arg2=True):
        self.attr1 = p_arg1
        pass


## -------------------------------------------------------------------------------------------------
    def example_method(self, p_arg1):
        """
        Example on how to document return type. 
        
        Notes
        -----
            The name of the return value is required for better understanding 
            of the code. The return value is parsed similarly as parameters 
            value, meaning that multiple return value is also possible.
        
        Parameters
        ----------
        p_arg1 : TYPE
            explanation of the first parameter.
                
        Returns
        -------
        p_arg1: TYPE
            Description of the returned value.
        """
        return p_arg1
        

## -------------------------------------------------------------------------------------------------
    def example_method_no_return(self, p_arg1):
        """
        Example on how to document return type. 
        
        Notes
        -----
            When there is no item to be returned, the return section is omitted.
        
        Parameters
        ----------
        p_arg1 : TYPE
            explanation of the first parameter.
                
        """
        return 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Example2:
    """
    ...
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_arg1, p_arg2):
        pass


## -------------------------------------------------------------------------------------------------
    def example_method(self, p_arg1):
        """
        """
        pass
