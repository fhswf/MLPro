## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : ml
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
    
    .. _Further_formatting_information: 
        https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy
        https://numpydoc.readthedocs.io/en/latest/format.html
    
    Notes
    -----
        The content inside the section should be indented. 
    
    Parameters
    ----------
    p_arg1 : int
        explanation of the first parameter.
    p_arg2 : bool
        explanation of the second parameter.
        
    Attributes
    ----------
    attr1: int
        explanation of the public attribute attr1.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_arg1, p_arg2):
        self.attr1 = p_arg1
        pass


## -------------------------------------------------------------------------------------------------
    def example_method(self, p_arg1):
        """
        Example on how to document return type. 
        
        Notes
        -----
            The name of the return value is optional, but the type is always 
            required. 
        
        Parameters
        ----------
        p_arg1 : int
            explanation of the first parameter.
                
        Returns
        -------
        int
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
        p_arg1 : int
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



