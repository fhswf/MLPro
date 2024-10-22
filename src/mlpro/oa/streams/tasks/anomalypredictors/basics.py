## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalypredictors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-04  0.0.0     DA/DS    Creation
## -- 2024-08-23  0.1.0     DA/DS    Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-08-23)

This module provides basic templates for online anomaly prediction in MLPro.
 
"""


from mlpro.bf.events import Event
from mlpro.oa.streams.tasks import OATask




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyPredictor (OATask):
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
    p_arg1 : str
        Explanation of the first parameter.
    p_arg2 : bool
        Explanation of the second parameter. The default is True.
        
    Attributes
    ----------
    C_MY_CONSTANT = 'My static value'
        Explanation of the public constant C_MY_CONSTANT.

    """
    
    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyPrediction (Event):
    """
    ...
    """

    pass
