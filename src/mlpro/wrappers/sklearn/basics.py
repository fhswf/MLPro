## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers.sklearn
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-11  0.0.0     SP       Creation
## -- 2022-06-11  1.0.0     SP       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-06-11)

This module provides wrapper classes to embed Scikit-learn functionalities into MLPro. Currently, the 
following topics are supported by the wrapper:

- Native data streams

- Selected anomaly detection algorithms

Learn more:
https://www.scikit-learn.org/

"""

from mlpro.bf.various import ScientificObject
from mlpro.wrappers.models import Wrapper




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrapperScikitLearn (Wrapper):
    """
    Root class for all River wrapper classes.
    """

    C_TYPE              = 'Wrapper ScikitLearn'
    C_WRAPPED_PACKAGE   = 'scikit-learn'
    C_MINIMUM_VERSION   = '0.4.0'
    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'Scikit-learn'
    C_SCIREF_URL        = 'scikit-learn.org'
