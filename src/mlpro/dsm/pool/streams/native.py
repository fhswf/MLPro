## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.dsm.pool.streams
## -- Module  : native.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-12  0.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-01-12)

This module provides native stream classes.

"""

from mlpro.bf.various import ScientificObject
from mlpro.dsm.models import StreamProvider, Stream





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamRnd (Stream): 
    """

    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamProviderMLPro (StreamProvider): 
    """
    Builtin provider for native data streams.
    """

    C_NAME              = 'MLPro'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'MLPro'
    C_SCIREF_URL        = 'https://mlpro.readthedocs.io'

## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, **p_kwargs) -> list:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id) -> Stream:
        raise NotImplementedError

