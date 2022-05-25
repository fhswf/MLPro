## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : openml.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-11  0.0.0     DA       Creation
## -- 2022-05-25  1.0.0     LSB      First Release with Stream and StreamProvider class
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-01-11)

This module provides wrapper functionalities to incorporate public data sets of the OpenML ecosystem.

Learn more: 
https://www.openml.org/
https://new.openml.org/
https://docs.openml.org/APIs/

"""

from mlpro.bf.various import ScientificObject
from mlpro.dsm.models import StreamProvider, Stream
import openml




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamProviderOpenML (StreamProvider): 
    """
    """

    C_NAME              = 'OpenML'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'OpenML'
    C_SCIREF_URL        = 'new.openml.org'

## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, **p_kwargs) -> list:

        stream_list = openml.datasets.list_datasets(output_format="dataframe")
        print(stream_list)
        return stream_list

## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id) -> Stream:

        dataset = openml.datasets.get_dataset(p_id)
        stream = WrStreamOpenML(dataset=dataset)
        return stream

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamOpenML(Stream):
    """
    """

    C_NAME = 'OpenML'

    C_SCIREF_TYPE = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR = 'OpenML'
    C_SCIREF_URL = 'new.openml.org'

    def _setup(self):
        pass