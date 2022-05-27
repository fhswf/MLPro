## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : openml.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-11  0.0.0     DA       Creation
## -- 2022-05-25  1.0.0     LSB      First Release with Stream and StreamProvider class
## -- 2022-05-27  1.0.1     LSB      Feature space setup
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
from mlpro.oa.models import StreamProvider, Stream
from mlpro.bf.math import *
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
    Wrapper class for Streams from OpenML
    """

    C_NAME = 'OpenML'

    C_SCIREF_TYPE = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR = 'OpenML'
    C_SCIREF_URL = 'new.openml.org'


## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):
        self._kwargs = p_kwargs.copy()
        dataset = self._kwargs['dataset']
        super().__init__(p_mode=self.C_MODE_SIM, dataset=dataset)


## -------------------------------------------------------------------------------------------------
    def _setup(self):
        feature_space = MSpace()
        _, _, _, features = self._kwargs['dataset'].get_data()
        for feature in features:
            feature_space.add_dim(Dimension(p_name_short=feature[0], p_name_long=str(feature)))

        return feature_space
        # pass