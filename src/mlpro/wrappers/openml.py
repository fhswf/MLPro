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
Ver. 1.0.1 (2022-05-27)

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
    Wrapper class for OpenML as StreamProvider
    """

    C_NAME              = 'OpenML'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'OpenML'
    C_SCIREF_URL        = 'new.openml.org'


## -------------------------------------------------------------------------------------------------
    def __init__(self):

        super().__init__()
        self._stream_list = []
        self._stream_ids = []


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, **p_kwargs) -> list:
        """
        Custom class to get alist of stream objects from OpenML

        Returns
        -------
        list_streams:List
            Returns a list of Streams in OpenML

        """

        list_datasets = openml.datasets.list_datasets(output_format='dict')
        # print(stream_list)

        for d in list_datasets.items():
            try:
                _name = d[1]['name']
            except:
                _name = None
            try:
                _id = d[1]['did']
            except:
                _id = 0
            try:
                _num_instances = d[1]['NumberOfInstances']
            except:
                _num_instances = 0

            s = WrStreamOpenML(_id, _name, _num_instances)

            self._stream_list.append(s)
            self._stream_ids.append(_id)

        return self._stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id) -> Stream:
        """
        Custom class to fetch an OpenML stream object

        Parameters
        ----------
        p_id
            id of the stream to be fetched

        Returns
        -------
        stream: Stream
            Returns the stream corresponding to the id
        """

        stream = self._stream_list[self._stream_ids.index(p_id)]
        return stream






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamOpenML(Stream):
    """
    Wrapper class for Streams from OpenML

    Parameters
    ----------
    p_id
        id of the Stream
    p_name
        name of the stream
    p_num_features
        Number of features of the Stream
    """

    C_NAME = 'OpenML'
    C_SCIREF_TYPE = ScientificObject.C_SCIREF_TYPE_ONLINE


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id, p_name, p_num_instances, **p_kwargs):

        self._downloaded = False
        self.C_ID = self._id = p_id
        self.C_NAME = self._name = p_name
        super().__init__(p_id,
                         p_name,
                         p_num_instances,
                         p_mode=self.C_MODE_SIM)
        self._kwargs = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def __repr__(self):
        return str(dict(id=str(self._id), name=self._name))


## -------------------------------------------------------------------------------------------------
    def _setup(self):
        """
        Custom class to setup feature space of an OpenML stream

        Returns
        -------
        feature_space = MSpace
            Feature Space of an OpenML stream
        """

        feature_space = MSpace()
        _, _, _, features = self._kwargs['dataset'].get_data()
        for feature in features:
            feature_space.add_dim(Dimension(p_name_long=str(feature)))

        return feature_space
        # pass


    ## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        """
        Custom reset method to download and reset an OpenML stream

        Parameters
        ----------
        p_seed
            Seed for resetting the stream
        """

        if not self._downloaded:
            self._dataset = openml.datasets.get_dataset(self._id)
            self._downloaded = True
        return



