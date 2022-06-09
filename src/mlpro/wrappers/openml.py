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
## -- 2022-06-09  1.0.2     LSB      Downloading, resetting OpenML stream and handling instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-06-09)

This module provides wrapper functionalities to incorporate public data sets of the OpenML ecosystem.

Learn more: 
https://www.openml.org/
https://new.openml.org/
https://docs.openml.org/APIs/

"""

from mlpro.bf.various import ScientificObject
from mlpro.oa.models import *
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
        try:
            stream = self._stream_list[self._stream_ids.index(p_id)]
        except:
            stream = None
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
    def _reset(self, p_seed=None):
        """
        Custom reset method to download and reset an OpenML stream

        Parameters
        ----------
        p_seed
            Seed for resetting the stream
        """

        if not self._downloaded:
            self._dataset = self._download()
            self._downloaded = True

        self._index = 0

        self._instance = Instance(self.get_feature_space())




    ## --------------------------------------------------------------------------------------------------
    def get_feature_space(self):
        """
        Method to get the feature space of a stream object

        Returns
        -------
        feature_space:
            Returns the Feature space as MSpace of MLPro
        """

        if not self._downloaded:
            self._download()
            self._downloaded = True

        try:

            feature_space = self._feature_space

        except:

            self._feature_space = feature_space = MSpace()
            _, _, _, features = self._dataset
            for feature in features:
                self._feature_space.add_dim(Feature(p_name_long=str(feature), p_name_short=str(self.C_NAME[0:5])))
            feature_space = self._feature_space
        return feature_space



    ## --------------------------------------------------------------------------------------------------
    def _download(self):
        """
        Custom method to download the corresponding OpenML dataset
        """
        self._dataset = openml.datasets.get_dataset(self._id).get_data(dataset_format = 'array')
        return


## ------------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to get the instances one after another sequentially in the OpenML stream

        Returns
        -------
        instance:
            Next instance in the OpenML stream object (None after the last instance in the dataset).
        """

        if self._index < len(self._dataset[0]):
            instance = self._dataset[0][self._index]
            self._index += 1
            return instance

        return None