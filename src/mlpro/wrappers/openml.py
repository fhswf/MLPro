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
## -- 2022-06-10  1.0.3     LSB      Code Optmization
## -- 2022-06-13  1.0.4     LSB      Bug Fix
## -- 2022-06-23  1.0.5     LSB      fetching meta data
## -- 2022-06-25  1.0.6     LSB      Refactoring due to new label and instance class, new instance
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.6 (2022-06-25)

This module provides wrapper functionalities to incorporate public data sets of the OpenML ecosystem.

Learn more: 
https://www.openml.org/
https://new.openml.org/
https://docs.openml.org/APIs/

"""

import numpy
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
    def __init__(self, p_logging = Log.C_LOG_ALL):

        super().__init__(p_logging = p_logging)
        self._stream_list = []
        self._stream_ids = []


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, p_logging = Log.C_LOG_ALL, **p_kwargs) -> list:
        """
        Custom class to get alist of stream objects from OpenML

        Returns
        -------
        list_streams:List
            Returns a list of Streams in OpenML

        """
        if len(self._stream_list) == 0:
            list_datasets = openml.datasets.list_datasets(output_format='dict')


            for d in list_datasets.items():
                try:
                    _name = d[1]['name']
                except:
                    _name = ''
                try:
                    _id = d[1]['did']
                except:
                    _id = ''
                try:
                    _num_instances = d[1]['NumberOfInstances']
                except:
                    _num_instances = 0
                try:
                    _version = d[1]['Version']
                except:
                    _version = 0

                s = WrStreamOpenML(_id, _name, _num_instances, _version, p_logging= p_logging)

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

            try:
                stream = self._stream_list[self._stream_ids.index(int(p_id))]

            except:
                self.get_stream_list()
                stream = self._stream_list[self._stream_ids.index(int(p_id))]

            return stream


        except ValueError:
            raise ValueError('Stream id not in the available list')





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
    def __init__(self, p_id, p_name, p_num_instances, p_version, p_logging = Log.C_LOG_ALL, p_mode = Mode.C_MODE_SIM, **p_kwargs):

        self._downloaded = False
        self.C_ID = self._id = p_id
        self.C_NAME = self._name = p_name

        super().__init__(p_id,
                         p_name,
                         p_num_instances,
                         p_version,
                         p_logging = p_logging,
                         p_mode=p_mode)

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
            self._downloaded = self._download()

        self._index = 0


## --------------------------------------------------------------------------------------------------
    def _set_feature_space(self):

        self._feature_space = MSpace()

        _, _, _, features = self._dataset
        for feature in features:
            self._feature_space.add_dim(Feature(p_name_long=str(feature), p_name_short=str(self.C_NAME[0:5])))


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
            self._downloaded = self._download()

        try:

            return self._feature_space

        except:
            self._set_feature_space()
            return self._feature_space


## --------------------------------------------------------------------------------------------------
    def _download(self):
        """
        Custom method to download the corresponding OpenML dataset

        Returns
        -------
        bool
            True for the download status of the stream
        """
        _stream_meta = openml.datasets.get_dataset(self._id)

        self._label_space = MSpace()

        self._label = _stream_meta.default_target_attribute
        self._label_space.add_dim(Label(p_name_long=str(self._label), p_name_short=str(self._label[0:5])))

        try:
            self.C_SCIREF_URL = _stream_meta.url
        except:
            self.C_SCIREF_URL = ''
        try:
            self.C_SCIREF_AUTHOR = _stream_meta.creator
            if isinstance(self.C_SCIREF_AUTHOR, list):
                self.C_SCIREF_AUTHOR = ' and '.join(self.C_SCIREF_AUTHOR)
        except:
            self.C_SCIREF_AUTHOR =''
        try:
            self.C_SCIREF_ABSTRACT = _stream_meta.description
        except:
            self.C_SCIREF_ABSTRACT =''

        self._dataset = _stream_meta.get_data(dataset_format = 'array')
        self._set_feature_space()

        if self._dataset is not None:
            return True

        else:
            raise ValueError("Dataset not downloaded or not available")


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
            _feature_data = Element(self._feature_space)
            _label_data = Element(self._label_space)
            _feature_data.set_values(numpy.delete(self._dataset[0][self._index] , self._dataset[3].index(self._label)))
            _label_data.set_values(numpy.asarray([self._dataset[0][self._index][self._dataset[3].index(self._label)]]))
            _instance = Instance(_feature_data, _label_data)
            self._index += 1

            return _instance

        return None