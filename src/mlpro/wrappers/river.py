## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : river.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-14  0.0.0     LSB      Creation
## -- 2022-06-14  1.0.0     LSB      Release of first version
## -- 2022-06-18  1.0.1     LSB      Stream names as Stream ids
## -- 2022-06-23  1.0.2     LSB      Meta data and instances in Numpy format
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-06-23)

This module provides wrapper functionalities to incorporate public data sets of the River ecosystem.

Learn more:
https://www.riverml.xyz/

"""

from mlpro.bf.various import ScientificObject
from mlpro.oa.models import *
from mlpro.bf.math import *
import river
import numpy




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamProviderRiver (StreamProvider):
    """
    Wrapper class for River as StreamProvider
    """

    C_NAME              = 'River'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'River'
    C_SCIREF_URL        = 'riverml.xyz'


## -------------------------------------------------------------------------------------------------
    def __init__(self):
        _datasets = [
                "AirlinePassengers",
                "Bananas",
                "Bikes",
                "ChickWeights",
                "CreditCard",
                "Elec2",
                "Higgs",
                "HTTP",
                "ImageSegments",
                "Insects",
                "Keystroke",
                "MaliciousURL",
                "MovieLens100K",
                "Music",
                "Phishing",
                "Restaurants",
                "SMSSpam",
                "SMTP",
                "SolarFlare",
                "Taxis",
                "TREC07",
                "TrumpApproval",
            ]
        self._stream_list = []
        self._stream_ids = _datasets
        super().__init__()
        for i in range(len(_datasets)):
            _num_instances = eval("river.datasets."+_datasets[i]+"().n_samples")
            self._stream_list.append(WrStreamRiver(self._stream_ids[i],_datasets[i],_num_instances))


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, **p_kwargs) -> list:
        """
        Custom class to get alist of stream objects from River

        Returns
        -------
        list_streams:List
            Returns a list of Streams in River

        """
        return self._stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id) -> Stream:
        """
        Custom class to fetch an River stream object

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
            return stream
        except ValueError:
            raise ValueError('Stream id not in the available list')





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamRiver(Stream):
    """
    Wrapper class for Streams from River

    Parameters
    ----------
    p_id
        id of the Stream
    p_name
        name of the stream
    p_num_features
        Number of features of the Stream
    """

    C_NAME = 'River'
    C_SCIREF_TYPE = ScientificObject.C_SCIREF_TYPE_ONLINE


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id, p_name, p_num_instances=None, p_version=None, **p_kwargs):

        self._downloaded = False
        self.C_ID = self._id = p_id
        self.C_NAME = self._name = p_name
        try:
            self.C_SCIREF_URL = eval("river.datasets."+self._name+"().url")
        except:
            self.C_SCIREF_URL = ''
        try:
            self.C_SCIREF_ABSTRACT = eval("river.datasets."+self._name+"().desc")
        except:
            self.C_SCIREF_ABSTRACT = ''
        super().__init__(p_id,
                         p_name,
                         p_num_instances,
                         p_version,
                         p_mode=self.C_MODE_SIM)
        self._kwargs = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def __repr__(self):
        return str(dict(id=str(self._id), name=self._name))


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        """
        Custom reset method to download and reset an River stream

        Parameters
        ----------
        p_seed
            Seed for resetting the stream
        """

        if not self._downloaded:
            self._downloaded = self._download()

        self._index = 0
        self._dataset = iter(eval("river.datasets."+self._name+"()"))
        self._instance = Instance(self.get_feature_space())




## --------------------------------------------------------------------------------------------------
    def get_feature_space(self) -> MSpace:
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

            self._feature_space = MSpace()
            features = next(self._dataset)[0].keys()
            for feature in features:
                self._feature_space.add_dim(Feature(p_name_long=str(feature), p_name_short=str(self.C_NAME[0:5])))
            return self._feature_space



    ## --------------------------------------------------------------------------------------------------
    def _download(self):
        """
        Custom method to download the corresponding River dataset

        Returns
        -------
        bool
            True for the download status of the stream
        """
        self._dataset = iter(eval("river.datasets."+self._name+"()"))
        if self._dataset is not None:
            return True
        else:
            raise ValueError("Dataset not downloaded or not available")


## ------------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to get the instances one after another sequentially in the River stream

        Returns
        -------
        instance:
            Next instance in the River stream object (None after the last instance in the dataset).
        """

        if not self._index < self._num_instances:return None
        _instance_dict = next(self._dataset)
        self._instance.set_values(numpy.asarray(list(_instance_dict[0].values())))
        self._index += 1
        return self._instance

