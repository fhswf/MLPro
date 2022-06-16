## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : sklearn.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-16  0.0.0     LSB      Creation
## -- 2022-mm-dd  1.0.0     LSB      Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-06-16)

This module provides wrapper functionalities to incorporate public data sets of the Sklearn ecosystem.

Learn more:


"""

from mlpro.bf.various import ScientificObject
from mlpro.oa.models import *
from mlpro.bf.math import *
import sklearn
from sklearn import datasets




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamProviderSklearn (StreamProvider):
    """
    Wrapper class for OpenML as StreamProvider
    """

    C_NAME              = 'Sklearn'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'River'
    C_SCIREF_URL        = 'riverml.xyz'

    _load_utils = [
        "fetch_20newsgroups()",
        "fetch_20newsgroups_vectorized(as_frame=True)",
        "fetch_california_housing()",
        "fetch_covtype()",
        "fetch_rcv1()",
        "fetch_kddcup99()",
        "load_diabetes()",
        "load_iris()",
        "load_breast_cancer()",
        "load_wine()",
    ]

## -------------------------------------------------------------------------------------------------
    def __init__(self):
        _data_utils = [
            "clear_data_home",
            "dump_svmlight_file"
        ]

        _datasets = [
            "20newsgroups",
            "20newsgroups_vectorized",
            "california_housing",
            "covtype",
            "rcv1",
            "kddcup99",
            "diabetes",
            "iris",
            "breast_cancer",
            "wine",
        ]
        self._stream_list = []
        self._stream_ids = []
        super().__init__()
        for i in range(len(_datasets)):
            self._stream_ids.append(i)
            # _num_instances = eval("len(sklearn.datasets."+self._load_utils[i]+".data)")
            self._stream_list.append(WrStreamSklearn(self._stream_ids[i],_datasets[i]))


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, **p_kwargs) -> list:
        """
        Custom class to get alist of stream objects from OpenML

        Returns
        -------
        list_streams:List
            Returns a list of Streams in OpenML

        """
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
            stream = self._stream_list[self._stream_ids.index(int(p_id))]
            return stream
        except ValueError:
            raise ValueError('Stream id not in the available list')





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamSklearn(Stream):
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

    C_NAME = 'River'
    C_SCIREF_TYPE = ScientificObject.C_SCIREF_TYPE_ONLINE


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id, p_name, p_num_instances=None, p_version=None, **p_kwargs):

        # self._downloaded = False
        self.C_ID = self._id = p_id
        self.C_NAME = self._name = p_name
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
        Custom reset method to download and reset an OpenML stream

        Parameters
        ----------
        p_seed
            Seed for resetting the stream
        """

        self._index = 0
        self._dataset = eval("sklearn.datasets." + WrStreamProviderSklearn._load_utils[self.C_ID] + ".data")
        self._num_instances = eval("len(sklearn.datasets." + WrStreamProviderSklearn._load_utils[self.C_ID] + ".data)")
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

        # if not self._downloaded:
        #     self._downloaded = self._download()

        try:

            return self._feature_space

        except:

            self._feature_space = MSpace()
            features = eval("sklearn.datasets." +WrStreamProviderSklearn._load_utils[self.C_ID] +".feature_names")
            for feature in features:
                self._feature_space.add_dim(Feature(p_name_long=str(feature), p_name_short=str(self.C_NAME[0:5])))
            return self._feature_space



    ## --------------------------------------------------------------------------------------------------
    # def _download(self):
    #     """
    #     Custom method to download the corresponding OpenML dataset
    #
    #     Returns
    #     -------
    #     bool
    #         True for the download status of the stream
    #     """
    #     self._dataset = iter(eval("river.datasets."+self._name+"()"))
    #     if self._dataset is not None:
    #         return True
    #     else:
    #         raise ValueError("Dataset not downloaded or not available")
    #

## ------------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to get the instances one after another sequentially in the OpenML stream

        Returns
        -------
        instance:
            Next instance in the OpenML stream object (None after the last instance in the dataset).
        """

        if not self._index < self._num_instances:return None
        self._instance.set_values(self._dataset[self._index])
        self._index += 1
        return self._instance

