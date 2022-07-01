## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : sklearn.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-16  0.0.0     LSB      Creation
## -- 2022-06-16  1.0.0     LSB      Release of first version
## -- 2022-06-18  1.0.1     LSB      Stream names as Stream ids
## -- 2022-06-23  1.0.2     LSB      Fetching stream meta data
## -- 2022-06-25  1.0.3     LSB      Refactoring for new label and instance class
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2022-06-dd)

This module provides wrapper functionalities to incorporate public data sets of the Sklearn ecosystem.

Learn more:
https://scikit-learn.org


"""

from mlpro.bf.various import ScientificObject
from mlpro.oa.models import *
from mlpro.bf.math import *
import sklearn
from sklearn import datasets
import numpy




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamProviderSklearn (StreamProvider):
    """
    Wrapper class for Sklearn as StreamProvider
    """

    C_NAME              = 'Sklearn'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'sklearn'
    C_SCIREF_URL        = 'https://scikit-learn.org'

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


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging = Log.C_LOG_ALL):

        self._stream_list = []
        self._stream_ids = self._datasets

        super().__init__(p_logging = p_logging)

        for i in range(len(self._datasets)):
            self._stream_list.append(WrStreamSklearn(self._stream_ids[i],self._datasets[i], p_logging=p_logging))


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, **p_kwargs) -> list:
        """
        Custom class to get alist of stream objects from Sklearn

        Returns
        -------
        list_streams:List
            Returns a list of Streams in Sklearn

        """

        return self._stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id) -> Stream:
        """
        Custom class to fetch an Sklearn stream object

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

        except:
            raise ValueError('Stream id not in the available list')





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamSklearn(Stream):
    """
    Wrapper class for Streams from Sklearn

    Parameters
    ----------
    p_id
        id of the Stream
    p_name
        name of the stream
    p_num_features
        Number of features of the Stream
    """

    C_NAME = 'Sklearn'
    C_SCIREF_TYPE = ScientificObject.C_SCIREF_TYPE_ONLINE


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id, p_name, p_num_instances=None, p_version=None, p_logging = Log.C_LOG_ALL, p_mode= Mode.C_MODE_SIM, **p_kwargs):

        self._downloaded = False
        self.C_ID = self._id = p_id
        self.C_NAME = self._name = p_name

        super().__init__(p_id,
                         p_name,
                         p_num_instances,
                         p_version,
                         p_logging=p_logging,
                         p_mode=p_mode)

        self._kwargs = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def __repr__(self):
        return str(dict(id=str(self._id), name=self._name))


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        """
        Custom reset method to download and reset an Sklearn stream

        Parameters
        ----------
        p_seed
            Seed for resetting the stream
        """

        self._index = 0

        if self._dataset is not None:
           self._downloaded = self._download()

        self._num_instances = len(self._dataset.data)

        try:
            self.C_SCIREF_ABSTRACT = eval("len(sklearn.datasets."
                                     + WrStreamProviderSklearn._load_utils[WrStreamProviderSklearn._datasets.index(self.C_ID)]
                                     + ".DESCR")

        except:
            self.C_SCIREF_ABSTRACT = ''


## --------------------------------------------------------------------------------------------------
    def _set_feature_space(self):

        self._feature_space = MSpace()
        self._label_space = MSpace()


        try:
            features = self._dataset.feature_names

        except:
            self._downloaded = self._download()
            if not isinstance(self._dataset['data'], list):
                features = self._dataset['feature_names']
            else:
                features = ['Attr_1']

        for feature in features:
            self._feature_space.add_dim(Feature(p_name_long=str(feature), p_name_short=str(feature[0:5])))

        for label in self._dataset['target_names']:
            self._label_space.add_dim(Feature(p_name_long=str(label), p_name_short=str(label[0:5])))

## --------------------------------------------------------------------------------------------------
    def get_feature_space(self) -> MSpace:
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

            self._set_feature_space()
            return self._feature_space


## --------------------------------------------------------------------------------------------------
    def _download(self):
        """
        Custom download class that assigns the related sklearn dataset and its functionalities to _dataset attribute
        """

        self._dataset = eval("sklearn.datasets."
                         + WrStreamProviderSklearn._load_utils[WrStreamProviderSklearn._datasets.index(self.C_ID)])


## --------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to get the instances one after another sequentially in the Sklearn stream

        Returns
        -------
        instance:
            Next instance in the Sklearn stream object (None after the last instance in the dataset).
        """

        if not self._index < self._num_instances:return None


        _feature_data = Element(self._feature_space)
        _label_data = Element(self._label_space)
        _feature_data.set_values(self._dataset['data'][self._index])
        _label_data.set_values(numpy.asarray([self._dataset['target'][self._index]]))
        self._index += 1


        return Instance(_feature_data, _label_data)

