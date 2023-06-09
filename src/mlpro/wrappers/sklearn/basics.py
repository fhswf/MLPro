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
## -- 2022-08-15  1.1.0     DA       Introduction of root class Wrapper
## -- 2022-11-08  1.2.0     DA       Class WrStreamSKlearn: refactoring to make it iterable
## -- 2022-11-19  1.3.0     DA       Method WrStreamSklearn._get_string(): new parameter p_name
## -- 2022-12-09  1.3.1     DA       Bugfix
## -- 2022-12-13  1.3.2     DA       Bugfix
## -- 2023-06-09  1.4.0     DA       Made the wrapper a sub-package
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2023-06-09)

This module provides wrapper functionalities to incorporate public data sets of the Scikit-learn ecosystem.

Learn more:
https://scikit-learn.org

"""


from mlpro.bf.various import ScientificObject
from mlpro.wrappers.models import Wrapper
from mlpro.bf.streams import *
from mlpro.bf.math import *
import sklearn
from sklearn import datasets
import numpy




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamProviderSklearn (Wrapper, StreamProvider):
    """
    Wrapper class for Sklearn as StreamProvider
    """

    C_NAME              = 'Stream Provider Scikit-learn'
    C_WRAPPED_PACKAGE   = 'scikit-learn'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'Scikit-learn'
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

        Wrapper.__init__(self, p_logging=p_logging)
        StreamProvider.__init__(self, p_logging=p_logging)

        self._stream_list = []
        self._stream_ids = self._datasets


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> list:
        """
        Custom class to get alist of stream objects from Sklearn

        Returns
        -------
        list_streams:List
            Returns a list of Streams in Sklearn

        """

        for i in range(len(self._datasets)):
            self._stream_list.append(WrStreamSklearn( p_id=self._stream_ids[i],
                                                      p_name=self._datasets[i], 
                                                      p_mode=p_mode,
                                                      p_logging=Log.C_LOG_WE))

        return self._stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id: str = None, p_name: str = None, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> Stream:
        """
        Custom class to fetch an Sklearn stream object.

        Parameters
        ----------
        p_id : str
            Optional Id of the requested stream. Default = None.
        p_name : str
            Optional name of the requested stream. Default = None.
        p_mode
            Operation mode. Default: Mode.C_MODE_SIM.
        p_logging
            Log level (see constants of class Log). Default: Log.C_LOG_ALL.
        p_kwargs : dict
            Further stream specific parameters.

        Returns
        -------
        s : Stream
            Stream object or None in case of an error.
        """

        self.get_stream_list(p_mode=p_mode, p_logging=p_logging, **p_kwargs)

        if p_id is not None:
            try:
                stream = self._stream_list[int(p_id)]
            except ValueError:
                raise ValueError('Stream with id', p_id, 'not found')

        elif p_name is not None:
            try:
                stream = self._stream_list[self._stream_ids.index(p_name)]
            except ValueError:
                raise ValueError('Stream with name "' + p_name + '" not found')

        stream.set_mode(p_mode=p_mode)
        stream.switch_logging(p_logging=p_logging)
        stream.log(Log.C_LOG_TYPE_I, 'Ready to access in mode', p_mode)

        return stream





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamSklearn (Stream):
    """
    Wrapper class for Streams from Sklearn

    Parameters
    ----------
    p_id
        Id of the stream.
    p_name : str
        Name of the stream. 
    p_num_instances : int
        Number of instances in the stream. 
    p_version : str
        Version of the stream. Default = ''.
    p_feature_space : MSpace
        Optional feature space. Default = None.
    p_label_space : MSpace
        Optional label space. Default = None.
    p_mode
        Operation mode. Valid values are stored in constant C_VALID_MODES.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    p_kwargs : dict
        Further stream specific parameters.
    """

    C_NAME              = 'Sklearn stream'
    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id, 
                  p_name, 
                  p_num_instances : int = 0, 
                  p_version : str = '', 
                  p_logging = Log.C_LOG_ALL, 
                  p_mode= Mode.C_MODE_SIM, 
                  **p_kwargs ):

        self._downloaded = False
        self.C_ID = self._id = p_id
        self._name = p_name

        Stream.__init__( self,
                         p_id=p_id,
                         p_name=self.C_NAME + ' "' + p_name + '"',
                         p_num_instances=p_num_instances,
                         p_version=p_version,
                         p_feature_space=None,
                         p_label_space=None,
                         p_mode=p_mode,
                         p_logging=p_logging,
                         **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def __repr__(self):
        return str(dict(id=str(self._id), name=self._name))


## -------------------------------------------------------------------------------------------------
    def _reset(self):
        """
        Custom reset method to download and reset an Sklearn stream.
        """

        self.get_feature_space()
        self.get_label_space()

        self._index = 0


## --------------------------------------------------------------------------------------------------
    def _setup_feature_space(self)-> MSpace:
        if not self._downloaded:
            self._downloaded = self._download()
            if not self._downloaded: return None       

        feature_space = MSpace()

        try:
            features = self._dataset.feature_names

        except:
            if not isinstance(self._dataset['data'], list):
                features = self._dataset['feature_names']
            else:
                features = ['Attr_1']

        for feature in features:
            feature_space.add_dim(Feature(p_name_short=str(feature)))

        return feature_space


## --------------------------------------------------------------------------------------------------
    def _setup_label_space(self) -> MSpace:
        if not self._downloaded:
            self._downloaded = self._download()
            if not self._downloaded: return None       

        label_space = MSpace()

        try:
            for label in self._dataset['target_names']:
                label_space.add_dim(Label(p_name_short=str(label)))
        except KeyError:
            pass

        return label_space


## --------------------------------------------------------------------------------------------------
    def _download(self):
        """
        Custom download class that assigns the related sklearn dataset and its functionalities to _dataset attribute
        """

        self._dataset = eval("sklearn.datasets."
                         + WrStreamProviderSklearn._load_utils[WrStreamProviderSklearn._datasets.index(self.C_ID)])

        self._num_instances = len(self._dataset.data)
        self.C_SCIREF_ABSTRACT = eval("len(sklearn.datasets."
                                 + WrStreamProviderSklearn._load_utils[WrStreamProviderSklearn._datasets.index(self.C_ID)]
                                 + ".DESCR)")

        return True


## --------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to get the instances one after another sequentially in the Sklearn stream

        Returns
        -------
        instance:
            Next instance in the Sklearn stream object (None after the last instance in the dataset).
        """

        if self._index >= self._num_instances: raise StopIteration

        feature_data = Element(self._feature_space)
        label_data = Element(self._label_space)
        feature_data.set_values(self._dataset['data'][self._index])
        label_data.set_values(numpy.asarray([self._dataset['target'][self._index]]))

        self._index += 1

        return Instance(feature_data, label_data)