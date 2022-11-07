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
## -- 2022-06-25  1.0.3     LSB      Refactoring for label and instance class
## -- 2022-08-15  1.1.0     DA       Introduction of root class Wrapper
## -- 2022-11-03  1.1.1     LSB      Bug fix for river update
## -- 2022-11-03  1.2.0     DA       - Refactoring
## --                                - Class WrStreamRiver: removed parent class Wrapper
## -- 2022-11-07  1.3.0     DA       Class WrStreamOpenML: refactoring to make it iterable
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2022-11-07)

This module provides wrapper functionalities to incorporate public data sets of the River ecosystem.

Learn more:
https://www.riverml.xyz/

"""

from mlpro.bf.various import ScientificObject
from mlpro.wrappers.models import Wrapper
from mlpro.bf.streams import *
from mlpro.bf.math import *
from river import datasets
import numpy





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamProviderRiver (Wrapper, StreamProvider):
    """
    Wrapper class for River as StreamProvider
    """

    C_NAME              = 'River'
    C_WRAPPED_PACKAGE   = 'river'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'River'
    C_SCIREF_URL        = 'riverml.xyz'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging = Log.C_LOG_ALL):

        self._stream_ids = [
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

        Wrapper.__init__(self, p_logging=p_logging)
        StreamProvider.__init__(self, p_logging = p_logging)


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> list:
        """
        Custom class to get alist of stream objects from River.

        Parameters
        ----------
        p_mode
            Operation mode. Default: Mode.C_MODE_SIM.
        p_logging
            Log level of stream objects (see constants of class Log). Default: Log.C_LOG_ALL.
        p_kwargs : dict
            Further stream specific parameters.

        Returns
        -------
        stream_list : list
            List of provided streams.
        """

        if len(self._stream_list) == 0:
            for i, stream_id in enumerate(self._stream_ids):
                self._stream_list.append( WrStreamRiver( p_id=stream_id,
                                                         p_name=stream_id,
                                                         p_num_instances=eval("datasets."+ stream_id + "().n_samples"),
                                                         p_version='',
                                                         p_mode=p_mode,
                                                         p_logging=p_logging,
                                                         **p_kwargs) )

        return self._stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> Stream:
        """
        Custom class to fetch an River stream object.

        Parameters
        ----------
        p_id : str
            Id of the requested stream.
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

        try:
            stream = self._stream_list[self._stream_ids.index(p_id)]
            stream.set_mode(p_mode=p_mode)
            stream.switch_logging(p_logging=p_logging)
            return stream

        except ValueError:
            raise ValueError('Stream id not in the available list')





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamRiver (Stream):
    """
    Wrapper class for Streams from River.

    Parameters
    ----------
    p_id
        Optional id of the stream. Default = 0.
    p_name : str
        Optional name of the stream. Default = ''.
    p_num_instances : int
        Optional number of instances in the stream. Default = 0.
    p_version : str
        Optional version of the stream. Default = ''.
    p_feature_space : MSpace
        Optional feature space. Default = None.
    p_label_space : MSpace
        Optional label space. Default = None.
    p_mode
        Operation mode. Default: Mode.C_MODE_SIM.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    p_kwargs : dict
        Further stream specific parameters.
    """

    C_NAME              = 'River stream'
    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id = 0,
                  p_name : str = '',
                  p_num_instances : int = 0,
                  p_version : str = '',
                  p_feature_space : MSpace = None,
                  p_label_space : MSpace = None,
                  p_mode = Mode.C_MODE_SIM,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        self._downloaded = False
        self.C_ID = self._id = p_id
        self._name = p_name

        try:
            self.C_SCIREF_URL = eval("datasets."+self._name+"().url")

        except:
            self.C_SCIREF_URL = ''

        try:
            self.C_SCIREF_ABSTRACT = eval("datasets."+self._name+"().desc")

        except:
            self.C_SCIREF_ABSTRACT = ''

        Stream.__init__( self, 
                         p_id=p_id,
                         p_name=self.C_NAME + ' "' + p_name + '"',
                         p_num_instances=p_num_instances,
                         p_version=p_version,
                         p_feature_space=p_feature_space,
                         p_label_space=p_label_space,
                         p_mode=p_mode,
                         p_logging=p_logging,
                         **p_kwargs )

        self._label = 'Label'


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
        self._dataset = iter(eval("datasets."+self._name+"()"))
        # self._instance = Instance(self.get_feature_space())


## --------------------------------------------------------------------------------------------------
    def _set_feature_space(self):

        self._feature_space = MSpace()

        features = next(self._dataset)[0].keys()
        for feature in features:
            self._feature_space.add_dim(Feature(p_name_long=str(feature), p_name_short=str(self.C_NAME[0:5])))
        self._label_space = MSpace()

        if isinstance(next(self._dataset)[1], dict):
            self._label = next(self._dataset)[1].keys()
            for label in self._label:
                self._label_space.add_dim(Label(p_name_long=str(label), p_name_short=str(label[0:5])))

        else:
            self._label_space.add_dim(Label(p_name_long=str(self._label), p_name_short=str(self._label[0:5])))


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
            self._set_feature_space()
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

        self._dataset = iter(eval("datasets."+self._name+"()"))
        self._set_feature_space()

        if self._dataset is not None:
            return True

        else:
            raise ValueError("Dataset not downloaded or not available")


## ------------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to get the next instance of the River stream.

        Returns
        -------
        instance : Instance
            Next instance in the River stream object (None after the last instance in the dataset).
        """

        # 1 Check: end of data stream reached?
        if self._index >= self._num_instances: raise StopIteration

        # 2 Determine feature data
        instance_dict = next(self._dataset)
        feature_data  = Element(self._feature_space)
        feature_data.set_values(list(instance_dict[0].values()))

        # 3 Determine label data
        label_data    = Element(self._label_space)
        if isinstance(instance_dict[1], dict):
            label_data.set_values(numpy.asarray(list(instance_dict[1].values())))
        else: 
            label_data.set_values(numpy.asarray([instance_dict[1]]))

        self._index += 1

        return Instance( p_feature_data=feature_data, p_label_data=label_data )

