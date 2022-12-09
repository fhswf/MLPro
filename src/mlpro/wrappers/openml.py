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
## -- 2022-08-15  1.1.0     DA       Introduction of root class Wrapper
## -- 2022-11-03  1.2.0     DA       Class WrStreamOpenML: refactoring after changes on class 
## --                                bf.streams.Stream
## -- 2022-11-04  1.3.0     DA       - Class WrStreamProviderOpenML: refactoring 
## --                                - Class WrStreamOpenML: removed parent class Wrapper
## -- 2022-11-05  1.4.0     DA       Class WrStreamOpenML: refactoring to make it iterable
## -- 2022-11-08  1.4.1     DA       Corrections
## -- 2022-11-11  1.5.0     DA       Class WrStreamOpenML: new support of optional parameters.
## -- 2022-11-11  1.5.1     LSB      Refactoring for the new target parameter for get_data() method
## -- 2022-11-12  1.5.2     DA       Correction in method WrStreamOpenML._download()
## -- 2022-11-19  1.6.0     DA       Class WrStreamOpenML: 
## --                                - changes due to stream options
## --                                - method _get_string(): new parameter p_name
## -- 2022-12-09  1.6.1     DA       Bugfix: features/labels need to be added under their full name
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.6.1 (2022-12-09)

This module provides wrapper functionalities to incorporate public data sets of the OpenML ecosystem.

Learn more: 
https://www.openml.org/
https://new.openml.org/
https://docs.openml.org/APIs/

"""

import numpy
from mlpro.bf.various import ScientificObject, Log
from mlpro.bf.ops import Mode
from mlpro.wrappers.models import Wrapper
from mlpro.bf.streams import Feature, Label, Instance, StreamProvider, Stream
from mlpro.bf.math import Element, MSpace
import openml





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamProviderOpenML (Wrapper, StreamProvider):
    """
    Wrapper class for OpenML as StreamProvider.

    Parameters
    ----------
    p_logging
        Log level of stream objects (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_NAME              = 'OpenML'
    C_WRAPPED_PACKAGE   = 'openml'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'OpenML'
    C_SCIREF_URL        = 'new.openml.org'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging = Log.C_LOG_ALL):
        Wrapper.__init__(self, p_logging = p_logging)
        StreamProvider.__init__(self, p_logging = p_logging)
        self._stream_list   = []
        self._stream_ids    = []
        self._stream_names  = []


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> list:
        """
        Custom class to get a list of stream objects from OpenML.

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
            list_datasets = openml.datasets.list_datasets(output_format='dict')


            for d in list_datasets.items():
                try:
                    name = d[1]['name']
                except:
                    name = ''
                try:
                    id = d[1]['did']
                except:
                    id = ''
                try:
                    num_instances = d[1]['NumberOfInstances']
                except:
                    num_instances = 0
                try:
                    version = d[1]['Version']
                except:
                    version = 0

                s = WrStreamOpenML( p_id=id, 
                                    p_name=name, 
                                    p_num_instances=num_instances, 
                                    p_version=version, 
                                    p_mode=p_mode,
                                    p_logging=Log.C_LOG_WE )

                self._stream_list.append(s)
                self._stream_ids.append(id)
                self._stream_names.append(name)

        return self._stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id: str = None, p_name: str = None, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> Stream:
        """
        Custom implementation to fetch an OpenML stream object.

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
                stream = self._stream_list[self._stream_ids.index(int(p_id))]
            except ValueError:
                raise ValueError('Stream with id', p_id, 'not found')

        elif p_name is not None:
            try:
                stream = self._stream_list[self._stream_names.index(p_name)]
            except ValueError:
                raise ValueError('Stream with name "' + p_name + '" not found')

        stream.set_mode(p_mode=p_mode)
        stream.switch_logging(p_logging=p_logging)
        stream.log(Log.C_LOG_TYPE_I, 'Ready to access in mode', p_mode)

        return stream





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrStreamOpenML (Stream):
    """
    Wrapper class for Streams from OpenML.

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
        Further stream specific parameters. See https://docs.openml.org/Python-API/ for more informations. 
        In particular, the optional parameters of method openml.datasets.OpenMLDataset.get_data() can
        be handed over here (or later by using method set_options()).
    """

    C_TYPE              = 'Wrapped OpenML stream'
    C_NAME              = ''
    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id, 
                  p_name : str, 
                  p_num_instances : int, 
                  p_version : str, 
                  p_mode = Mode.C_MODE_SIM, 
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        self._downloaded = False
        self.C_ID = self._id = p_id
        self._name = p_name

        Stream.__init__( self,
                         p_id=p_id,
                         p_name=p_name,
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
        Custom reset method to download and reset an OpenML stream.
        """

        # Just to ensure the data download and set up of feature and label space
        self.get_feature_space()
        self.get_label_space()

        self._index = 0


## --------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        if not self._downloaded:
            self._downloaded = self._download()
            if not self._downloaded: return None       

        feature_space = MSpace()

        _, _, _, features = self._dataset
        for feature in features:
            feature_space.add_dim(Feature(p_name_short=str(feature), p_name_long=str(feature)))

        return feature_space


## --------------------------------------------------------------------------------------------------
    def _setup_label_space(self) -> MSpace:
        if not self._downloaded:
            self._downloaded = self._download() 
            if ( not self._downloaded ) or ( self._label == '' ):
                return None       

        label_space = MSpace()
        label_space.add_dim(Label(p_name_short=str(self._label), p_name_long=str(self._label)))
        return label_space


## --------------------------------------------------------------------------------------------------
    def _download(self) -> bool:
        """
        Custom method to download the corresponding OpenML dataset

        Returns
        -------
        bool
            True for the download status of the stream
        """

        self._stream_meta = openml.datasets.get_dataset(self._id)
        try:
            self._label = str(self._kwargs['target']).lstrip()
            self._kwargs['target'] = self._label
        except:
            self._label = self._stream_meta.default_target_attribute
            self._kwargs['target'] = self._label
        try:
            self.C_SCIREF_URL = self._stream_meta.url
        except:
            self.C_SCIREF_URL = ''
        try:
            self.C_SCIREF_AUTHOR = self._stream_meta.creator
            if isinstance(self.C_SCIREF_AUTHOR, list):
                self.C_SCIREF_AUTHOR = ' and '.join(self.C_SCIREF_AUTHOR)
        except:
            self.C_SCIREF_AUTHOR =''
        try:
            self.C_SCIREF_ABSTRACT = self._stream_meta.description
        except:
            self.C_SCIREF_ABSTRACT =''

        self._dataset = self._stream_meta.get_data(dataset_format = 'array', **self._kwargs)

        if self._dataset is not None:
            return True

        else:
            raise ValueError("Dataset not downloaded or not available")


## ------------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to get the next instance of the OpenML stream.

        Returns
        -------
        instance : Instance
            Next instance in the OpenML stream object (None after the last instance in the dataset).
        """

        # 1 Check: end of data stream reached?
        if self._index >= len(self._dataset[0]): raise StopIteration

        # 2 Determine feature data
        feature_data  = Element( self.get_feature_space() )
        feature_data.set_values(self._dataset[0][self._index])

        # 3 Determine label data
        label_space = self.get_label_space()
        if label_space is not None:
            label_data = Element(self.get_label_space())
            label_data.set_values(numpy.asarray([self._dataset[1][self._index]]))
        else:
            label_data = None

        self._index += 1

        return Instance( p_feature_data=feature_data, p_label_data=label_data )