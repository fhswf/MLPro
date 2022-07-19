## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa
## -- Module  : models_streams.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-06  0.0.0     DA       Creation
## -- 2022-05-25  0.0.1     LSB      Minor bug fix
## -- 2022-06-02  0.1.0     LSB      Refactoring for list of stream objects in get stream list
## -- 2022-06-04  0.1.1     DA       Specialization in stream providers and streams
## -- 2022-06-09  0.1.2     LSB      Additional attributes to stream object
## -- 2022-06-14  0.1.3     LSB      Enhancement
## -- 2022-06-18  0.1.4     LSB      Logging of stream list based on p_display_list parameter
## -- 2022-06-19  0.1.5     DA       - Class Stream: internal use of self.C_NAME instead of self._name
## --                                - Check/completion of doc strings
## -- 2022-06-25  0.2.5     LSB      New Label class with modified instance class
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.5 (2022-06-25)

Model classes for stream providers and streams. 
"""


from mlpro.bf.various import *
from mlpro.bf.ml import *
from mlpro.bf.math import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Feature (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Label (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Instance:
    """
    Instance class to store the current instance and the corresponding labels of the stream

    Parameters
    ----------
    p_feature_data : Element
        feature data of the instance
    p_label_data : Element
        label data of the corresponding instance

    """

    C_TYPE          = 'Instance'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_feature_data:Element, p_label_data:Element = None, **p_kwargs):

        self._feature_data = p_feature_data
        self._label_data = p_label_data
        self._kwargs = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def get_feature_data(self) -> Element:
        return self._feature_data


## -------------------------------------------------------------------------------------------------
    def get_label_data(self) -> Element:
        return self._label_data


## -------------------------------------------------------------------------------------------------
    def get_kwargs(self):
        return self._kwargs





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream (Mode, LoadSave, ScientificObject):
    """
    Template class for data streams.

    Parameters
    ----------
    p_id
        id of the stream
    p_name : str
        name of the stream
    p_num_instances : int
        Number of instances in the stream
    p_version : str
        Version of the stream
    p_mode
        Operation mode. Valid values are stored in constant C_VALID_MODES.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs
        Further stream specific parameters

    """

    C_TYPE          = 'Stream'

    ## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id=0,
                  p_name:str='',
                  p_num_instances:int=0,
                  p_version:str='',
                  p_mode=Mode.C_MODE_SIM,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs):

        super().__init__(p_mode=p_mode, p_logging=p_logging)
        self._id = p_id
        self.C_NAME = self.C_SCIREF_TITLE = p_name
        self._num_instances = p_num_instances
        self._version = p_version
        self._kwargs = p_kwargs.copy()


    ## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        return self._id


    ## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self.C_NAME


    ## -------------------------------------------------------------------------------------------------
    def get_url(self) -> str:
        return self.C_SCIREF_URL


    ## -------------------------------------------------------------------------------------------------
    def get_num_features(self) -> int:
        return self._num_instances


## -------------------------------------------------------------------------------------------------
    def get_feature_space(self):
        return self.get_feature_space()


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=None):
        """
        Resets stream generator and initializes an internal random generator with the given seed
        value by calling the custom method _reset().

        Parameters
        ----------
        p_seed : int
            Seed value for random generator.

        """

        self._reset(p_seed=p_seed)
        self.log(self.C_LOG_TYPE_W, "\n\n")
        self.log(self.C_LOG_TYPE_W, "Resetting the stream")


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Custom reset method for data stream. See method reset() for more details.

        Parameters
        ----------
        p_seed : int
            Seed value for random generator.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_next(self) -> Instance:
        """
        Returns next data stream instance or None at the end of the stream. The next instance is
        determined by calling the custom method _get_next().

        Returns
        -------
        instance : Instance
            Next instance of data stream or None.

        """

        return self._get_next()


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to determine the next data stream instance. See method get_next() for more
        details.

        Returns
        -------
        instance : Instance
            Next instance of data stream or None.
            
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamProvider (Log, ScientificObject):
    """
    Template class for stream providers.

    Parameters
    ----------
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_TYPE          = 'Stream Provider'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_stream_list(self, p_logging = Log.C_LOG_ALL, **p_kwargs) -> list:
        """
        Gets a list of provided streams by calling custom method _get_stream_list().

        Parameters
        ----------
        p_display_list:bool
            boolean value to log the list of streams

        Returns
        -------
        stream_list : list
            List of provided streams.

        """
        stream_list = self._get_stream_list(p_logging = p_logging ,**p_kwargs)
        self.log(self.C_LOG_TYPE_I, "\n\n\n")
        self.log(self.C_LOG_TYPE_W, 'Getting list of streams...')
        for stream in stream_list:
            self.log(self.C_LOG_TYPE_I, "Stream ID: {:<15} Stream Name: {:<30}".format(stream.C_ID, stream.C_NAME))
        self.log(self.C_LOG_TYPE_I, 'Number of streams found:', len(stream_list),'\n\n\n')
        return stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, **p_kwargs) -> list:
        """
        Custom method to get the list of provided streams. See method get_stream_list() for further
        details.

        Returns
        -------
        stream_list : list
            List of provided streams.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_stream(self, p_id) -> Stream:
        """
        Returns stream with the specified id by calling custom method _get_stream().

        Parameters
        ----------
        p_id : str
            Id of the requested stream.

        Returns
        -------
        s : Stream
            Stream object or None in case of an error.

        """

        self.log(self.C_LOG_TYPE_I, 'Requested stream:', str(p_id))
        s = self._get_stream(p_id)
        if s is None:
            self.log(self.C_LOG_TYPE_E, 'Stream', str(p_id), 'not found\n')

        return s


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id) -> Stream:
        """
        Custom method to get the specified stream. See method get_stream() for further details.

        Parameters
        ----------
        p_id : str
            Id of the requested stream.

        Returns
        -------
        s : Stream
            Stream object or None in case of an error.

        """

        raise NotImplementedError 