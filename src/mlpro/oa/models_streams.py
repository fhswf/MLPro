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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.2 (2022-06-09)

Model classes for stream providers and streams.
"""



#from time import CLOCK_THREAD_CPUTIME_ID
#from itertools import combinations_with_replacement
from mlpro.bf.various import *
from mlpro.bf.ml import *
from mlpro.bf.math import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Feature (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Instance (Element): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream (Mode, LoadSave, ScientificObject):
    """
    Template class for data streams.

    Parameters
    ----------
    p_id
        id of the stream
    p_name
        name of the stream
    p_num_instances
        Number of instances in the stream
    p_mode
        Operation mode. Valid values are stored in constant C_VALID_MODES.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_TYPE          = 'Stream'
    C_NAME          = '????'
    C_ID            = '????'
    C_URL           = '????'


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id,
                 p_name,
                 p_num_instances,
                 p_version,
                 p_mode=Mode.C_MODE_SIM,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_mode=p_mode, p_logging=p_logging)
        self._id = p_id
        self._name = self.C_SCIREF_TITLE = p_name
        self._num_instances = p_num_instances
        self._version = p_version
        self._kwargs = p_kwargs.copy()


    ## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        return self._id


    ## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self.C_SCIREF_TITLE


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
    """

    C_TYPE          = 'Stream Provider'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_stream_list(self, **p_kwargs) -> list:
        """
        Gets a list of provided streams by calling custom method _get_stream_list().

        Returns
        -------
        stream_list : list
            List of provided streams.

        """

        self.log(self.C_LOG_TYPE_I, 'Getting list of streams...')
        stream_list = self._get_stream_list(**p_kwargs)
        self.log(self.C_LOG_TYPE_I, 'Stream found:', len(stream_list))
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
            self.log(self.C_LOG_TYPE_E, 'Stream', str(p_id), 'not found') 

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