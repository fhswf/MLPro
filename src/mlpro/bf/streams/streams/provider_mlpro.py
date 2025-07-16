## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams
## -- Module  : provider_mlpro.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-12  0.0.0     DA       Creation 
## -- 2022-11-08  0.1.0     DA       First draft implementation
## -- 2022-12-14  1.0.0     DA       First release
## -- 2023-03-03  1.0.1     SY       Add p_kwargs in StreamProviderMLPro and StreamMLProBase
## -- 2023-04-12  1.0.2     SY       Remove p_kwargs in StreamProviderMLPro
## -- 2023-12-26  1.1.0     DA       StreamProviderMLPro.__init__(): recursive consideration of all 
## --                                subclasses of class StreamMLProBase
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2023-12-26)

This module consists of a native stream provider and a template for builtin streams.

"""

from mlpro.bf.various import Log, ScientificObject
from mlpro.bf.ops import Mode
from mlpro.bf.various import Log
from mlpro.bf.math import Element
from mlpro.bf.streams import *




# Export list for public API
__all__ = [ 'StreamMLProBase',
            'StreamProviderMLPro' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProBase (Stream): 
    """
    Base class for MLPro's native data streams.
    """

    C_ID                = ''
    C_NAME              = '????'
    C_VERSION           = '0.0.0'
    C_NUM_INSTANCES     = 0

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'MLPro'
    C_SCIREF_URL        = 'https://mlpro.readthedocs.io'

## -------------------------------------------------------------------------------------------------
    def __init__( self, p_logging=Log.C_LOG_ALL, **p_kwargs ):

        super().__init__( p_id = self.C_ID, 
                          p_name = self.C_NAME, 
                          p_num_instances = self.C_NUM_INSTANCES, 
                          p_version = self.C_VERSION,
                          p_feature_space = self._setup_feature_space(), 
                          p_label_space = self._setup_label_space(), 
                          p_mode=Mode.C_MODE_SIM,
                          p_logging = p_logging,
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def _reset(self):
        self._index = 0
        self._init_dataset()


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):
        """
        Custom method to generate stream data as a numpy array named self._dataset.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        if self._index == self.C_NUM_INSTANCES: raise StopIteration

        feature_data = Element(self._feature_space)
        feature_data.set_values(p_values=self._dataset[self._index])

        self._index += 1

        return Instance( p_feature_data=feature_data )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamProviderMLPro (StreamProvider): 
    """
    MLPro's builtin provider class for native data streams.
    """

    C_NAME          = 'MLPro'

    C_SCIREF_TYPE   = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR = 'MLPro'
    C_SCIREF_URL    = 'https://mlpro.readthedocs.io'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_seed = None, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)

        self._stream_list      = []
        self._streams_by_id    = {}
        self._streams_by_name  = {}

        mlpro_stream_classes = StreamMLProBase.__subclasses__()
        for cls in mlpro_stream_classes:
            stream = cls(p_seed=p_seed, p_logging=p_logging)
            self._stream_list.append(stream)
            self._streams_by_id[stream.get_id()] = stream
            self._streams_by_name[stream.get_name()] = stream
            mlpro_stream_classes.extend(cls.__subclasses__())


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> list:
        return self._stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream( self, 
                     p_id: str = None, 
                     p_name: str = None, 
                     p_mode=Mode.C_MODE_SIM, 
                     p_logging=Log.C_LOG_ALL, 
                     **p_kwargs ) -> Stream:

        if p_id is not None:
            stream = self._streams_by_id[p_id]
        else:
            stream = self._streams_by_name[p_name]

        stream.switch_logging(p_logging=p_logging)
        stream.set_mode(p_mode=p_mode)
        return stream

