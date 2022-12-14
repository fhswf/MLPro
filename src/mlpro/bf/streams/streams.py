## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams
## -- Module  : streams.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-12  0.0.0     DA       Creation 
## -- 2022-11-08  0.1.0     DA       First draft implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2022-11-08)

This module provides native stream classes.

"""

import numpy as np
from numpy.random import default_rng

from mlpro.bf.various import Log, ScientificObject
from mlpro.bf.math import MSpace
from mlpro.bf.ops import Mode
from mlpro.bf.streams.models import *




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
    def __init__( self, p_logging=Log.C_LOG_ALL ):

        super().__init__( p_id = self.C_ID, 
                          p_name = self.C_NAME, 
                          p_num_instances = self.C_NUM_INSTANCES, 
                          p_version = self.C_VERSION,
                          p_feature_space = self._setup_feature_space(), 
                          p_label_space = None, 
                          p_mode=Mode.C_MODE_SIM,
                          p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
    def _reset(self):
        self._index = 0





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProRnd10D (StreamMLProBase):
    """
    Demo stream consisting of 1000 instances of 10-dimensional unlabelled real-valued random numbers.
    """

    C_ID                = 'Rnd10Dx1000'
    C_NAME              = 'Random 10D x 1000'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = 1000

    C_SCIREF_ABSTRACT   = 'Demo stream of 1000 10-dimensional unlabelled real-valued random numbers.'

    C_BOUNDARIES        = [-10,10]

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = MSpace()

        for i in range(10):
            feature_space.add_dim( Feature( p_name_short = 'f' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            p_boundaries = self.C_BOUNDARIES,
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _reset(self):

        try:
            seed = self._random_seed
        except:
            self.set_random_seed()
            seed = self._random_seed

        num = self.C_NUM_INSTANCES
        dim = self._feature_space.get_num_dim()
        f   = self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0]
        t   = self.C_BOUNDARIES[0]

        self._dataset = np.random.RandomState(seed).rand(num, dim) * f + t
        super()._reset()


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self._random_seed = p_seed


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
    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)

        self._stream_list      = []
        self._streams_by_id    = {}
        self._streams_by_name  = {}

        for cls in StreamMLProBase.__subclasses__():
            stream = cls(p_logging=p_logging)
            self._stream_list.append(stream)
            self._streams_by_id[stream.get_id()] = stream
            self._streams_by_name[stream.get_name()] = stream


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

