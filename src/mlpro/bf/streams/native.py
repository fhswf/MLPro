## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.pool.streams
## -- Module  : native.py
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

from mlpro.bf.various import Log, ScientificObject
from mlpro.bf.math import MSpace
from mlpro.bf.ops import Mode
from mlpro.bf.streams import *




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
class StreamMLProRnd2D (StreamMLProBase):
    """
    Demo stream consisting of 1000 instances of 2-dimensional unlabelled real-valued random numbers.
    """

    C_ID                = 'Rnd2D'
    C_NAME              = 'Random 2D'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = 1000

    C_SCIREF_ABSTRACT   = 'Demo stream of 2-dimensional unlabelled real-valued random numbers.'

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = MSpace()

        feature_space.add_dim( Feature( p_name_short = 'f1',
                                        p_base_set = Feature.C_BASE_SET_R,
                                        p_name_long = 'Feature #1',
                                        p_name_latex = '',
                                        p_boundaries = [],
                                        p_description = '',
                                        p_symmetrical = False,
                                        p_logging=Log.C_LOG_NOTHING ) )

        feature_space.add_dim( Feature( p_name_short = 'f2',
                                        p_base_set = Feature.C_BASE_SET_R,
                                        p_name_long = 'Feature #2',
                                        p_name_latex = '',
                                        p_boundaries = [],
                                        p_description = '',
                                        p_symmetrical = False,
                                        p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProRnd3D (StreamMLProRnd2D):

    C_NAME              = 'Random 3D'

    def _setup_feature_space(self) -> MSpace:
        return super()._setup_feature_space()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProRnd10D (StreamMLProRnd3D):

    C_NAME              = 'Random 10D'





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

        # Determine all stream classes inherited from StreamMLProBase



## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> list:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id, p_mode=Mode.C_MODE_SIM, p_logging=Log.C_LOG_ALL, **p_kwargs) -> Stream:
        raise NotImplementedError

