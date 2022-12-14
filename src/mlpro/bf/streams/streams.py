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
from math import sin, cos, pi

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
                          p_label_space = self._setup_label_space(), 
                          p_mode=Mode.C_MODE_SIM,
                          p_logging = p_logging )


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
class StreamMLProRnd10D (StreamMLProBase):
    """
    Demo stream consisting of 1000 instances with 10-dimensional random feature data and 2-dimensional label data.
    """

    C_ID                = 'Rnd10Dx1000'
    C_NAME              = 'Random 10D x 1000'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = 1000

    C_SCIREF_ABSTRACT   = 'Demo stream of 1000 instances with 10-dimensional random feature data and 2-dimensional label data.'

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
    def _setup_label_space(self) -> MSpace:
        label_space : MSpace = MSpace()

        for i in range(2):
            label_space.add_dim( Label( p_name_short = 'l' + str(i),
                                        p_base_set = Label.C_BASE_SET_R,
                                        p_name_long = 'Label #' + str(i),
                                        p_name_latex = '',
                                        p_boundaries = self.C_BOUNDARIES,
                                        p_description = '',
                                        p_symmetrical = False,
                                        p_logging=Log.C_LOG_NOTHING ) )

        return label_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):

        try:
            seed = self._random_seed
        except:
            self.set_random_seed()
            seed = self._random_seed

        num   = self.C_NUM_INSTANCES
        dim   = self._feature_space.get_num_dim()
        dim_l = self._label_space.get_num_dim()
        f     = self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0]
        t     = self.C_BOUNDARIES[0]

        self._dataset   = np.random.RandomState(seed).rand(num, dim) * f + t
        self._dataset_l = np.random.RandomState(seed).rand(num, dim_l) * f + t


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self._random_seed = p_seed


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        inst       = super()._get_next()
        label_data = Element(self._label_space)
        label_data.set_values(p_values=self._dataset_l[self._index-1])
        inst.set_label_data(p_label_data=label_data)
        return inst





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DoubleSpiral2D (StreamMLProBase):
    """
    """

    C_ID                = 'DoubleSpiral2D'
    C_NAME              = 'Double Spiral 2D x 721'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = 721

    C_SCIREF_ABSTRACT   = 'This benchmark test generates 721 2-dimensional inputs positioned in a double spiral.'

    C_BOUNDARIES        = [-10,10]

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = MSpace()

        for i in range(2):
            feature_space.add_dim( Feature( p_name_short = 'f' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):
        try:
            self._dataset
            return
        except:
            self._dataset = np.empty( (self.C_NUM_INSTANCES, 2))

        center_x1       = ( (self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0]) / 2 ) + self.C_BOUNDARIES[0]
        center_x2       = ( (self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0]) / 2 ) + self.C_BOUNDARIES[0]
        
        radius_x1       = (self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0]) / 2
        radius_step_x1  = radius_x1 / 360
        radius_x2       = (self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0]) / 2
        radius_step_x2  = radius_x2 / 360
        
        radius_sign = 1
        
        for i in range(self.C_NUM_INSTANCES):
            
            bm = i *2 * pi / 180
            self._dataset[i][0] = cos(bm) * radius_x1 * radius_sign + center_x1
            self._dataset[i][1] = sin(bm) * radius_x2 + center_x2
                       
            radius_x1 -= radius_step_x1
            radius_x2 -= radius_step_x2
            if radius_x1 < 0:
                radius_x1       = 0
                radius_step_x1  *= -1
                radius_x2       = 0
                radius_step_x2  *= -1
                radius_sign     = -1        





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

