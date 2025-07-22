## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams
## -- Module  : rnd10d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-13  0.0.0     DA       Creation 
## -- 2022-12-13  1.0.0     DA       First implementation
## -- 2024-06-04  1.0.1     DA       Bugfix: ESpace instead of MSpace
## -- 2025-04-02  1.0.2     DA       Little refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2025-04-02)

This module provides the native stream class StreamMLProRnd10D. This stream provides 1000 instances
with 10-dimensional random feature data and 2-dimensional random label data.
"""

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.math import Element, ESpace, MSpace
from mlpro.bf.streams import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



# Export list for public API
__all__ = [ 'StreamMLProRnd10D' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProRnd10D (StreamMLProBase):
    """
    Demo stream consisting of 1000 instances with 10-dimensional random feature data and 2-dimensional
    label data. All values are in range defined by attribute C_BOUNDARIES.

    Attributes
    ----------
    C_NUM_INSTANCES = 1000
        Number of instances.
    C_BOUNDARIES    = [-10,10]
        Boundaries for all random values.
    """

    C_ID                = 'Rnd10Dx1000'
    C_NAME              = 'Random 10D x 1000'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = 1000

    C_SCIREF_ABSTRACT   = 'Demo stream of 1000 instances with 10-dimensional random feature data and 2-dimensional label data.'

    C_BOUNDARIES        = [-10,10]

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = ESpace()

        for i in range(10):
            feature_space.add_dim( Feature( p_name_short = 'f' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            #p_boundaries = self.C_BOUNDARIES,
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _setup_label_space(self) -> MSpace:
        label_space : MSpace = ESpace()

        for i in range(2):
            label_space.add_dim( Label( p_name_short = 'l' + str(i),
                                        p_base_set = Label.C_BASE_SET_R,
                                        p_name_long = 'Label #' + str(i),
                                        p_name_latex = '',
                                        #p_boundaries = self.C_BOUNDARIES,
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