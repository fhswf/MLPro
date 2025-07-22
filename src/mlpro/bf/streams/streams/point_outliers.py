## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams
## -- Module  : point_outliers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-02-05  0.0.0     DA       Creation
## -- 2024-02-06  1.0.0     DA       First Release
## -- 2024-04-26  1.1.0     DA       Refactoring: replaced parameter p_outlier_frequency by
## --                                p_outlier_rate
## -- 2024-06-04  1.1.1     DA       Bugfix: ESpace instead of MSpace
## -- 2025-06-22  1.2.0     DA       Code cleanup and refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-06-22)

This module provides a multivariate benchmark stream with configurable baselines per feature and
additional random point outliers.

"""


import random
import math

from mlpro.bf.various import Log
from mlpro.bf.math import Element, MSpace, ESpace
from mlpro.bf.streams import Feature, Instance
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



# Export list for public API
__all__ = [ 'StreamMLProPOutliers' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProPOutliers (StreamMLProBase):
    """
    This benchmark stream provides multidimensional instances with configurable baselines 
    per feature. Additionally, random point outliers per feature are induced.

    Parameters
    ----------
    p_num_instances : int
        Total number of instances. The value '0' means indefinite. Default = 1000.
    p_functions : list[str]
        List of mathematical functions per feature. 
    p_outlier_rate : float
        A value in [0,1] that defines the number of random outliers in % per feature.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'PointOutliersND'
    C_NAME                  = 'Point Outliers N-Dim'
    C_TYPE                  = 'Benchmark'
    C_VERSION               = '1.1.0'
    C_SCIREF_ABSTRACT       = 'This benchmark stream provides multidimensional instances with configurable baselines per feature. Additionally, random point outliers per feature are induced.'
    C_BOUNDARIES            = [0,0]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_instances : int = 1000,
                  p_functions : list[str] = ['sin', 'cos', 'const', 'lin'],
                  p_outlier_rate : float = 0.05,
                  p_seed = None,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        self._num_dim            = len(p_functions)
        self.C_NUM_INSTANCES     = p_num_instances
        self._functions          = p_functions
        self.p_outlier_rate      = p_outlier_rate
        self._fct_methods        = []

        for fct in p_functions:
            self._fct_methods.append( getattr(self, '_fct_' + fct) )

        self.set_random_seed(p_seed=p_seed)

        StreamMLProBase.__init__ (self,
                                  p_logging=p_logging,
                                  **p_kwargs)
        

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = ESpace()

        for i, fct in enumerate(self._functions):
            feature_space.add_dim( Feature( p_name_short = 'f_' + str(i) + fct,
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i) + ': ' + fct,
                                            p_name_latex = '',
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        # 0 Preparation
        if self.C_NUM_INSTANCES== 0: pass
        elif self._index == self.C_NUM_INSTANCES: raise StopIteration

        values       = []
        feature_data = Element(self._feature_space)     

        for fct_method in self._fct_methods:
            outlier = random.uniform(0,1) <= self.p_outlier_rate
            values.append( fct_method(self._index, outlier) )   

        feature_data.set_values(values)

        self._index += 1

        return Instance( p_feature_data=feature_data )
    

## -------------------------------------------------------------------------------------------------
    def _fct_sin(self, p_x, p_outlier : bool):
        if p_outlier:
            return random.random() * 6 - 3

        return math.sin( p_x * math.pi / 180 )
    

## -------------------------------------------------------------------------------------------------
    def _fct_cos(self, p_x, p_outlier : bool):
        if p_outlier:
            return random.random() * 6 - 3

        return math.cos( p_x * math.pi / 180 )


## -------------------------------------------------------------------------------------------------
    def _fct_const(self, p_x, p_outlier : bool):
        if p_outlier:
            return random.random() * 6 - 2

        return 1.0
    
    
## -------------------------------------------------------------------------------------------------
    def _fct_lin(self, p_x, p_outlier : bool):
        if p_outlier:
            return p_x + random.random() * 20 - 10

        return p_x    