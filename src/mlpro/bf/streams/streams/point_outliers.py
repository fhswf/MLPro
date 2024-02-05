## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : point_outliers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-02-05  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-02-05)

This module provides ...

"""


import random
import math
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProPOutliers (StreamMLProBase):
    """
    This demo stream provides self.C_NUM_INSTANCES n-dimensional instances randomly positioned around
    centers which may or may not move over time.

    p_num_dim : int
        The number of dimensions or features of the data. Default = 3.
    p_num_instances : int
        Total number of instances. The value '0' means indefinite. Default = 1000.
    p_functions : list[str]
        List of mathematical functions per feature. 
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'CloudsNDim'
    C_NAME                  = 'Clouds N-Dim'
    C_TYPE                  = 'Benchmark'
    C_VERSION               = '1.0.0'
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES C_NUM_DIMENSIONS-dimensional instances per cluster randomly positioned around centers which may or maynot move over time.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int = 5,
                  p_num_instances : int = 1000,
                  p_functions : list[str] = ['sin', 'cos', 'tan', 'const', 'lin'],
                  p_seed = None,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        self._num_dim         = len(p_functions)
        self.C_NUM_INSTANCES  = p_num_instances
        self._functions       = p_functions
        self._fct_methods     = []

        for fct in p_functions:
            self._fct_methods.append( getattr(self, 'self._fct_' + fct) )

        self.set_random_seed(p_seed=p_seed)

        StreamMLProBase.__init__ (self,
                                  p_logging=p_logging,
                                  **p_kwargs)
        

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = MSpace()

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
            values.append( fct_method(self._index) )   

        feature_data.set_values(values)

        self._index += 1

        return Instance( p_feature_data=feature_data )
    

## -------------------------------------------------------------------------------------------------
    def _fct_sin(self, p_x):
        return math.sin( p_x * math.pi / 180 )
    

## -------------------------------------------------------------------------------------------------
    def _fct_cos(self, p_x):
        return math.cos( p_x * math.pi / 180 )


## -------------------------------------------------------------------------------------------------
    def _fct_tan(self, p_x):
        return math.tan( p_x * math.pi / 180 )


## -------------------------------------------------------------------------------------------------
    def _fct_const(self, p_x):
        return 1.0
    
    
## -------------------------------------------------------------------------------------------------
    def _fct_lin(self, p_x):
        return p_x    