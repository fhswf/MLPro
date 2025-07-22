## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams
## -- Module  : doublespiral2d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-14  0.0.0     DA       Creation 
## -- 2022-12-14  1.0.0     DA       First implementation
## -- 2024-06-04  1.0.1     DA       Bugfix: ESpace instead of MSpace
## -- 2025-04-02  1.0.2     DA       Little refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2025-04-02)

This module provides the native stream class DoubleSpiral2D. It provides 721 instances with 
2-dimensional feature data that follow a double spiral pattern.
"""

import numpy as np
from math import sin, cos, pi

from mlpro.bf.various import Log
from mlpro.bf.math import MSpace, ESpace
from mlpro.bf.streams.basics import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



# Export list for public API
__all__ = [ 'DoubleSpiral2D' ]




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
        feature_space : MSpace = ESpace()

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