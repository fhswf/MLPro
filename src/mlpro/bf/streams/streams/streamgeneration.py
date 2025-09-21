## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams
## -- Module  : streamgeneration.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-09-21  1.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-09-21)

This module provides ...

"""

#from typing import Union, Literal, List
#from dataclasses import dataclass
#import random

import numpy as np

from mlpro.bf import Log, Mode
from mlpro.bf.exceptions import ParamError
from mlpro.bf.math import Element, MSpace, ESpace
from mlpro.bf.streams import Feature, Instance, Stream, Sampler



# Export list for public API
__all__ = [ 'StreamGenerator' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamGenerator(Stream):
    """
    
    """

    C_TYPE              = 'Stream Generator'
    C_BOUNDARIES        = [-1000, 1000] # Boundaries of the feature space
  
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int,
                  p_id = None,
                  p_name : str = '',
                  p_num_instances : int = 0,
                  p_version : str = '',
                  p_sampler : Sampler = None,
                  p_boundaries_rescale : list = None,
                  p_outlier_rate : float = 0.0,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        self._num_dim = p_num_dim
        if self._num_dim < 1:
            raise ParamError("Number of dimensions must be at least 1.")
        

        # Optional explicit boundaries per dimension and resulting rescaling parameters
        if p_boundaries_rescale is not None:
            if len(p_boundaries_rescale) != self._num_dim:
                raise ParamError(f"Expected {self._num_dim} dimensions for feature boundaries.")
            self._boundaries_rescale    = p_boundaries_rescale
        else:
            self._boundaries_rescale    = [self.C_BOUNDARIES]*self._num_dim

        self._rescaling_params      = self._get_rescaling_params(self._boundaries_rescale)


        # Outlier generation
        self._outlier_appearance = p_outlier_rate > 0.0
        self._outlier_rate       = p_outlier_rate


        super().__init__( p_id = p_id,
                          p_name  = p_name,
                          p_num_instances = p_num_instances,
                          p_version = p_version,
                          p_feature_space = self.get_feature_space(),
                          p_label_space = self.get_label_space(),
                          p_sampler = p_sampler,
                          p_mode = Mode.C_MODE_SIM,
                          p_logging = p_logging,
                          **p_kwargs )        


## -------------------------------------------------------------------------------------------------
    def _get_rescaling_params(self, p_boundaries_rescale : list):
        if p_boundaries_rescale is None: return None

        if len(p_boundaries_rescale) != self._num_dim:
            raise ParamError(f"Expected {self._num_dim} dimensions for rescale boundaries.")

        params = np.zeros((self._num_dim, 2))

        for dim in range(self._num_dim):
            params[dim,0] = ( p_boundaries_rescale[dim][1] - p_boundaries_rescale[dim][0] ) / ( self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0] )
            params[dim,1] = p_boundaries_rescale[dim][0] - self.C_BOUNDARIES[0] * params[dim,0]

        return params

        
## -------------------------------------------------------------------------------------------------
    def __next__(self) -> Instance:

        if self._outlier_appearance and np.random.rand() < self._outlier_rate:
            # ...
            pass

        else:
            new_inst = super().__next__()

        if self._rescaling_params is not None:
            # ...
            pass


        return new_inst