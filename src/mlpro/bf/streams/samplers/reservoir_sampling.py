## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.samplers
## -- Module  : reservoir_sampling.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-16  0.0.0     SY       Creation 
## -- 2023-04-16  1.0.0     SY       First version release
## -- 2025-06-06  1.1.0     DA       Refactoring: p_inst -> p_instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-06-06)

This module provides a ready-to-use stream sampler class SamplerReservoir.
Reservoir sampling is a simple algorithm that is still part of a random sampling algorithm.
This algorithm is proposed by Jeffrey Vitter. In this module, we apply the default algorithm of
reservoir sampling with Algorithm R. However, we enhance the algorithm, where the p_num_instances
remains unknown.

"""

from mlpro.bf.streams.basics import Sampler, Instance
from mlpro.bf.exceptions import ParamError
import random



# Export list for public API
__all__ = [ 'SamplerReservoir' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SamplerReservoir(Sampler):
    """
    A ready-to-use class for data streams with reservoir sampler using algorithm R.
    This object can be used in Stream.

    Parameters
    ----------
    p_num_instances : int
        Number of instances. This parameter is optional. Default = None.
    p_reservoir_size : int
        Size of an reservoir. Default = 10.
    p_seed : int
        Random seeding. Default = 0.
    """

    C_SCIREF_TYPE_ARTICLE   = "Journal Article"
    C_SCIREF_TYPE           = C_SCIREF_TYPE_ARTICLE
    C_SCIREF_AUTHOR         = 'Jeffrey S. Vitter'
    C_SCIREF_TITLE          = 'Random Sampling with a Reservoir'
    C_SCIREF_YEAR           = '1985'
    C_SCIREF_PUBLISHER      = 'Association for Computing Machinery'
    C_SCIREF_VOLUME         = '11'
    C_SCIREF_NUMBER         = '1'
    C_SCIREF_URL            = 'https://doi.org/10.1145/3147.3165'
    C_SCIREF_DOI            = '10.1145/3147.3165'
    C_SCIREF_JOURNAL        = 'ACM Trans. Math. Softw.'
    C_SCIREF_MONTH          = 'Mar'
    C_SCIREF_PAGES          = '37-57'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_num_instances:int=None, p_reservoir_size:int=10, p_seed:int=0):
        
        super().__init__(p_num_instances=p_num_instances, p_reservoir_size=p_reservoir_size, p_seed=p_seed)
        
        try:
            self._res_size = self._kwargs['p_reservoir_size']
        except:
            raise ParamError('Parameter p_reservoir_size is missing.')
        
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        A method to reset the sampler's settings.
        """
        
        self._idx       = 0
        self._res       = []
        self._empty_res = True
        random.seed(self._kwargs['p_seed'])
        

## -------------------------------------------------------------------------------------------------
    def _omit_instance(self, p_instance:Instance) -> bool:
        """
        A custom method to filter any incoming instances, which is being called by omit_instance()
        method.

        Parameters
        ----------
        p_instance : Instance
            An input instance to be filtered.

        Returns
        -------
        bool
            False means the input instance is not omitted, otherwise True.

        """

        if self._num_instances is not None:
            if self._empty_res:
                for i in range(self._res_size):
                    self._res.append(i)
                
                while i < self._num_instances:
                    j = random.randrange(i+1)
                    if j < self._res_size:
                        self._res[j] = i
                    i += 1
                
                self._empty_res = False
            
            if self._idx in self._res:
                self._idx += 1
                return False
            else:
                self._idx += 1
                return True

        else:
            if self._idx <= self._res_size:
                self._idx += 1
                return False
            else:
                self._idx += 1
                j = random.randrange(self._idx)
                if j < self._res_size:
                    return False
                else:
                    return True
        
            