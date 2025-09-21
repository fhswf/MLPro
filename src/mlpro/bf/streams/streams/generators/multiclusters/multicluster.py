## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams.multiclusters
## -- Module  : multicluster.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-09-21  1.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-09-21)

This module provides a multi-stream class specialized for cluster data generation in a d-dimensional
feature space. It allows the combination of multiple cluster streams with individual parameters such
as number of dimensions, number of instances, cluster states (center and radii), batch size, and start
instance. The multi-stream generator can be used for simulating complex data streams with multiple
clusters, each with its own characteristics.

"""

from mlpro.bf.streams.streams.generators.basics import MultiStreamGenerator



# Export list for public API
__all__ = [ 'MultiStreamGenCluster' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiStreamGenCluster (MultiStreamGenerator):
    """
    This is a template class for native MLPro streams. It provides the basic functionality for all
    native MLPro streams.
    """

    C_TYPE              = 'Multi-stream Cluster'
