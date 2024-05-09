## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers
## -- Module  : sparccstream.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-09  0.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-05-09)

This module provides ...

"""


from typing import List
from mlpro.bf.math.normalizers.basics import Normalizer
from mlpro.bf.streams import Instance, List
from mlpro.bf.various import Log
from mlpro.oa.streams import OATask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SPARCCStream (ClusterAnalyzer): 
    """
    """
    
## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_cls_cluster: type = ..., 
                  p_cluster_limit: int = 0, 
                  p_name: str = None, 
                  p_range_max=OATask.C_RANGE_THREAD, 
                  p_ada: bool = True, 
                  p_duplicate_data: bool = False, 
                  p_visualize: bool = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        super().__init__( p_cls_cluster = p_cls_cluster, 
                          p_cluster_limit = p_cluster_limit, 
                          p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_ada = p_ada, 
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new: List[Instance]) -> bool:
        return super()._adapt(p_inst_new)
    

## -------------------------------------------------------------------------------------------------
    def _adapt_reverse(self, p_inst_del: List[Instance]) -> bool:
        return super()._adapt_reverse(p_inst_del)
    

## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer: Normalizer):
        return super()._renormalize(p_normalizer)
    
