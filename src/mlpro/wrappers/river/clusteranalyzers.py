## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers.river
## -- Module  : clusteranalyzers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-12  0.0.0     DA       Creation
## -- 2023-05-xx  1.0.0     SY       First version release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-05-xx)

This module provides wrapper classes from River to MLPro, specifically for cluster analyzers. This
module includes three clustering algorithms from River that are embedded to MLPro, such as:

1) DBSTREAM (https://riverml.xyz/latest/api/cluster/DBSTREAM/)

2) CluStream (https://riverml.xyz/latest/api/cluster/CluStream/)

3) DenStream (https://riverml.xyz/latest/api/cluster/DenStream/)

Learn more:
https://www.riverml.xyz/

"""


from mlpro.wrappers.river.basics import WrapperRiver
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.bf.mt import Task as MLTask
from mlpro.bf.various import Log
from mlpro.bf.streams import *
from river import base, cluster
from typing import List, Tuple





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrClusterAnalyzerRiver2MLPro (WrapperRiver, ClusterAnalyzer):

    C_TYPE              = 'River Cluster Analyzer'
    C_NAME              = '????'
    
    C_WRAPPED_PACKAGE   = 'river'
    C_MINIMUM_VERSION   = '0.15.0'
    
    C_CLS_CLUSTER       = Cluster


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_river_algo:base.Clusterer,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 **p_kwargs):
        
        self._river_algo = p_river_algo

        WrapperRiver.__init__(self, p_logging=p_logging)

        ClusterAnalyzer.__init__(self,
                                 p_name=p_name,
                                 p_range_max=p_range_max,
                                 p_ada=p_ada,
                                 p_visualize=p_visualize,
                                 **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: List[Instance], p_inst_del: List[Instance]):
        # p_inst_del has no use

        # transform new instance to a dictionary of features
        x = {0: 1, 1: 1} # example

        self._river_algo.learn_one(x)


## -------------------------------------------------------------------------------------------------
    def get_clusters(self):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_cluster_membership(self, p_inst:Instance) -> List[Tuple[str, float, Cluster]]:
        # to be added
        pass
        


    