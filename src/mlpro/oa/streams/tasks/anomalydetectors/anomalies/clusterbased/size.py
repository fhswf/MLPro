## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : density.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-15  0.0.0     SK       Creation
## -- 2024-06-15  1.0.0     SK       Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-06-15)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import Instance
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.basics import CBAnomaly





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterSizeVariation (CBAnomaly):
    """
    Event class to be raised when the density of a cluster changes.
    
    """

    C_NAME      = 'Cluster size variation'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)

