## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : drift.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-22  1.2.1     SK       Refactoring
## -- 2024-05-28  1.2.2     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2024-05-28)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import Instance
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.basics import CBAnomaly





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDrift (CBAnomaly):
    """
    Event class to be raised when cluster drift detected.
    
    """

    C_NAME      = 'Cluster drift Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : list[Instance] = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)

