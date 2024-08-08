## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : enlargement.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-28  1.3.0     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.9 (2024-05-28)

This module provides a template class for cluster enlargement to be used in anomaly detection algorithms.
"""

from mlpro.oa.streams.basics import Instance
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.basics import CBAnomaly





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterEnlargement (CBAnomaly):
    """
    Event class to be raised when a cluster enlarges.
    
    Parameters
    ----------
    p_id : int
        Anomaly ID. Default value = 0.
    p_instances : Instance
        List of instances. Default value = None.
    p_clusters : dict[Cluster]
        Clusters associated with the anomaly. Default = None.
    p_properties : dict
        Poperties of clusters associated with the anomaly. Default = None.
    p_ano_scores : list
        List of anomaly scores of instances. Default = None.
    p_det_time : str
        Time of occurance of anomaly. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_NAME      = 'Cluster Enlargement'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id : int = 0,
                 p_instances : list[Instance] = None,
                 p_clusters : dict[Cluster] = None,
                 p_properties : dict = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_id=p_id,
                         p_instances=p_instances,
                         p_clusters=p_clusters,
                         p_properties=p_properties,
                         p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize,
                         p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)

