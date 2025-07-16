## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters
## -- Module  : centroid.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-24  0.0.0     DA       Creation
## -- 2023-04-18  0.1.0     DA       First implementation of classes ClusterMembership, ClusterAnalyzer
## -- 2023-05-06  0.2.0     DA       New class ClusterCentroid
## -- 2023-05-14  0.3.0     DA       Class ClusterAnalyzer: simplification
## -- 2023-05-30  0.3.1     DA       Further comments, docstrings
## -- 2023-06-03  0.4.0     DA       Method ClusterAnalyzer.get_cluster_memberships():
## --                                - renaming
## --                                - new parameter p_scope
## --                                - refactoring
## --                                New Method ClusterAnalyzer.new_cluster_allowed()
## -- 2023-11-18  0.5.0     DA       Class ClusterCentroid: added plot functionality
## -- 2023-12-08  0.6.0     DA       Class ClusterAnalyzer: 
## --                                - changed internal cluster storage from list to dictionary
## --                                - added method _remove_cluster()
## -- 2023-12-10  0.6.1     DA       Bugfix in method ClusterAnalyzer.get_cluster_membership()
## -- 2023-12-20  0.7.0     DA       Renormalization
## -- 2024-02-23  0.8.0     DA       Class ClusterCentroid: implementation of methods _remove_plot*
## -- 2024-02-24  0.8.1     DA       Method ClusterAnalyzer._remove_cluster() explicitely removes
## --                                the plot of a cluster before removal of the cluster itself
## -- 2024-02-24  0.8.2     DA       Class ClusterCentroid: redefined method remove_plot()
## -- 2024-04-10  0.8.3     DA       Refactoring
## -- 2024-04-29  0.9.0     DA       Refactoring after changes on class Point
## -- 2024-05-06  1.0.0     DA       Refactoring
## -- 2024-05-07  1.0.1     DA       Bugfix in ClusterCentroid.__init__(): internal properties first
## -- 2024-05-27  1.1.0     DA       Changes on property management
## -- 2024-05-29  1.1.1     DA       Method ClusterCentroid.renormalize() removed
## -- 2024-06-08  1.2.0     DA       Refactoring:
## --                                - removed implementation for get_membership() since this 
## --                                  depends on the shape of a cluster body
## --                                - implemented new method get_influence()
## -- 2024-06-18  1.3.0     DA       Removed method ClusterCentroid.__init__()
## -- 2025-04-13  1.4.0     DA       Introduction of C_EPSILON
## -- 2025-06-06  1.5.0     DA       Refactoring: p_inst -> p_instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.0 (2025-06-06)

This module provides a template class for clusters with a centroid.
"""

from mlpro.bf.streams import Instance
from mlpro.bf.math import Element
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_centroid



# Export list for public API
__all__ = [ 'ClusterCentroid' ]





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterCentroid (Cluster):
    """
    Extended cluster class with a centroid. 

    Attributes
    ----------
    centroid : Centroid
        Centroid object.
    """

    C_PROPERTIES    = [ cprop_centroid ]
    C_EPSILON       = 0.0001

# ## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id):
        super().set_id( p_id = p_id )
        self.centroid.id = p_id


## -------------------------------------------------------------------------------------------------
    def get_influence(self, p_instance: Instance) -> float:
        """
        Default strategy to determine the influence of a cluster on a specified instance based
        on the metric distance between the instance and the cluster centroid.
        """

        feature_data = p_instance.get_feature_data()

        try:
            centroid_elem = self._centroid_elem
        except:
            self._centroid_elem = Element( p_set=feature_data.get_related_set() )
            centroid_elem = self._centroid_elem

        centroid_elem.set_values( p_values=self.centroid.value )

        return 1 / ( feature_data.get_related_set().distance( p_e1 = feature_data, p_e2 = centroid_elem ) + self.C_EPSILON )
