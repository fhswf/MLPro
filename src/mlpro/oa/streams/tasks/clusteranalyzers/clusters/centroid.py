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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2024-05-07)

This module provides templates for cluster analysis to be used in the context of online adaptivity.
"""

from typing import List, Tuple

from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.bf.math.properties import *
from mlpro.bf.math.normalizers import Normalizer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_centroid




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterCentroid (Cluster):
    """
    Extended cluster class with a centroid.

    Parameters
    ----------
    p_id
        Optional external id
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_cls_centroid = Point
        Name of a point class. Default = Point
    **p_kwargs
        Further optional keyword arguments.

    Attributes
    ----------
    centroid : Centroid
        Centroid object.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id = None,
                  p_properties : PropertyDefinitions = [],
                  p_visualize : bool = False ):

        super().__init__( p_id=p_id, p_visualize=p_visualize )

        self.add_properties( p_property_definitions = [ cprop_centroid ], p_visualize = p_visualize )
        self.add_properties( p_property_definitions = p_properties, p_visualize = p_visualize )

        self.centroid.set_id( p_id = self.get_id() )
        self._centroid_elem : Element = None


# ## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id=None):
        super().set_id( p_id = p_id )
        try:
            self.centroid.set_id( p_id = p_id )
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_inst: Instance) -> float:
        feature_data = p_inst.get_feature_data()

        if self._centroid_elem is None:
            self._centroid_elem = Element( p_set=feature_data.get_related_set() )

        self._centroid_elem.set_values( p_value=self.centroid.value )

        return feature_data.get_related_set().distance( p_e1 = feature_data, p_e2 = self._centroid_elem )


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer: Normalizer):
        self.centroid.renormalize( p_normalizer=p_normalizer)

