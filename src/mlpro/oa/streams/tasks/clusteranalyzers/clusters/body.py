## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters
## -- Module  : body.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-13  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-03-13)

This module provides a template class for clusters with a centroid and a body.

"""


from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.bf.math.geometry import cprop_size_geo
from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import ClusterId
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import ClusterCentroid, cprop_centroid
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_center_geo, cprop_body, cprop_deformation_index



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterBody (ClusterCentroid):
    """
    Template class for clusters with a centroid and a body.

    """

    C_PROPERTIES        = [ cprop_centroid,
                            cprop_size,
                            cprop_body,
                            cprop_size_geo,
                            cprop_center_geo,
                            cprop_deformation_index ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id : ClusterId,
                  p_properties : PropertyDefinitions = [],
                  p_visualize : bool = False,
                  **p_kwargs ): 

        super().__init__( p_id = p_id, 
                          p_properties = p_properties, 
                          p_visualize = p_visualize, 
                          **p_kwargs )
        
        self._link_property( p_attr = cprop_size_geo[0], p_prop = self.body.size_geo )
        self._link_property( p_attr = cprop_center_geo[0], p_prop = self.body.center_geo )
        self._link_property( p_attr = cprop_deformation_index[0], p_prop = self.body.deformation_index )


## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_inst : Instance ) -> float:
        return self.body.get_membership( p_inst = p_inst )