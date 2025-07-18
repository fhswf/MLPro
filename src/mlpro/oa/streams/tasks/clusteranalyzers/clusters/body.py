## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters
## -- Module  : body.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-13  0.1.0     DA       Creation
## -- 2025-03-18  0.1.1     DA       Bugfix in ClusterBody.__init__()
## -- 2025-03-19  0.1.2     DA       Refactoring (cprop_center_geo)
## -- 2025-06-06  0.2.0     DA       Refactoring: p_inst -> p_instances
## -- 2025-06-11  0.3.0     DA       - Redefintion of method ClusterBody.update_properties()
## --                                - New method ClusterBody._update_density()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-06-11)

This module provides a template class for clusters with a centroid and a body.

"""


from mlpro.bf.various import TStampType
from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.bf.math.geometry import cprop_size_geo, cprop_center_geo
from mlpro.bf.streams import Instance

from mlpro.oa.streams.tasks.clusteranalyzers.clusters import ClusterId
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import ClusterCentroid 
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import *


# Export list for public API
__all__ = [ 'ClusterBody' ]




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
                            cprop_deformation_index,
                            cprop_density ]

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
        
        self._link_property( p_attr = cprop_size_geo[0], p_prop = self.body )
        self._link_property( p_attr = cprop_center_geo[0], p_prop = self.body )
        self._link_property( p_attr = cprop_deformation_index[0], p_prop = self.body )


## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_instance : Instance ) -> float:
        return self.body.get_membership( p_instance = p_instance )
    

## -------------------------------------------------------------------------------------------------
    def _update_density(self, p_tstamp : TStampType):
        
        try:
            density = self.size.value / self.size_geo.value
            self.density.set( p_value = density, p_time_stamp = p_tstamp )
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def update_properties(self, p_tstamp : TStampType):
        super().update_properties( p_tstamp = p_tstamp )
        self._update_density( p_tstamp = p_tstamp )