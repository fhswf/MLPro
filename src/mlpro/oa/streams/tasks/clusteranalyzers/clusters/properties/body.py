## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties
## -- Module  : body.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-12  0.1.0     DA       Creation
## -- 2025-03-19  0.1.1     DA       Refactoring (cprop_center_geo)
## -- 2025-06-06  0.2.0     DA       Refactoring: p_inst -> p_instance/s
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2025-06-06)

This module provides a template class for the cluster property 'body'.

"""


from mlpro.bf.streams import Instance
from mlpro.bf.math.properties import *
from mlpro.bf.math.geometry import cprop_size_geo, cprop_center_geo
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_deformation_index



# Export list for public API
__all__ = [ 'Body',
            'cprop_body' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Body (MultiProperty): 
    """
    ...

    Parameters
    ----------
    p_name : str
        Name of the property
    p_derivative_order_max : DerivativeOrderMax
        Maximum order of auto-generated derivatives (numeric properties only).
    p_value_prev : bool
        If True, the previous value is stored in value_prev whenever value is updated.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_kwargs : dict
        Keyword parameters.
    """

    C_PROPERTIES : PropertyDefinitions = [ cprop_center_geo,
                                           cprop_size_geo,
                                           cprop_deformation_index ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name : str,
                  p_derivative_order_max: int = 0, 
                  p_value_prev : ValuePrev = False,
                  p_properties : PropertyDefinitions = [],
                  p_visualize: bool = False,
                  **p_kwargs ):

        super().__init__( p_name = p_name, 
                          p_derivative_order_max = p_derivative_order_max,
                          p_value_prev = p_value_prev, 
                          p_properties = p_properties,
                          p_visualize = p_visualize,
                          **p_kwargs )
        
        self.color = None


## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_instance : Instance ) -> float:
        """
        Custom method to determine a scalar membership value for the given instance.

        Parameters
        ----------
        p_instance : Instance
            Instance.
        Returns
        -------
        float
            A scalar value in [0,1] that determines the given instance's membership in this cluster. 
            A value of 0 means that the given instance is not a member of the cluster at all while
            a value of 1 confirms full membership.
        """

        raise NotImplementedError
  




# Property definitions for cluster bodies
cprop_body  : PropertyDefinition = ( 'body', 0, False, Body )