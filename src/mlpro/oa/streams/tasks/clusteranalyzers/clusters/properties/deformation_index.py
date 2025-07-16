## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties
## -- Module  : deformation_index.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-13  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-03-13)

This module provides a template class for the property 'deformation_index' of cluster bodies.

"""


from mlpro.bf.math.properties import MultiProperty, PropertyDefinition



# Export list for public API
__all__ = [ 'DeformationIndex',
            'cprop_deformation_index',
            'cprop_deformation_index1',
            'cprop_deformation_index2' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DeformationIndex (MultiProperty): 
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

    pass

  



# Property definitions for deformation indices of cluster bodies
cprop_deformation_index  : PropertyDefinition = ( 'deformation_index', 0, False, DeformationIndex )
cprop_deformation_index1  : PropertyDefinition = ( 'deformation_index', 1, False, DeformationIndex )
cprop_deformation_index2  : PropertyDefinition = ( 'deformation_index', 2, False, DeformationIndex )