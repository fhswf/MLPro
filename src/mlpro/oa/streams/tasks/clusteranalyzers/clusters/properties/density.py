## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties
## -- Module  : density.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-10  0.1.0     DA/DS    Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-06-10)

This module provides a template class for the property 'density' of cluster bodies.

"""


from mlpro.bf.math.properties import MultiProperty, PropertyDefinition




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Density (MultiProperty): 
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
cprop_density  : PropertyDefinition = ( 'density', 0, False, Density )
cprop_density1 : PropertyDefinition = ( 'density', 1, False, Density )
cprop_density2 : PropertyDefinition = ( 'density', 2, False, Density )