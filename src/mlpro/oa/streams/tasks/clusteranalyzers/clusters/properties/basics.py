## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-05  0.1.0     DA       Creation
## -- 2024-05-07  0.2.0     DA       Extensions
## -- 2024-05-24  0.3.0     SK       Addition of further properties
## -- 2024-05-29  0.4.0     DA       Moved Centroid-based properties to centroid.py
## -- 2024-05-30  0.5.0     DA       Global aliases: new boolean param ValuePrev
## -- 2024-06-03  0.6.0     DA       Moved all geometric properties to mlpro.bf.math.geometry.basics
## -- 2025-03-11  0.7.0     DA       New properties cprop_size*_prev
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.7.0 (2025-03-11)

This module provides typical cluster properties to be reused in own cluster analyzers.
"""


from mlpro.bf.math.properties import PropertyDefinition, Property


# Export list for public API
__all__ = [ 'cprop_size',
            'cprop_size1',
            'cprop_size2',
            'cprop_size_prev',
            'cprop_size1_prev',
            'cprop_size2_prev',
            'cprop_age',
            'cprop_age1',
            'cprop_age2',
            'cprop_density',
            'cprop_density1',
            'cprop_density2',
            'cprop_compactness',
            'cprop_compactness1',
            'cprop_compactness2' ]


#
# Typical cluster properties to be reused in your own cluster analyzers
#

# Size (=number of associated instances) with 0,1,2 order derivatives 
cprop_size         : PropertyDefinition = ( 'size', 0, False, Property )
cprop_size1        : PropertyDefinition = ( 'size', 1, False, Property )
cprop_size2        : PropertyDefinition = ( 'size', 2, False, Property )
cprop_size_prev    : PropertyDefinition = ( 'size', 0, True, Property )
cprop_size1_prev   : PropertyDefinition = ( 'size', 1, True, Property )
cprop_size2_prev   : PropertyDefinition = ( 'size', 2, True, Property )

# Age with 0,1,2 order derivatives
cprop_age          : PropertyDefinition = ( 'age', 0, False, Property )
cprop_age1         : PropertyDefinition = ( 'age', 1, False, Property )
cprop_age2         : PropertyDefinition = ( 'age', 2, False, Property )

# Density with 0,1,2 order derivatives
cprop_density      : PropertyDefinition = ( 'density', 0, False, Property )
cprop_density1     : PropertyDefinition = ( 'density', 1, False, Property )
cprop_density2     : PropertyDefinition = ( 'density', 2, False, Property )

# Compactness with 0,1,2 order derivatives
cprop_compactness  : PropertyDefinition = ( 'compactness', 0, False, Property )
cprop_compactness1 : PropertyDefinition = ( 'compactness', 1, False, Property )
cprop_compactness2 : PropertyDefinition = ( 'compactness', 2, False, Property )