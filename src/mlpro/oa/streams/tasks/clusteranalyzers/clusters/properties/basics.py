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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2024-05-24)

This module provides typical cluster properties to be reused in own cluster analyzers.
"""


from mlpro.bf.math.properties import PropertyDefinition, Property
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.centroid import Centroid


#
# Typical cluster properties to be reused in your own cluster analyzers
#

# Size (=number of associated instances) with 0,1,2 order derivatives 
cprop_size         : PropertyDefinition = ( 'size', 0, Property )
cprop_size1        : PropertyDefinition = ( 'size', 1, Property )
cprop_size2        : PropertyDefinition = ( 'size', 2, Property )

# Age with 0,1,2 order derivatives
cprop_age          : PropertyDefinition = ( 'age', 0, Property )
cprop_age1         : PropertyDefinition = ( 'age', 1, Property )
cprop_age2         : PropertyDefinition = ( 'age', 2, Property )

# Centroid with 0,1,2 order derivatives and plot functionality
cprop_centroid     : PropertyDefinition = ( 'centroid', 0, Centroid )
cprop_centroid1    : PropertyDefinition = ( 'centroid', 1, Centroid )
cprop_centroid2    : PropertyDefinition = ( 'centroid', 2, Centroid )

# Geometric center with 0,1,2 order derivatives and plot functionality
cprop_center_geo   : PropertyDefinition = ( 'center_geo', 0, Centroid )
cprop_center_geo1  : PropertyDefinition = ( 'center_geo', 1, Centroid )
cprop_center_geo2  : PropertyDefinition = ( 'center_geo', 2, Centroid )

# Density with 0,1,2 order derivatives
cprop_density      : PropertyDefinition = ( 'density', 0, Property )
cprop_density1     : PropertyDefinition = ( 'density', 1, Property )
cprop_density2     : PropertyDefinition = ( 'density', 2, Property )

# Geometric size with 0,1,2 order derivatives
cprop_size_geo     : PropertyDefinition = ( 'size_geo', 0, Property )
cprop_size_geo1    : PropertyDefinition = ( 'size_geo', 1, Property )
cprop_size_geo2    : PropertyDefinition = ( 'size_geo', 2, Property )

# Compactness with 0,1,2 order derivatives
cprop_compactness  : PropertyDefinition = ( 'compactness', 0, Property )
cprop_compactness1 : PropertyDefinition = ( 'compactness', 1, Property )
cprop_compactness2 : PropertyDefinition = ( 'compactness', 2, Property )

