## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-05  0.1.0     DA       Creation
## -- 2024-05-07  0.2.0     DA       Extensions
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-05-07)

This module provides typical cluster properties to be reused in own cluster analyzers.
"""


from mlpro.bf.math.properties import PropertyDefinition, Property
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.centroid import Centroid


#
# Typical cluster properties to be reused in your own cluster analyzers
#

# Size with 0,1,2 order derivatives 
cprop_size      : PropertyDefinition = ( 'size', 0, Property )
cprop_size1     : PropertyDefinition = ( 'size', 1, Property )
cprop_size2     : PropertyDefinition = ( 'size', 2, Property )

# Age with 0,1,2 order derivatives
cprop_age       : PropertyDefinition = ( 'age', 0, Property )
cprop_age1      : PropertyDefinition = ( 'age', 1, Property )
cprop_age2      : PropertyDefinition = ( 'age', 2, Property )

# Centroid with 0,1,2 order derivatives and plot functionality
cprop_centroid  : PropertyDefinition = ( 'centroid', 0, Centroid )
cprop_centroid1 : PropertyDefinition = ( 'centroid', 1, Centroid )
cprop_centroid2 : PropertyDefinition = ( 'centroid', 2, Centroid )

