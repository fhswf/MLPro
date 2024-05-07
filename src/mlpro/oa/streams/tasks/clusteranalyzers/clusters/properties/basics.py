## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-05  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-05-05)

This module provides typical cluster properties to be reused in own cluster analyzers.
"""


from mlpro.bf.math.properties import PropertyDefinition, Property
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.centroid import Centroid



# Typical cluster properties to be reused in your own cluster analyzers
cprop_size      : PropertyDefinition = ( 'size', 0, Property )
cprop_age       : PropertyDefinition = ( 'age', 0, Property )
cprop_centroid  : PropertyDefinition = ( 'centroid', 2, Centroid )

