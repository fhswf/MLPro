## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.geometry
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-03  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-06-03)

This module provides basic definitions for geometric shapes.

""" 

from mlpro.bf.math.properties import *



# Export list for public API
__all__ = [ 'cprop_size_geo',
            'cprop_size_geo1',
            'cprop_size_geo2' ]



# Geometric size with 0,1,2 order derivatives
cprop_size_geo     : PropertyDefinition = ( 'size_geo', 0, False, Property )
cprop_size_geo1    : PropertyDefinition = ( 'size_geo', 1, False, Property )
cprop_size_geo2    : PropertyDefinition = ( 'size_geo', 2, False, Property )