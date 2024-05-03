## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.data
## -- Module  : properties.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-04-28  1.0.0     DA       Creation and first implemenation
## -- 2024-04-29  1.1.0     DA       - Method Properties.define_property: new return value
## --                                - Class Property: new property attribute dim
## -- 2024-04-30  1.1.1     DA       Method Properties.set_property() converts non-scalar types to
## --                                numpy arrays
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2024-04-30)

This module provides a systematics for numerical and textual properties to be added to a class. 

"""


# from mlpro.bf.various import *
from typing import Union
from datetime import datetime
import numpy as np




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Property:
    """
    This class stores details of a property.

    Parameters
    ----------
    p_derivative_order_max : int
        Maximum order of auto-generated derivatives (numeric properties only).

    Atttributes
    -----------
    value : Any
        Current value of the property.
    time_stamp : Union[datetime, float, int]
        Time stamp of the last value update.
    time_stamp_old : Union[datetime, float, int]
        Time stamp of the previous value update.
    derivative_order_max : int
        Maximum order of auto-generated derivatives (numeric properties only).
    derivatives : dict
        Current derivatives, stored by order (numeric properties only).
    derivatives_old : dict
        Previous derivatives, stored by order (numeric properties only).
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_derivative_order_max : int = 0):
        self.value                  = None
        self.time_stamp_old         = None
        self.time_stamp             = None
        self.derivative_order_max   = p_derivative_order_max
        self.derivatives_old        = {}
        self.derivatives            = {}


## -------------------------------------------------------------------------------------------------
    def _get_dim(self) -> int:
        """
        Internal method to determine the dimensionality of the currently stored values. It is used
        implicitely when accessing attribute 'dim'.
        """

        if self.value is None: return None
        try:
            # Try to treat the stored values as a numpy array
            return self.value.size
        except:
            try:
                # Try to treat the values as a python list
                return len(self.value)
            except:
                # Obviously a one-dimensional value
                return 1
            

## -------------------------------------------------------------------------------------------------
    dim = property( fget=_get_dim )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Properties:
    """
    Inherit from this class to add MLPro's property systematics to your own class. It enables
    defining properties of any type. Values can be set at a well-defined implicite or explicite 
    time point. Numeric (scalar or vectorial) properties can be derived automatically up to a pre-
    defined order.

    Parameters
    ----------
    p_property_cls = Property
        Property class to be used internally to manage properties. Default = Property.

    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, p_property_cls = Property ):

        self._property_cls  = p_property_cls
        self._properties    = {}
                  

## -------------------------------------------------------------------------------------------------
    def define_property(self, p_property : str, p_derivative_order_max : int = 0 ) -> Property:
        """
        Defines a new property by it's name. Optionally, auto-derivation can be added for numeric
        properties (scalar or vectorial).

        Parameters
        ----------
        p_property : str
            Name of the property.
        p_derivative_order_max : int
            Maximum order of auto-generated derivatives. Default = 0 (no derivative).

        Returns
        -------
        Property
            New property object.
        """

        prop_obj = self._property_cls( p_derivative_order_max = p_derivative_order_max )
        self._properties[p_property] = prop_obj

        return prop_obj


## -------------------------------------------------------------------------------------------------
    def get_property(self, p_property : str) -> Property:
        """
        Returns the property object stored under it's textual name.

        Parameters
        ----------
        p_property : str
            Name of the property.

        Returns
        -------
        Property
            Property object.
        """

        return self._properties[p_property]


## -------------------------------------------------------------------------------------------------
    def get_property_value(self, p_property : str):
        """
        Returns the current value of a property.

        Parameters
        ----------
        p_property : str
            Name of the property.

        Returns
        -------
        Any
            Current value of the property.
        """
        
        return self._properties[p_property].value
    

## -------------------------------------------------------------------------------------------------
    def get_properties(self):
        """
        Returns the dictionary of currently stored propery object.

        Returns
        -------
        dict
            Dictionary of property objects.
        """

        return self._properties


## -------------------------------------------------------------------------------------------------
    def set_property(self, p_property : str, p_value, p_time_stamp : Union[datetime, int, float] = None):
        """
        Sets the value of a property at a given time point.

        Parameters:
        -----------
        p_property : str
            Name of the property.
        p_value 
            Value of the property of any type (numeric, vectorial, textual, list, dict, ...).
            In case of auto-derivation only lists, numpy arrays and scalar numbers are supported. Lists
            are converted to numpy arrays.
        p_time_stamp : : Union[datetime, int, float]
            Optional time stamp of type datetime, int or float. If not provided, an internal continuous
            integer time stamp is generated.

        """
    
        # 0 Get stored property object
        prop                = self._properties[p_property]
        prop.time_stamp     = p_time_stamp
        prop.value          = p_value


        # 1 Preparation of time stamp
        if p_time_stamp is None:
            try:
                prop.time_stamp = prop.time_stamp_old + 1
            except:
                prop.time_stamp = 0


        # 2 Numeric derivation
        if prop.derivative_order_max > 0:

            # 2.1 Computation of time delta
            if prop.time_stamp_old is not None:

                delta_t = prop.time_stamp - prop.time_stamp_old
            
                try: 
                    delta_t = delta_t.total_seconds()
                except:
                    pass

            
            # 2.2 Derivation
            prop.derivatives_old = prop.derivatives
            prop.derivatives     = {}
            if np.isscalar(p_value):
                prop.derivatives[0]  = p_value
            else:
                prop.value           = np.array( p_value )
                prop.derivatives[0]  = prop.value

            for order in range(prop.derivative_order_max):
                try:
                    prop.derivatives[order+1] = ( prop.derivatives[order] - prop.derivatives_old[order] ) / delta_t
                except:
                    break


        # 3 Outro
        prop.time_stamp_old  = prop.time_stamp