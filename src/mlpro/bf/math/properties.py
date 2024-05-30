## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math
## -- Module  : properties.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-04-28  0.1.0     DA       Creation and first implemenation
## -- 2024-04-29  0.2.0     DA       - Method Properties.define_property: new return value
## --                                - Class Property: new property attribute dim
## -- 2024-04-30  0.3.0     DA       Method Properties.set_property() converts non-scalar types to
## --                                numpy arrays
## -- 2024-05-04  0.4.0     DA       Introduction of type aliases
## -- 2024-05-05  0.5.0     DA       Redesign: alignment with Python's managed attributes
## -- 2024-05-06  0.6.0     DA       Completion of plot functionality
## -- 2024-05-27  0.7.0     DA       Class Properties: 
## --                                - new parent classes Plottable, Renormalizable
## --                                - implementation of *_plot() and renormalize
## --                                - constructor: new parameters p_properties, p_visualization
## -- 2024-05-29  0.8.0     DA       Class Property: 
## --                                - standalone plot turned off
## --                                - new parameter p_name
## -- 2024-05-30  0.9.0     DA       Class Property:
## --                                - new attribute value_prev
## --                                - new parameter p_value_prev
## --                                Class Properties:
## --                                - method add_property(): new parameter p_value_prev
## --                                Global aliases:
## --                                - new alias ValuePrev
## --                                - extension of PropertyDefinition by ValuePrev
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.9.0 (2024-05-30)

This module provides a systematics for enriched managed properties. MLPro's enriched properties
store any data like class attributes and they can be used like class attributes. They extend the
basic functionalities of a classic attibute by the following features:

- Optional numeric auto-derivation up to a well-defined maximum order
- Plottable (see class mlpro.bf.plot.Plottable)
- Renormalizable (see class mlpro.bf.math.Renormalizable)

Hint: plot and renormalization functionality is to be implemented in child classes.
"""


from typing import List, Union, Tuple
from datetime import datetime
import numpy as np
from matplotlib.figure import Figure

from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.math.normalizers import Normalizer, Renormalizable



# Type aliases for property definitions
PropertyName        = str
DerivativeOrderMax  = int
ValuePrev           = bool
PropertyClass       = type

PropertyDefinition  = Tuple[ PropertyName, DerivativeOrderMax, ValuePrev, PropertyClass ]
PropertyDefinitions = List[ PropertyDefinition ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Property (Plottable, Renormalizable):
    """
    This class implements an enriched managed property. It enables storing a value of any type. In
    case of numeric data (one- or multi-dimensional), an auto-derivation up to a well-defined maximum
    order can be turned on. Whenever the value is updated, all derivatives are automatically updated
    as well.

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

    Atttributes
    -----------
    value : Any
        Current value of the property.
    value_prev : Any
        Previous value of the property (readonly).
    dim : int
        Dimensionality of the stored value. In case of strings the length is returned.
    time_stamp : Union[datetime, float, int]
        Time stamp of the last value update.
    derivative_order_max : DerivativeOrderMax
        Maximum order of auto-generated derivatives (numeric properties only). 
    derivatives : dict
        Current derivatives, stored by order (numeric properties only).
    """

    C_PLOT_STANDALONE               = False

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name : str, 
                  p_derivative_order_max : DerivativeOrderMax = 0, 
                  p_value_prev : ValuePrev = False,
                  p_visualize : bool = False ):

        Plottable.__init__(self, p_visualize=p_visualize)

        self.name                   = p_name
        self._value                 = None
        self._value_prev            = None
        self._time_stamp            = None
        self._time_stamp_prev       = None
        self._derivative_order_max  = p_derivative_order_max
        self._sw_value_prev         = p_
        self._derivatives           = {}
        self._derivatives_prev      = {}


## -------------------------------------------------------------------------------------------------
    def _get(self):
        return self._value
    

## -------------------------------------------------------------------------------------------------
    def _get_prev(self):
        return self._value_prev
    

## -------------------------------------------------------------------------------------------------
    def set(self, p_value, p_time_stamp : Union[datetime, int, float] = None): 
        """
        Sets the value of a property at a given time point.

        Parameters:
        -----------
        p_value 
            Value of the property of any type (numeric, vectorial, textual, list, dict, ...).
            In case of auto-derivation only lists, numpy arrays and scalar numbers are supported. Lists
            are converted to numpy arrays.
        p_time_stamp : : Union[datetime, int, float]
            Optional time stamp of type datetime, int or float. If not provided, an internal continuous
            integer time stamp is generated.
        """

        # 1 Set value
        if self._sw_value_prev: self._value_prev = self._value
        self._value      = p_value

    
        # 2 Preparation of time stamp
        self._time_stamp_prev = self._time_stamp

        if p_time_stamp is None:
            try:
                self._time_stamp = self._time_stamp_prev + 1
            except:
                self._time_stamp = 0
        else:
            self._time_stamp = p_time_stamp


        # 3 Numeric derivation
        if self._derivative_order_max > 0:

            # 3.1 Computation of time delta
            if self._time_stamp_prev is not None:

                delta_t = self._time_stamp - self._time_stamp_prev
            
                try: 
                    delta_t = delta_t.total_seconds()
                except:
                    pass

            
            # 3.2 Derivation
            self._derivatives_prev = self._derivatives
            self._derivatives      = {}
            
            if np.isscalar(p_value):
                self._derivatives[0]  = p_value
            else:
                self._value           = np.array( p_value )
                self._derivatives[0]  = self._value

            for order in range(self._derivative_order_max):
                try:
                    self._derivatives[order+1] = ( self._derivatives[order] - self._derivatives_prev[order] ) / delta_t
                except:
                    break


## -------------------------------------------------------------------------------------------------
    def _get_dim(self) -> int:
        """
        Internal method to determine the dimensionality of the currently stored values. It is used
        implicitely when accessing attribute 'dim'.
        """

        if self._value is None: return None
        try:
            # Try to treat the stored values as a numpy array
            return self._value.size
        except:
            try:
                # Try to treat the values as a python list
                return len(self._value)
            except:
                # Obviously a one-dimensional value
                return 1
            

## -------------------------------------------------------------------------------------------------
    def _get_derivatives(self):
        return self._derivatives


## -------------------------------------------------------------------------------------------------
    def _get_time_stamp(self):
        return self._time_stamp
                

## -------------------------------------------------------------------------------------------------
    value       = property( fget = _get, fset = set)
    value_prev  = property( fget = _get_prev )
    dim         = property( fget = _get_dim )
    time_stamp  = property( fget = _get_time_stamp )
    derivatives = property( fget = _get_derivatives )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Properties (Plottable, Renormalizable):
    """
    Inherit from this class to add MLPro's property systematics to your own class. It enables
    defining properties of any type. See class Property for further details.

    Parameters
    ----------
    p_properties : PropertyDefinitions
        List of property definitions. 
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    
    Attributes
    ----------
    C_PROPERTIES : PropertyDefinitions
        List of property definitions to be created on instantiation
    """

    C_PROPERTIES : PropertyDefinitions = []
    
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_properties : PropertyDefinitions = [],
                  p_visualize : bool = False ):
                     
        self._properties = {}
        self.add_properties( p_property_definitions = self.C_PROPERTIES, p_visualize = p_visualize )
        self.add_properties( p_property_definitions = p_properties, p_visualize = p_visualize )

        Plottable.__init__( self, p_visualize = p_visualize )
                  

## -------------------------------------------------------------------------------------------------
    def add_property( self, 
                      p_name : PropertyName, 
                      p_derivative_order_max : DerivativeOrderMax = 0, 
                      p_value_prev : ValuePrev = False,
                      p_cls : PropertyClass = Property,
                      p_visualize : bool = False ):
        """
        Adds a new managed property as an attribute to the own class. Optionally, auto-derivation can 
        be added for numeric properties (scalar or vectorial). The property is stored in the protected 
        dictionary self._properties and can be accessed directly via self.[p_name]. 

        Parameters
        ----------
        p_name : PropertyName
            Name of the property. Add a leading '_' to the name to make the related attribute protected.
        p_derivative_order_max : DerivativeOrderMax
            Maximum order of auto-generated derivatives. Default = 0 (no auto-derivation).
        p_value_prev : bool
            If True, the previous value is stored in value_prev whenever value is updated.
        p_cls : PropertyClass
            Optional property class to be used. Default = Property.
        p_visualize : bool
            Boolean switch for visualisation. Default = False.
        """

        prop_obj = p_cls( p_name = p_name, 
                          p_derivative_order_max = p_derivative_order_max, 
                          p_value_prev = p_value_prev,
                          p_visualize = p_visualize )
        self._properties[p_name] = prop_obj
        setattr(self, p_name, prop_obj )


## -------------------------------------------------------------------------------------------------
    def add_properties( self, 
                        p_property_definitions : PropertyDefinitions,
                        p_visualize : bool = False ):
        """
        Adds new managed properties to the own class. See method add_property() for further details.
            
        Parameters
        ----------
        p_property_definitions : PropertyDefinitions
            List of property definitions.
        p_visualize : bool
            Boolean switch for visualisation. Default = False.
        """

        for p in p_property_definitions:
            self.add_property( p_name = p[0], 
                               p_derivative_order_max = p[1], 
                               p_value_prev = p[2],
                               p_cls = p[3], 
                               p_visualize = p_visualize )


## -------------------------------------------------------------------------------------------------
    def get_properties(self):
        """
        Returns the dictionary of currently stored propery objects.

        Returns
        -------
        dict
            Dictionary of property objects.
        """

        return self._properties


## -------------------------------------------------------------------------------------------------
    def set_plot_settings(self, p_plot_settings : PlotSettings ):
        
        Plottable.set_plot_settings( self, p_plot_settings = p_plot_settings )
        
        for prop in self.get_properties().values():
            prop.set_plot_settings( p_plot_settings = p_plot_settings )

            
## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        Plottable.init_plot(self, p_figure = p_figure, p_plot_settings = p_plot_settings )

        for prop in self.get_properties().values():
            prop.init_plot( p_figure = p_figure, p_plot_settings = p_plot_settings)


## -------------------------------------------------------------------------------------------------
    def update_plot( self, **p_kwargs ):

        if not self.get_visualization(): return

        Plottable.update_plot(self, **p_kwargs )

        for prop in self.get_properties().values():
            prop.update_plot(**p_kwargs)


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh:bool = True):

        if not self.get_visualization(): return

        for prop in self.get_properties().values():
            prop.remove_plot( p_refresh = False)

        Plottable.remove_plot(self, p_refresh = p_refresh )
            

## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer : Normalizer ):

        for prop in self.get_properties().values():
            prop.renormalize( p_normalizer = p_normalizer )