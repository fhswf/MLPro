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
## -- 2024-05-31  1.0.0     DA       New class MultiProperty
## -- 2024-06-03  1.0.1     DA       Method Properties.update_plot(): changed order of plotting
## -- 2024-06-05  1.1.0     DA       New method Properties.replace_property()
## -- 2024-06-06  1.2.0     DA       New custom method Properties._update_property_links()
## -- 2024-06-16  1.3.0     DA       New method Properties.get_property_definitions()
## -- 2024-06-26  1.4.0     DA       New method Properties.set_plot_color()
## -- 2024-06-30  1.5.0     DA       Method Property.set(): new parameters p_upd_time_stamp, 
## --                                p_upd_derivatives
## -- 2026-07-08  1.6.0     DA       Introduction of kwargs
## -- 2024-07-27  1.7.0     DA       Class Property: introduction of self._value_bak
## -- 2024-12-11  1.7.1     DA       Pseudo class Figure if matplotlib is not installed
## -- 2025-03-19  1.7.2     DA       Corrections in method Property.set()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.7.2 (2025-03-19)

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

try:
    from matplotlib.figure import Figure
except:
    class Figure: pass

from mlpro.bf.various import KWArgs
from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.math.normalizers import Normalizer, Renormalizable



# Export list for public API
__all__ = [ 'Property',
            'Properties',
            'MultiProperty',
            'PropertyName',
            'DerivativeOrderMax',
            'ValuePrev',
            'PropertyClass',
            'PropertyDefinition',
            'PropertyDefinitions' ]





# Type aliases for property definitions
PropertyName        = str
DerivativeOrderMax  = int
ValuePrev           = bool
PropertyClass       = type

PropertyDefinition  = Tuple[ PropertyName, DerivativeOrderMax, ValuePrev, PropertyClass ]
PropertyDefinitions = List[ PropertyDefinition ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Property (Plottable, Renormalizable, KWArgs):
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
    p_kwargs : dict
        Keyword parameters.

    Attributes
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
                  p_visualize : bool = False,
                  **p_kwargs ):

        Plottable.__init__(self, p_visualize=p_visualize)
        KWArgs.__init__(self, **p_kwargs)

        self.name                   = p_name
        self._value                 = None
        self._value_bak             = None
        self._value_prev            = None
        self._time_stamp            = None
        self._time_stamp_prev       = None
        self._derivative_order_max  = p_derivative_order_max
        self._sw_value_prev         = p_value_prev
        self._derivatives           = {}
        self._derivatives_prev      = {}


## -------------------------------------------------------------------------------------------------
    def _get(self):
        return self._value
    

## -------------------------------------------------------------------------------------------------
    def _get_prev(self):
        return self._value_prev
    

## -------------------------------------------------------------------------------------------------
    def set( self, 
             p_value, 
             p_time_stamp : Union[datetime, int, float] = None,
             p_upd_time_stamp : bool = True,
             p_upd_derivatives : bool = True ): 
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
        p_upd_time_stamp : bool
            Boolean switch to enable/disable updating the inner time stamps.
        p_upd_derivatives : bool
            Boolean swtich to enable/disable updating the derivatives.
        """

        # 1 Update value
        if self._sw_value_prev:

            try:
                self._value_prev = self._value_bak.copy()
                self._value_bak  = p_value.copy()
            except:
                self._value_prev = self._value_bak
                self._value_bak  = p_value

        try:
            self._value = p_value.copy()
        except:
            self._value = p_value


        # 2 Update time stamps and derivatives
        if p_upd_time_stamp and ( p_time_stamp is not None ):

            # 2.1 Update time stamps
            self._time_stamp_prev = self._time_stamp
            self._time_stamp      = p_time_stamp


            # 2.2 Numeric derivation
            if p_upd_derivatives and ( self._derivative_order_max > 0 ):

                # 2.2.1 Computation of time delta
                try:
                    delta_t = self._time_stamp - self._time_stamp_prev
                
                    try: 
                        delta_t = delta_t.total_seconds()
                    except:
                        pass
                    
                except:
                    return

                
                # 2.2.2 Derivation
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
    p_kwargs : dict
        Keyword parameters to be handed over to all properties.

    Attributes
    ----------
    C_PROPERTIES : PropertyDefinitions
        List of property definitions to be created on instantiation
    """

    C_PROPERTIES : PropertyDefinitions = []
    
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_properties : PropertyDefinitions = [],
                  p_visualize : bool = False,
                  **p_kwargs ):
                     
        self._properties = {}
        self._property_definitions = {}
        self.add_properties( p_property_definitions = self.C_PROPERTIES, p_visualize = p_visualize, **p_kwargs )
        self.add_properties( p_property_definitions = p_properties, p_visualize = p_visualize, **p_kwargs )
        self._update_property_links()

        Plottable.__init__( self, p_visualize = p_visualize )
                  

## -------------------------------------------------------------------------------------------------
    def add_property( self, 
                      p_name : PropertyName, 
                      p_derivative_order_max : DerivativeOrderMax = 0, 
                      p_value_prev : ValuePrev = False,
                      p_cls : PropertyClass = Property,
                      p_visualize : bool = False,
                      **p_kwargs ):
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
        p_kwargs : dict
            Optional keyword parameters.
        """

        prop_obj = p_cls( p_name = p_name, 
                          p_derivative_order_max = p_derivative_order_max, 
                          p_value_prev = p_value_prev,
                          p_visualize = p_visualize,
                          **p_kwargs )
        self._properties[p_name] = (prop_obj, False)
        self._property_definitions[p_name] = ( p_name, p_derivative_order_max, p_value_prev, p_cls )
        setattr(self, p_name, prop_obj )


## -------------------------------------------------------------------------------------------------
    def add_properties( self, 
                        p_property_definitions : PropertyDefinitions,
                        p_visualize : bool = False,
                        **p_kwargs ):
        """
        Adds new managed properties to the own class. See method add_property() for further details.
            
        Parameters
        ----------
        p_property_definitions : PropertyDefinitions
            List of property definitions.
        p_visualize : bool
            Boolean switch for visualisation. Default = False.
        p_kwargs : dict
            Keyword parameters to be handed over to all properties.        
        """

        for p in p_property_definitions:
            self.add_property( p_name = p[0], 
                               p_derivative_order_max = p[1], 
                               p_value_prev = p[2],
                               p_cls = p[3], 
                               p_visualize = p_visualize,
                               **p_kwargs )


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
    def get_property_definitions(self) -> PropertyDefinitions:
        """
        Returns a list of currently stored property definitions.

        Returns
        -------
        PropertyDefinitions
            List of property definitions
        """

        return list(self._property_definitions.values())
    

## -------------------------------------------------------------------------------------------------
    def replace_property(self, p_property : Property ):
        """
        This method (re-)assigns the top-level attibute self.[p_property.name] with the property object
        p_property.

        Parameters
        ----------
        p_property : Property
            New property object to be assigned.
        """

        setattr(self, p_property.name, p_property)
        self._properties[p_property.name] = (p_property, False)
        self._update_property_links()


## -------------------------------------------------------------------------------------------------
    def _link_property(self, p_attr : str, p_prop : Property):
        """
        This method enables internal linking of properties. This is helpful if key information of
        deep properties shall be provided as top-level attributes in self. After calling this method
        an attributeself.[p_attr] is available that is linked to the sub-attribute p_prop.[p_attr]. 
        A former attribute self.[attr] is overwritten.

        Parameters
        ----------
        p_attr : str
            Name of the top level attribute linked to a deeper sub-property of property p_prop.
        p_prop : Property
            Deep property with a sub-property named p_attr.
        """

        attr_src  = getattr(p_prop, p_attr)
        attr_dest = getattr(self, p_attr)
        attr_src._derivative_order_max = attr_dest._derivative_order_max
        setattr(self, p_attr, attr_src)

        # Mark top-level attribute as link
        self._properties[p_attr] = (attr_dest, True)


## -------------------------------------------------------------------------------------------------
    def _update_property_links(self):
        """
        Custom method to define internal property links. Use method _link_property() to describe
        link relations. This method is automatically called by the contructor and method replace_property().
        """

        pass
    

## -------------------------------------------------------------------------------------------------
    def set_plot_settings(self, p_plot_settings : PlotSettings ):
        
        Plottable.set_plot_settings( self, p_plot_settings = p_plot_settings )
        
        for (prop, link) in self.get_properties().values():
            if not link:
                prop.set_plot_settings( p_plot_settings = p_plot_settings )


## -------------------------------------------------------------------------------------------------
    def set_plot_color(self, p_color : str):

        Plottable.set_plot_color( self, p_color = p_color )

        for (prop, link) in self.get_properties().values():
            if not link:
                prop.set_plot_color( p_color = p_color )     


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        Plottable.init_plot(self, p_figure = p_figure, p_plot_settings = p_plot_settings )
        try:
            if not self._plot_initialized: return
        except:
            return

        for (prop, link) in self.get_properties().values():
            if not link: 
                prop.init_plot( p_figure = self._figure, p_plot_settings = p_plot_settings)


## -------------------------------------------------------------------------------------------------
    def update_plot( self, **p_kwargs ):

        if not self.get_visualization(): return

        if ( self._plot_settings.detail_level > 0 ) and ( self._plot_settings.detail_level < self.plot_detail_level ): return

        for (prop, link) in self.get_properties().values():
            if not link: 
                prop.update_plot(**p_kwargs)

        Plottable.update_plot(self, **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh:bool = True):

        if not self.get_visualization(): return

        for (prop, link) in self.get_properties().values():
            if not link: prop.remove_plot( p_refresh = False)

        Plottable.remove_plot(self, p_refresh = p_refresh )
            

## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer : Normalizer ):

        for (prop, link) in self.get_properties().values():
            if not link: prop.renormalize( p_normalizer = p_normalizer )

                 
## -------------------------------------------------------------------------------------------------
    color = property( fget = Plottable.get_plot_color, fset = set_plot_color )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiProperty (Property, Properties):

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name : str,
                  p_derivative_order_max: int = 0, 
                  p_value_prev : ValuePrev = False,
                  p_properties : PropertyDefinitions = [],
                  p_visualize: bool = False,
                  **p_kwargs ):
        
        Property.__init__( self, 
                           p_name,
                           p_derivative_order_max = p_derivative_order_max, 
                           p_value_prev = p_value_prev,
                           p_visualize = p_visualize,
                           **p_kwargs )
        
        Properties.__init__( self,
                             p_properties = p_properties,
                             p_visualize = p_visualize,
                             **p_kwargs )