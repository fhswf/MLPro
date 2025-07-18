## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-23  0.0.0     DA       Creation 
## -- 2021-05-30  1.0.0     DA       Release of first version
## -- 2021-08-26  1.1.0     DA       Class Dimension extended by base set and description
## -- 2021-09-11  1.1.0     MRD      Change Header information to match our new library name
## -- 2021-09-23  1.2.0     DA       Changes to deal with big data objects:
## --                                - new class DataObject
## --                                - new base set type 'D' in class Dimension
## --                                - changes in class Element: list instead of np.array
## -- 2021-10-25  1.3.0     DA       New class Function
## -- 2021-12-03  1.3.1     DA       New methods Dimension.copy(), Set.copy(), Set.append()
## -- 2021-12-03  1.3.2     MRD      Fix Set.append() due to the usage of max() on empty list
## -- 2022-01-21  1.4.0     DA       New class TrendAnalyzer
## -- 2022-02-25  1.4.1     SY       Class Dimension extended by auto generated ID
## -- 2022-09-11  1.5.0     DA       - Class Dimension: new method set_boundaries (event)
## --                                - Class TrendAnalyzer removed
## --                                - Code reformatting
## -- 2022-10-06  1.5.1     DA       Class Dimension: event C_EVENT_BOUNDARIES converted to string
## -- 2022-10-08  1.6.0     DA       New method Set.get_dims()
## -- 2022-10-21  1.7.0     DA       Class Dimension: extension by optional property symmetry
## -- 2022-10-24  1.8.0     DA       Class Element: new method copy()
## -- 2022-12-05  1.9.0     DA       Class Dimension: new param p_kwargs and method get_kwargs()
## -- 2022-12-09  2.0.0     DA       Class Set: 
## --                                - new method get_dim_by_name()
## --                                - internal optimizations
## -- 2022-12-13  2.1.0     DA       Class Element:
## --                                - new method set_related_set()
## --                                - internal optimizations
## --                                Class Set:
## --                                - new method is_numeric()
## -- 2023-02-28  2.2.0     DA       Class Function: new method __call__()
## -- 2023-03-07  2.2.1     SY       Refactoring
## -- 2023-04-09  2.2.2     SY       Refactoring
## -- 2023-05-06  2.2.3     DA       Class Element: completion of data type definitions
## -- 2024-12-02  2.3.0     DA       Class Dimension: new parent KWArgs
## -- 2025-07-07  3.0.0     DA       - New data type Data
## --                                - Class Function: refactoring and extension
## --                                - New class Scaler
## -- 2025-07-11  3.0.1     DA       Bugfix in method Scaler.rescale()
## -------------------------------------------------------------------------------------------------

"""
Ver. 3.0.1 (2025-07-11)

This module provides basic mathematical classes.
"""


import numpy as np
from itertools import repeat
import uuid

from mlpro.bf.various import Log, KWArgs, ScientificObject
from mlpro.bf.events import *
from mlpro.bf.exceptions import ParamError
from typing import Union



# Export list for public API
__all__ = [ 'Dimension',
            'Set',
            'DataObject',
            'Element',
            'ElementList',
            'BatchElement',
            'MSpace',
            'ESpace',
            'Function',
            'Data',
            'Scaler' ]





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Dimension (EventManager, KWArgs):
    """
    Objects of this type specify properties of a dimension of a set.

    Parameters:
    -----------
    p_name_short : str
        Short name of dimension
    p_base_set 
        Base set of dimension. See constants C_BASE_SET_*. Default = C_BASE_SET_R.
    p_name_long :str
        Long name of dimension (optional)
    p_name_latex : str
        LaTeX name of dimension (optional)
    p_unit : str
        Unit (optional)
    p_unit_latex : str
        LaTeX code of unit (optional)
    p_boundaries : list
        List with minimum and maximum value (optional)
    p_description : str
        Description of dimension (optional)
    p_symmetrical : bool
        Information about the symmetry of the dimension (optional, default is False)
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional keyword parameters.
    """

    C_TYPE              = 'Dimension'

    C_BASE_SET_R        = 'R'           # real numbers
    C_BASE_SET_N        = 'N'           # natural numbers
    C_BASE_SET_Z        = 'Z'           # integer numbers
    C_BASE_SET_DO       = 'DO'          # (big) data objects (like images, point clouds, ...)

    C_EVENT_BOUNDARIES  = 'BOUNDARIES'  # raised by method set_boundaries()

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name_short, 
                  p_base_set=C_BASE_SET_R, 
                  p_name_long='', 
                  p_name_latex='', 
                  p_unit='',
                  p_unit_latex='', 
                  p_boundaries:list=[], 
                  p_description='',
                  p_symmetrical:bool=False,
                  p_logging=Log.C_LOG_NOTHING,
                  **p_kwargs ):

        KWArgs.__init__(self, **p_kwargs)
        EventManager.__init__(self, p_logging=p_logging)

        self._id = str(uuid.uuid4())
        self._name_short    = self.C_NAME = p_name_short
        self._base_set      = p_base_set
        self._name_long     = p_name_long
        self._name_latex    = p_name_latex
        self._unit          = p_unit
        self._unit_latex    = p_unit_latex
        self._description   = p_description
        self._symmetrical   = p_symmetrical

        self.set_boundaries(p_boundaries=p_boundaries)


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self._id


## -------------------------------------------------------------------------------------------------
    def get_name_short(self):
        return self._name_short


## -------------------------------------------------------------------------------------------------
    def get_base_set(self):
        return self._base_set


## -------------------------------------------------------------------------------------------------
    def get_name_long(self):
        return self._name_long


## -------------------------------------------------------------------------------------------------
    def get_name_latex(self):
        return self._name_latex


## -------------------------------------------------------------------------------------------------
    def get_unit(self):
        return self._unit


## -------------------------------------------------------------------------------------------------
    def get_unit_latex(self):
        return self._unit_latex


## -------------------------------------------------------------------------------------------------
    def get_boundaries(self):
        return self._boundaries


## -------------------------------------------------------------------------------------------------
    def set_boundaries(self, p_boundaries:list):
        """
        Sets new boundaries with respect to the symmmetry and raises event C_EVENT_BOUNDARIES.

        Parameters
        ----------
        p_boundaries : list
            New boundaries (lower and upper value)
        """

        self._boundaries = p_boundaries.copy()
        
        if ( self._symmetrical ) and ( len(self._boundaries)== 2 ):
            abs_low  = abs(self._boundaries[0])
            abs_high = abs(self._boundaries[1])
            if abs_high > abs_low:
                self._boundaries[0] = - abs_high
            else:
                self._boundaries[1] = abs_low

        self._raise_event( p_event_id=self.C_EVENT_BOUNDARIES, p_event_object=Event(p_raising_object=self, p_boundaries=p_boundaries) )


## -------------------------------------------------------------------------------------------------
    def get_description(self):
        return self._description


## -------------------------------------------------------------------------------------------------
    def get_symmetrical(self) -> bool:
        return self._symmetrical


## -------------------------------------------------------------------------------------------------
    def get_kwargs(self) -> dict:
        """
        Returns all keyword arguments provided during initialization as a dictionary. Alternatively,
        public attribute kwargs can be used directly.
        """
        return self.kwargs


## -------------------------------------------------------------------------------------------------
    def copy(self):
        return self.__class__( p_name_short=self._name_short,
                               p_base_set=self._base_set,
                               p_name_long=self._name_long,
                               p_name_latex=self._name_latex,
                               p_unit=self._unit,
                               p_unit_latex=self._unit_latex,
                               p_boundaries=self._boundaries,
                               p_description=self._description,
                               p_symmetrical=self._symmetrical,
                               p_logging=self.get_log_level(),
                               **self.kwargs )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Set:
    """
    Objects of this type describe a (multivariate) set in a mathematical sense.
    """

    C_NUMERIC_BASE_SETS     = [ Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z, Dimension.C_BASE_SET_R ]

## -------------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self._dim_by_id     = {}
        self._dim_by_name   = {}
        self._is_numeric    = True


## -------------------------------------------------------------------------------------------------
    def add_dim(self, p_dim:Dimension, p_ignore_duplicates:bool=False ):
        """
        Raises the dimensionality of the set by adding a new dimension.

        Parameters
        ----------
        p_dim : Dimension
            Dimension to be added.
        p_ignore_duplicates : bool
            If True, duplicated short names of dimensions are accepted. Default = False.
        """

        # 1 Check, whether a dimension with same name was already added
        name_short = p_dim.get_name_short()

        if not p_ignore_duplicates:
            try:
                dim = self._dim_by_name[name_short]
            except:
                pass
            else:
                raise ParamError('Dimension "' + name_short + '" already exists!')


        # 2 Store new dimension under it's id and name
        self._dim_by_name[name_short]   = p_dim
        self._dim_by_id[p_dim.get_id()] = p_dim


        # 3 Update numeric-flag
        self._is_numeric = self._is_numeric and ( p_dim.get_base_set() in self.C_NUMERIC_BASE_SETS )


## -------------------------------------------------------------------------------------------------
    def is_numeric(self) -> bool:
        """
        Returns True if the set consists of numeric dimensions only.
        """

        return self._is_numeric


## -------------------------------------------------------------------------------------------------
    def get_dim(self, p_id) -> Dimension:
        """
        Returns the dimension specified by it's unique id.
        """

        return self._dim_by_id[p_id]


## -------------------------------------------------------------------------------------------------
    def get_dim_by_name(self, p_name) -> Dimension:
        return self._dim_by_name[p_name]


## -------------------------------------------------------------------------------------------------
    def get_dims(self) -> list:
        """"
        Returns all dimensions.
        """

        return list(self._dim_by_id.values())


## -------------------------------------------------------------------------------------------------
    def get_num_dim(self):
        """
        Returns the dimensionality of the set (=number of dimensions of the set).
        """

        return len(self.get_dims())


## -------------------------------------------------------------------------------------------------
    def get_dim_ids(self):
        """
        Returns the unique ids of the related dimensions.
        """

        return list(self._dim_by_id.keys())


## -------------------------------------------------------------------------------------------------
    def spawn(self, p_id_list:list):
        """
        Spawns a new class with same type and a subset of dimensions specified
        by an index list.

        Parameters:
            p_id_list       List of indices of dimensions to be adopted

        Returns:
            New object with subset of dimensions
        """

        new_set = self.__class__()
        for dim_id in p_id_list:
            new_set.add_dim(self._dim_by_id[dim_id])

        return new_set


## -------------------------------------------------------------------------------------------------
    def copy(self, p_new_dim_ids:bool=True):
        new_set = self.__class__()

        if p_new_dim_ids:
            for dim in self._dim_by_id.values():
                new_set.add_dim(p_dim=dim.copy())
        else:
            for dim in self._dim_by_id.values():
                new_set.add_dim(p_dim=dim)

        return new_set


## -------------------------------------------------------------------------------------------------
    def append(self, p_set, p_new_dim_ids:bool=True, p_ignore_duplicates:bool=False):
        if p_new_dim_ids:
            for dim in p_set.get_dims():
                self.add_dim(p_dim=dim.copy(), p_ignore_duplicates=p_ignore_duplicates)
        else:
            for dim in p_set.get_dims():
                self.add_dim(p_dim=dim, p_ignore_duplicates=p_ignore_duplicates)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataObject:
    """
    Container class for (big) data objects of any type with optional additional meta data.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_data, *p_meta_data) -> None:
        self._data = p_data
        self._meta_data = p_meta_data


## -------------------------------------------------------------------------------------------------
    def get_data(self):
        return self._data


## -------------------------------------------------------------------------------------------------
    def get_meta_data(self) -> tuple:
        return self._meta_data





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Element:
    """
    Element of a (multivariate) set.

    Parameters
    ----------
    p_set : Set
        Underlying set.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_set:Set):
        self.set_related_set(p_set=p_set)
        if p_set.is_numeric():
            self._values = np.zeros(self._set.get_num_dim())
        else:
            self._values = list(repeat(0, self._set.get_num_dim()))
        

## -------------------------------------------------------------------------------------------------
    def get_related_set(self) -> Set:
        return self._set


## -------------------------------------------------------------------------------------------------
    def set_related_set(self, p_set:Set):
        self._set = p_set


## -------------------------------------------------------------------------------------------------
    def get_dim_ids(self) -> list:
        return self._set.get_dim_ids()


## -------------------------------------------------------------------------------------------------
    def get_values(self) -> Union[list,np.ndarray]:
        return self._values


## -------------------------------------------------------------------------------------------------
    def set_values(self, p_values : Union[list, np.ndarray]):
        """
        Overwrites the values of all components of the element.

        Parameters:
            p_values        Something iterable with same length as number of element dimensions.
        """

        self._values = p_values


## -------------------------------------------------------------------------------------------------
    def get_value(self, p_dim_id):
        return self._values[self._set.get_dim_ids().index(p_dim_id)]


## -------------------------------------------------------------------------------------------------
    def set_value(self, p_dim_id, p_value):
        try:
            self._values[self._set.get_dim_ids().index(p_dim_id)] = p_value
        except:
            self._values = self._values.tolist()
            self._values[self._set.get_dim_ids().index(p_dim_id)] = p_value
            self._values = np.array(self._values, dtype=object)


## -------------------------------------------------------------------------------------------------
    def copy(self):
        duplicate = self.__class__(p_set=self._set)
        duplicate.set_values(p_values=self._values.copy())
        return duplicate





Data = Union[ float, list, np.ndarray, Element ]





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ElementList:
    """
    List of Element objects.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, ):
        self._elem_list = []
        self._elem_ids = []


## -------------------------------------------------------------------------------------------------
    def add_elem(self, p_id, p_elem: Element):
        """
        Adds an element object under it's id in the internal element list.

        Parameters:
            p_id        Unique id of the element
            p_elem      Element object to be added
        """

        self._elem_ids.append(p_id)
        self._elem_list.append(p_elem)


## -------------------------------------------------------------------------------------------------
    def get_elem_ids(self) -> list:
        return self._elem_ids


## -------------------------------------------------------------------------------------------------
    def get_elem(self, p_id) -> Element:
        return self._elem_list[self._elem_ids.index(p_id)]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BatchElement(Element):


    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MSpace(Set):
    """
    Objects of this type represent a metric space. The method distance implements the metric of the 
    space.
    """

## -------------------------------------------------------------------------------------------------
    def distance(self, p_e1: Element, p_e2: Element):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ESpace(MSpace):
    """
    Objects of this type represent an Euclidian space. The distance method
    implements the Euclidian norm.
    """

## -------------------------------------------------------------------------------------------------
    def distance(self, p_e1: Element, p_e2: Element):
        return np.sum((np.array(p_e1.get_values()) - np.array(p_e2.get_values())) ** 2) ** 0.5





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Function (KWArgs, ScientificObject):
    """
    Model class for an elementary bi-multivariate mathematical function that maps elements of a
    multivariate input space to elements of a multivariate output space.

    Parameters
    ----------
    p_input_set : Set = None
        Optional input set, needed for the mapping of objects of type Element.
    p_output_set : Set = None
        Optional output set, needed for the mapping of objects of type Element.
    p_output_elem_cls : type = Element  
        Output element class (compatible to class Element)
    p_autocreate_elements : bool = True
        If True, elements of the output space are created automatically during mapping of objects of 
        type Element.
    **p_kwargs
        Further optional keyword arguments needed for particular custom implementations.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_input_set : Set = None, 
                  p_output_set : Set = None,
                  p_input_space = None,       # hidden parameter ensuring backward compatibility
                  p_output_space = None,      # hidden parameter ensuring backward compatibility
                  p_output_elem_cls : type = Element,
                  p_autocreate_elements : bool = True,
                  **p_kwargs ):
        
        KWArgs.__init__(self, **p_kwargs )

        if ( p_input_set is None ):
            self._input_set = p_input_space
        else:
            self._input_set = p_input_set

        if ( p_output_set is None ):
            self._output_set = p_output_space
        else:
            self._output_set = p_output_set

        # Obsolete private attibutes, kept for backward compatibility. Do not use in new implementations.
        self._input_space         = self._input_set
        self._output_space        = self._output_set

        self._output_elem_cls     = p_output_elem_cls
        self._autocreate_elements = p_autocreate_elements

        if self._autocreate_elements and ( self._output_set is None ) and ( self._output_elem_cls is None ):
            raise ParamError('For element auto-creation the output set and type of element needs to be supplied')

        # Hard redirection of method self.__call__() to method self.map
        self.__call__ = self.map


## -------------------------------------------------------------------------------------------------
    def map( self, 
             p_input : Data,
             p_output : Data = None,
             p_dim : int = None ) -> Data:
        """
        Maps an input to an output by calling the custom methods _map_[Type]().        

        Parameters
        ----------
        p_input : Data
            Input to be mapped.
        p_output : Data = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        Data
            Result of the mapping.
        """

        try:
            meth_name = '_map_' + type(p_input).__name__
            method    = getattr(self, meth_name)
            return method( p_input = p_input, 
                           p_output = p_output, 
                           p_dim = p_dim )
        
        except AttributeError:
            # Treat parameters as of type Element...
            output = p_output
            if ( p_output is None ) and self._autocreate_elements:
                output = self._output_elem_cls(self._output_set)

            return self._map( p_input = p_input, 
                              p_output = output, 
                              p_dim = p_dim )


## -------------------------------------------------------------------------------------------------
    def __call__(self, p_input : Data ) -> Data:
        """
        Simplified pythonic mapping. Enables calls like y = my_fct(x). See the method map() for
        further details.

        Parameters
        ----------
        p_input : Data
            Input to be mapped.
        
        Returns
        -------
        Data
            Result of the mapping.
        """

        # This code is never executed due of the hard redirection to the method map() in the
        # constructor. Redefinition has no effect.
        pass


## -------------------------------------------------------------------------------------------------
    def _map( self, 
              p_input: Element, 
              p_output: Element = None,
              p_dim : int = None ) -> Element:
        """
        Default custom method for own mappings of single objects of type Element.

        Parameters
        ----------
        p_input : Element
            Input to be mapped.
        p_output : Element = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        Element
            Result of the mapping.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _map_float( self, 
                    p_input: float, 
                    p_output: float = None,
                    p_dim : int = None ) -> Element:
        """
        Custom method for own mappings of single floats.

        Parameters
        ----------
        p_input : Element
            Input to be mapped.
        p_output : Element = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        float
            Result of the mapping.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _map_list( self, 
                   p_input: list, 
                   p_output: list = None,
                   p_dim : int = None ) -> list:
        """
        Custom method for own mass mappings of lists.

        Parameters
        ----------
        p_input : list
            Input to be mapped.
        p_output : list = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        list
            Result of the mapping.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _map_ndarray( self, 
                      p_input: np.ndarray, 
                      p_output: np.ndarray = None,
                      p_dim : int = None ) -> np.ndarray:
        """
        Custom method for own mass mappings of Numpy arrays.

        Parameters
        ----------
        p_input : list
            Input to be mapped.
        p_output : list = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        np.ndarray
            Result of the mapping.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def map_inverse( self, 
                     p_input : Data,
                     p_output : Data = None,
                     p_dim : int = None ) -> Data:
        """
        Inverse mapping of an input to an output by calling the custom methods _map_inverse_[Type]().        

        Parameters
        ----------
        p_input : Data
            Input to be mapped.
        p_output : Data = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        Data
            Result of the mapping.
        """

        try:
            meth_name = '_map_inverse_' + type(p_input).__name__
            method    = getattr(self, meth_name)
            return method( p_input = p_input, 
                           p_output = p_output, 
                           p_dim = p_dim )
        
        except AttributeError:
            # Treat parameters as of type Element...
            output = p_output
            if ( p_output is None ) and self._autocreate_elements:
                output = self._output_elem_cls(self._input_set)

            return self._map_inverse( p_input = p_input, 
                                      p_output = output, 
                                      p_dim = p_dim )


## -------------------------------------------------------------------------------------------------
    def _map_inverse( self, 
                      p_input: Element, 
                      p_output: Element = None,
                      p_dim : int = None ) -> Element:
        """
        Default custom method for own inverse mappings of single objects of type Element.

        Parameters
        ----------
        p_input : Element
            Input to be mapped.
        p_output : Element = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        Element
            Result of the mapping.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _map_inverse_float( self, 
                            p_input: float, 
                            p_output: float = None,
                            p_dim : int = None ) -> Element:
        """
        Custom method for own inverse mappings of single floats.

        Parameters
        ----------
        p_input : Element
            Input to be mapped.
        p_output : Element = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        float
            Result of the mapping.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _map_inverse_list( self, 
                           p_input: list, 
                           p_output: list = None,
                           p_dim : int = None ) -> list:
        """
        Custom method for own mass inverse mappings of lists.

        Parameters
        ----------
        p_input : list
            Input to be mapped.
        p_output : list = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        list
            Result of the mapping.
        """
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def _map_inverse_ndarray( self, 
                              p_input: np.ndarray, 
                              p_output: np.ndarray = None,
                              p_dim : int = None ) -> np.ndarray:
        """
        Custom method for own mass inverse mappings of Numpy arrays.

        Parameters
        ----------
        p_input : list
            Input to be mapped.
        p_output : list = None
            Optional output object as the intended receiver of the mapping result.
        p_dim : int = None
           An optional dimension index to which the mapping is restricted.
        
        Returns
        -------
        np.ndarray
            Result of the mapping.
        """
        
        raise NotImplementedError
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Scaler (Function):
    """
    Template for scaler algorithms scaling, unscaling, rescaling data. This class introduces a
    parameter handling based on the three attributes _param, _param_old, and _param_new. All custom
    methods inherited from the class Function shall apply the parameters stored in _param, which is 
    a reference to eigther _param_old or _param_new.  

    Parameters
    ----------
    See class Function for further details.

    Attributes
    ----------
    _param
        Internal reference to the active set of function parameters applied to the next scaler action.
    _param_old
        Previous parameter set.
    _param_new
        Current parameter set.   
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_input_set : Set = None, 
                  p_output_set : Set = None,
                  p_input_space = None,       # hidden parameter ensuring backward compatibility
                  p_output_space = None,      # hidden parameter ensuring backward compatibility
                  p_output_elem_cls : type = Element,
                  p_autocreate_elements : bool = True,
                  **p_kwargs ):
        
        super(). __init__( p_input_set = p_input_set, 
                           p_output_set = p_output_set,
                           p_input_space = p_input_space,        # hidden parameter ensuring backward compatibility
                           p_output_space = p_output_space,      # hidden parameter ensuring backward compatibility
                           p_output_elem_cls = p_output_elem_cls,
                           p_autocreate_elements = p_autocreate_elements,
                           **p_kwargs )

        self._param     = None
        self._param_old = None
        self._param_new = None


## -------------------------------------------------------------------------------------------------
    def scale( self, 
               p_data : Data, 
               p_dim : int = None,
               p_param = None ) -> Data:
        """
        Scales the specified data.

        Parameters
        ----------
        p_data : Data
            Data to be scaled.
        p_dim : int = None
            Optional index of the dimension to be scaled.
        p_param = None
            Optional parameter set to be applied to the scaling operation. If None the set stored in
            self._param_new is used.

        Returns
        -------
        Data
            The scaled data.
        """

        if p_param is not None:
            self._set_parameters( p_param = p_param )
        else:
            self._set_parameters( p_param = self._param_new )

        return self.map( p_input = p_data, p_dim = p_dim )


## -------------------------------------------------------------------------------------------------
    def unscale( self, 
                 p_data : Data, 
                 p_dim : int = None,
                 p_param = None ) -> Data:
        """
        Unscales the specified data.

        Parameters
        ----------
        p_data : Data
            Data to be unscaled.
        p_dim : int = None
            Optional index of the dimension to be unscaled.
        p_param = None
            Optional parameter set to be applied to the unscaling operation. If None the set stored in
            self._param_new is used.

        Returns
        -------
        Data
            The unscaled data.
        """

        if p_param is not None:
            self._set_parameters( p_param = p_param )
        else:
            self._set_parameters( p_param = self._param_new )

        return self.map_inverse( p_input = p_data, p_dim = p_dim )


## -------------------------------------------------------------------------------------------------
    def rescale( self, 
                 p_data : Data, 
                 p_dim : int = None,
                 p_param_old = None,
                 p_param_new = None ) -> Data:
        """
        Rescales the specified data by unscaling them with previous parameters stored in _param_old and
        scaling them with the current parameters in _param_new.

        Parameters
        ----------
        p_data : Data
            Data to be rescaled.
        p_dim : int = None
            Optional index of the dimension to be rescaled.
        p_param_old = None
            Optional parameter set to be applied to the unscaling operation. If None the set stored in
            self._param_old is used.
        p_param_new = None
            Optional parameter set to be applied to the scaling operation. If None the set stored in
            self._param_new is used.

        Returns
        -------
        Data
            The rescaled data.
        """

        if ( self._param_old is None ) and ( p_param_old is None ): return p_data

        param_old = p_param_old if p_param_old is not None else self._param_old
        param_new = p_param_new if p_param_new is not None else self._param_new
        
        return self.scale( p_data = self.unscale( p_data = p_data, 
                                                  p_dim = p_dim,
                                                  p_param = param_old ), 
                           p_dim = p_dim,
                           p_param = param_new )


## -------------------------------------------------------------------------------------------------
    def _set_parameters( self, p_param ):
        """
        Private service method to activate the parameter set suitable for the next scaler operation.

        Parameters
        ----------
        p_param 
            Parameter set to be activated.
        """

        self._param = p_param


## -------------------------------------------------------------------------------------------------
    def update_parameters( self, **p_kwargs ) -> bool:
        """
        Method to update the parameters of the scaler. It calls the custom method _update_parameters(),
        which specifies the actual parameters needed by the particular algorithm.

        Parameters
        ----------
        p_data : Data
            Data needed to update the parameters of the scaler.

        Returns
        -------
        bool
            True, if the parameters were changed. False otherwise.
        """

        return self._update_parameters( **p_kwargs )
    

## -------------------------------------------------------------------------------------------------
    def _update_parameters( self, **p_kwargs ) -> bool:
        """
        Custom method to update the parameters of the scaler based on data specific to the particular 
        algorithm. Please set the internal attribute p_param_new with new values and backup the previous content in
        p_param_old before.

        Parameters
        ----------
        p_data : Data
            Data needed to update the parameters of the scaler.

        Returns
        -------
        bool
            True, if the parameters were changed. False otherwise.
        """

        raise NotImplementedError