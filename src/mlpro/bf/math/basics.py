## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : math
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.3.0 (2024-12-02)

This module provides basic mathematical classes.
"""


import numpy as np
from itertools import repeat
import uuid
from mlpro.bf.various import Log, KWArgs
from mlpro.bf.events import *
from typing import Union





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
class Function:
    """
    Model class for an elementary bi-multivariate mathematical function that maps elements of a
    multivariate input space to elements of a multivariate output space.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_input_space: MSpace, p_output_space: MSpace, p_output_elem_cls=Element):
        """
        Parameters:
            p_input_space       Input space
            p_output_space      Output space
            p_output_elem_cls   Output element class (compatible to class Element)
        """

        self._input_space     = p_input_space
        self._output_space    = p_output_space
        self._output_elem_cls = p_output_elem_cls
        self.__call__         = self.map 


## -------------------------------------------------------------------------------------------------
    def __call__(self, p_input: Union[Element,np.array] ) -> Union[Element,np.array]:
        """
        Maps a multivariate abscissa/input element to a multivariate ordinate/output element by 
        calling the custom method _map().

        Parameters
        ----------
        p_input : Element
            Input element to be mapped.
        
        Returns
        -------
        output : Element
            Output element.
        """

        return self.map(p_input=p_input)


## -------------------------------------------------------------------------------------------------
    def map(self, p_input: Union[Element,np.array] ) -> Union[Element,np.array]:
        """
        Maps a multivariate abscissa/input element to a multivariate ordinate/output element by 
        calling the custom method _map().        

        Parameters
        ----------
        p_input : Union[Element,np.array]
            Input to be mapped.
        
        Returns
        -------
        output : Union[Element,np.array]
            Output.
        """

        output = self._output_elem_cls(self._output_space)
        self._map(p_input, output)
        return output


## -------------------------------------------------------------------------------------------------
    def _map(self, p_input: Element, p_output: Element):
        """
        Custom method for own mapping algorithm. See methods __call__() and map() for further details.
        """
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def map_inverse(self, p_input: Union[Element,np.array] ) -> Union[Element,np.array]:
        """
        Inverse mapping by calling custom method _map_inverse().
        """

        output = self._output_elem_cls(self._output_space)
        self._map_inverse(p_input, output)
        return output


## -------------------------------------------------------------------------------------------------
    def _map_inverse(self, p_input: Element, p_output: Element):
        """
        Custom method for own inverse mapping algorithm. See method map_inverse() for further details.
        """
        
        raise NotImplementedError