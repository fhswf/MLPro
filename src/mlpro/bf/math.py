## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2021-09-23)

This module provides basic mathematical classes.
"""


import numpy as np
from itertools import repeat




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataObject:
    """
    Container class for (big) data objects of any type with optional additional meta data.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_data, *p_meta_data) -> None:
        self._data      = p_data
        self._meta_data = p_meta_data


## -------------------------------------------------------------------------------------------------
    def get_data(self):
        return self._data


## -------------------------------------------------------------------------------------------------
    def get_meta_data(self) -> tuple:
        return self._meta_data





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Dimension:
    """
    Objects of this type specify properties of a dimension of a set.
    """

    C_BASE_SET_R    = 'R'       # real numbers
    C_BASE_SET_N    = 'N'       # natural numbers 
    C_BASE_SET_Z    = 'Z'       # integer numbers
    C_BASE_SET_D    = 'D'       # big data objects (like images, point clouds, ...)

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id, p_name_short, p_base_set=C_BASE_SET_R, p_name_long='', p_name_latex='', p_unit='', p_unit_latex='', p_boundaries=[], p_description='') -> None:
        """
        Parameters:
            p_id                Id of the dimension that is unique in the 
                                related set
            p_name_short        Short name of dimension
            p_base_set          Base set (real numbers by default)
            p_name_long         Long name of dimension (optional)
            p_name_latex        LaTeX name of dimension (optional)
            p_unit              Unit (optional)
            p_unit_latex        LaTeX code of unit (optional)
            p_boundaries        List with minimum and maximum value (optional)
            p_description       Description of dimension (optional)
        """

        self._id            = p_id
        self._name_short    = p_name_short
        self._base_set      = p_base_set
        self._name_long     = p_name_long
        self._name_latex    = p_name_latex
        self._unit          = p_unit 
        self._unit_latex    = p_unit_latex
        self._boundaries    = p_boundaries
        self._description   = p_description


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
    def get_description(self):
        return self._description





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Set:
    """
    Objects of this type describe a (multivariate) set in a mathematical sense.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self._dim_list       = []
        self._dim_ids        = []


## -------------------------------------------------------------------------------------------------
    def add_dim(self, p_dim:Dimension):
        """
        Raises the dimensionality of the set by adding a new dimension.

        Parameters:
            p_dim       Dimension to be added
        """

        self._dim_ids.append(p_dim.get_id())
        self._dim_list.append(p_dim)


## -------------------------------------------------------------------------------------------------
    def get_dim(self, p_id) -> Dimension:
        """
        Returns the dimension specified by it's unique id.
        """

        return self._dim_list[self._dim_ids.index(p_id)]


## -------------------------------------------------------------------------------------------------
    def get_num_dim(self):
        """
        Returns the dimensionality of the set (=number of dimensions of the set).
        """

        return len(self._dim_list)


## -------------------------------------------------------------------------------------------------
    def get_dim_ids(self):
        """
        Returns the unique ids of the related dimensions.
        """

        return self._dim_ids


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
        for i in p_id_list:
            new_set.add_dim(self._dim_list[self._dim_ids.index(i)])
 
        return new_set





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Element:
    """
    Element of a (multivariate) set.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_set:Set) -> None:
        self._set       = p_set
        self._values    = list(repeat(0,self._set.get_num_dim()))


## -------------------------------------------------------------------------------------------------
    def get_related_set(self) -> Set:
        return self._set


## -------------------------------------------------------------------------------------------------
    def get_dim_ids(self) -> list:
        return self._set.get_dim_ids()


## -------------------------------------------------------------------------------------------------
    def get_values(self):
        return self._values


## -------------------------------------------------------------------------------------------------
    def set_values(self, p_values):
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
        self._values[self._set.get_dim_ids().index(p_dim_id)] = p_value





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ElementList:
    """
    List of Element objects.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, ):
        self._elem_list  = []
        self._elem_ids   = []


## -------------------------------------------------------------------------------------------------
    def add_elem(self, p_id, p_elem:Element):
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
class MSpace(Set):
    """
    Objects of this type represent a metric space. The method distance implements the metric of the 
    space.
    """

## -------------------------------------------------------------------------------------------------
    def distance(self, p_e1:Element, p_e2:Element):
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
        return np.sum( ( np.array(p_e1.get_values()) - np.array(p_e2.get_values()) )**2 )**0.5
